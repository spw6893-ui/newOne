#!/usr/bin/env python3
"""
AlphaGen（子模块 alphagen/）加密货币因子挖掘训练入口。

重要说明（避免踩坑）：
1) 仓库里同时存在 `alphagen/`（子模块）与 `AlphaQCM/alphagen`（历史目录），
   必须保证 Python 导入时优先使用子模块 alphagen 的实现。
2) 如果你希望把宽表里“所有因子列”都纳入 AlphaGen 的特征空间，
   需要在导入 alphagen 前动态构造 FeatureType（枚举）= 所有可用特征列。
"""

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch


repo_root = Path(__file__).resolve().parent
# 导入优先级：
# 1) alphagen 子模块（repo_root/alphagen）用于 AlphaGen 核心实现
# 2) AlphaQCM 作为“包”使用（repo_root 默认已在 sys.path），只通过 `AlphaQCM.alphagen_qlib` 引用适配层
sys.path.insert(0, str(repo_root / "alphagen"))


@dataclass(frozen=True)
class FeatureSpace:
    feature_cols: List[str]


def _read_csv_header(fp: Path) -> List[str]:
    with fp.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [h.strip() for h in header if h is not None]


def _detect_feature_space(data_dir: Path) -> FeatureSpace:
    """
    从 alphagen_ready/*.csv 的表头推断“特征列全集”。

    规则：
    - 排除元数据：datetime/symbol/split
    - 排除 label：y_*
    - 其余列默认都作为特征列（包含 open/high/low/close/volume 以及你工程生成的高阶因子）
    """
    files = sorted(data_dir.glob("*_train.csv"))
    if not files:
        raise FileNotFoundError(f"未找到训练数据：{data_dir}（期望 *_train.csv）")

    exclude = {"datetime", "symbol", "split"}
    exclude_env = os.environ.get("ALPHAGEN_EXCLUDE_COLS", "").strip()
    exclude_cols = {c.strip() for c in exclude_env.split(",") if c.strip()} if exclude_env else set()

    # 优先使用 prepare_alphagen_training_data.py 写出的 schema.json（若存在）
    schema_fp = data_dir / "schema.json"
    if schema_fp.exists():
        try:
            obj = json.loads(schema_fp.read_text(encoding="utf-8"))
            cols = obj.get("columns", [])
            header = [str(c).strip() for c in cols if str(c).strip()]
        except Exception:
            header = []
    else:
        header = []

    # 如果没有 schema.json，则用扫描表头的方式构造特征列集合
    # mode=union（默认）：把所有币种出现过的列都纳入（“全因子”）
    # mode=intersection：只保留所有币种都具备的列（显著减少 NaN，适合先跑通训练）
    mode = os.environ.get("ALPHAGEN_FEATURE_SCHEMA_MODE", "union").strip().lower()
    if not header:
        ordered: List[str] = []
        seen = set()
        per_file_sets: List[set[str]] = []
        for fp in files:
            h = _read_csv_header(fp)
            cols = [c for c in h if c and (c not in exclude) and (not c.startswith("y_")) and (c not in exclude_cols)]
            per_file_sets.append(set(cols))
            for c in cols:
                if c not in seen:
                    seen.add(c)
                    ordered.append(c)
        if mode == "intersection":
            inter = set.intersection(*per_file_sets) if per_file_sets else set()
            header = [c for c in ordered if c in inter]
        else:
            header = ordered

    feature_cols = [c for c in header if c and (c not in exclude) and (not c.startswith("y_")) and (c not in exclude_cols)]
    if not feature_cols:
        raise RuntimeError(f"未能从表头推断出任何特征列：{data_dir}")

    return FeatureSpace(feature_cols=feature_cols)


def _cs_mean_pearson_ic(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    计算“按时点的截面皮尔逊相关(IC)”，并对所有时点取均值。

    x/y: shape=(time, symbols)，允许 NaN/inf，自动忽略无效值。
    """
    if x.shape != y.shape:
        raise ValueError(f"x/y 形状不一致: {tuple(x.shape)} vs {tuple(y.shape)}")
    # mask=True 表示无效值（不参与统计）
    mask = (~torch.isfinite(x)) | (~torch.isfinite(y))
    n = (~mask).sum(dim=1)  # (time,)

    # 至少需要 2 个样本才能算相关
    valid = n >= 2
    if not bool(valid.any()):
        return float("nan")

    n_safe = torch.clamp(n, min=1)
    x0 = x.clone()
    y0 = y.clone()
    x0[mask] = 0.0
    y0[mask] = 0.0

    mean_x = x0.sum(dim=1) / n_safe
    mean_y = y0.sum(dim=1) / n_safe
    xc = (x0 - mean_x[:, None]) * (~mask)
    yc = (y0 - mean_y[:, None]) * (~mask)
    var_x = (xc * xc).sum(dim=1) / n_safe
    var_y = (yc * yc).sum(dim=1) / n_safe
    std_x = torch.sqrt(var_x)
    std_y = torch.sqrt(var_y)

    # std=0 的时点视为无效（价格常数/全相等等会出现）
    valid = valid & (std_x > 0) & (std_y > 0)
    if not bool(valid.any()):
        return float("nan")

    corr = (xc * yc).sum(dim=1) / (n_safe * std_x * std_y)
    ic = corr[valid].mean().item()
    return float(ic)


def _select_top_features_by_ic(
    data: torch.Tensor,
    feature_cols: Sequence[str],
    k: int,
    corr_threshold: float,
    ensure_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    用训练集做单变量 IC 打分并选 Top-K，同时用 mutual-IC 做去冗余（近似“高度相关特征剔除”）。

    - IC 口径：每个时点做截面相关（跨币种），再对时点取均值。
    - mutual-IC：同样口径对 (feature_i, feature_j) 做相关，用于剔除 |corr|>=阈值 的冗余特征。
    """
    if k <= 0:
        return list(feature_cols)
    if data.ndim != 3:
        raise ValueError(f"data 期望 shape=(time, features, symbols)，但得到: {tuple(data.shape)}")
    if len(feature_cols) != int(data.shape[1]):
        raise ValueError(f"feature_cols 数量与 data features 维不一致: {len(feature_cols)} vs {int(data.shape[1])}")

    ensure = [c for c in (ensure_cols or []) if c]
    ensure_set = set(ensure)

    if "close" not in feature_cols:
        raise RuntimeError("特征列里缺少 close，无法计算 forward return 作为 target")
    close_idx = list(feature_cols).index("close")
    close = data[:, close_idx, :]  # (time, symbols)
    # forward 1h return: close[t+1]/close[t]-1
    y = close[1:, :] / close[:-1, :] - 1.0

    scores: List[tuple[str, float]] = []
    time_aligned = slice(0, -1)
    for j, col in enumerate(feature_cols):
        x = data[time_aligned, j, :]
        ic = _cs_mean_pearson_ic(x, y)
        if not np.isfinite(ic):
            ic = 0.0
        scores.append((str(col), float(ic)))

    # 按 |IC| 排序（绝对值越大越好）
    scores.sort(key=lambda t: abs(t[1]), reverse=True)

    # 贪心去冗余：从高分到低分依次尝试加入
    corr_threshold = float(corr_threshold)
    corr_threshold = max(0.0, min(1.0, corr_threshold))

    kept: List[str] = []

    # 先把 ensure_cols 放进去（顺序保持），避免 target 依赖列丢失
    for c in ensure:
        if c in feature_cols and c not in kept:
            kept.append(c)

    for col, _ic in scores:
        if col in ensure_set:
            continue
        if col in kept:
            continue
        if len(kept) >= k:
            break

        # 与已保留特征计算 mutual-IC，过高则跳过
        j = list(feature_cols).index(col)
        x = data[time_aligned, j, :]
        redundant = False
        for exist in kept:
            jj = list(feature_cols).index(exist)
            xx = data[time_aligned, jj, :]
            mic = _cs_mean_pearson_ic(x, xx)
            if np.isfinite(mic) and abs(float(mic)) >= corr_threshold:
                redundant = True
                break
        if not redundant:
            kept.append(col)

    # 如果 ensure 占满了 k，也允许超过 k（因为 close 必须有）；否则补足到至少包含 ensure
    if "close" in feature_cols and "close" not in kept:
        kept = ["close"] + kept

    # 最终保持原始列名（snake_case），并保证都在 feature_cols 内
    out = [c for c in kept if c in feature_cols]
    # 兜底：至少返回 close
    if not out:
        out = ["close"]
    return out


def _install_dynamic_feature_type(feature_cols: Sequence[str]) -> None:
    """
    动态构造 alphagen_qlib.stock_data.FeatureType，使得 AlphaGen 能把"宽表全部因子列"当作可选 Feature。

    注意：必须在导入 `alphagen.data.tokens` / `alphagen.rl.env.wrapper` 之前调用，
    因为它们会在 import 时读取 `len(FeatureType)` 来构建 action space。
    """
    from enum import IntEnum

    import alphagen_qlib.stock_data as sd

    members = {}
    for i, col in enumerate(feature_cols):
        name = col.upper()
        # 防御：列名如果意外包含非法字符，做一次保守归一化
        #（正常情况下你的列名都是 snake_case，不会触发）
        name = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
        if not name or name[0].isdigit():
            name = f"F_{name}"
        # 去重
        base = name
        k = 2
        while name in members:
            name = f"{base}_{k}"
            k += 1
        members[name] = i

    sd.FeatureType = IntEnum("FeatureType", members)  # type: ignore[attr-defined]
    # 同时暴露"列顺序"，供 CryptoData 做 index->列名映射
    sd.FEATURE_COLUMNS = list(feature_cols)  # type: ignore[attr-defined]


def _dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    # ==================== 配置参数 ====================

    # 数据配置
    # 训练数据默认落在 AlphaQCM/AlphaQCM_data/alphagen_ready（由 run_training.sh / prepare_alphagen_training_data.py 生成）
    DATA_DIR = os.environ.get("ALPHAGEN_DATA_DIR", "AlphaQCM/AlphaQCM_data/alphagen_ready")
    SYMBOLS = os.environ.get("ALPHAGEN_SYMBOLS", "top100")  # 或指定列表: ['BTCUSDT', 'ETHUSDT', ...]

    # 时间分割
    START_TIME = os.environ.get("ALPHAGEN_START_TIME", "2020-01-01")
    TRAIN_END = os.environ.get("ALPHAGEN_TRAIN_END", "2024-01-01")
    VAL_END = os.environ.get("ALPHAGEN_VAL_END", "2024-07-01")
    END_TIME = os.environ.get("ALPHAGEN_END_TIME", "2025-02-15")

    # 特征：默认把 alphagen_ready 里的"全部因子列"都扔进 AlphaGen（FeatureType 动态构造）
    # 可选：用训练集的单变量 IC 做预筛选，从而缩小 action space（更容易探索/更快收敛）
    feature_space = _detect_feature_space(Path(DATA_DIR))
    feature_list_loaded = False

    # 重要：为保证 resume 时 action_space 完全一致，优先从上一次输出的 selected_features.json 恢复特征列表。
    # 否则哪怕只差 1~2 个特征/子表达式，SB3 也会因为 observation_space 不一致而拒绝加载。
    resume_flag_early = os.environ.get("ALPHAGEN_RESUME", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    resume_path_early = os.environ.get("ALPHAGEN_RESUME_PATH", "").strip()
    resume_step_early = None
    if resume_flag_early and resume_path_early:
        # 兼容：model_step_12345.zip 或 .../model_step_12345.zip
        name = Path(resume_path_early).name
        if name.startswith("model_step_") and name.endswith(".zip"):
            s = name[len("model_step_") : -len(".zip")]
            try:
                resume_step_early = int(s)
            except Exception:
                resume_step_early = None

    force_load_feat_raw = os.environ.get("ALPHAGEN_FEATURE_LIST_LOAD", "auto").strip().lower()
    force_load_feat = force_load_feat_raw in {"1", "true", "yes", "y", "on"}
    auto_load_feat = force_load_feat_raw in {"auto", ""}
    default_feat_path = Path("./alphagen_output") / "selected_features.json"
    # 优先：如果 resume 且当前 checkpoint 目录下存在 features_step_{step}.json，则用它确保严格一致
    if resume_step_early is not None:
        cand = Path("./alphagen_output") / "checkpoints" / f"features_step_{resume_step_early}.json"
        if cand.exists():
            default_feat_path = cand
    feat_list_path = Path(os.environ.get("ALPHAGEN_FEATURE_LIST_PATH", str(default_feat_path)).strip() or str(default_feat_path))
    if (force_load_feat or (auto_load_feat and resume_flag_early)) and feat_list_path.exists():
        try:
            obj = json.loads(feat_list_path.read_text(encoding="utf-8"))
            feats = obj.get("features", obj.get("feature_cols", obj.get("cols", [])))
            if isinstance(feats, list):
                feats = [str(x).strip() for x in feats if str(x).strip()]
                # 只保留当前数据中存在的列，避免旧文件污染
                feats = [c for c in feats if c in feature_space.feature_cols]
                if "close" in feature_space.feature_cols and "close" not in feats:
                    feats = ["close"] + feats
                if feats:
                    feature_space = FeatureSpace(feature_cols=feats)
                    feature_list_loaded = True
                    print(f"✓ 已从文件恢复特征列表: {feat_list_path}（n={len(feature_space.feature_cols)}）")
        except Exception as e:
            print(f"⚠ 读取特征列表失败（将忽略）：{feat_list_path} err={e}")

    features_max = int(os.environ.get("ALPHAGEN_FEATURES_MAX", "0").strip() or 0)
    prune_corr = float(os.environ.get("ALPHAGEN_FEATURES_PRUNE_CORR", "0.95").strip() or 0.95)
    # 如果已经从文件恢复特征列表，则跳过 IC 预筛选（否则会破坏 resume 的 action_space 一致性）
    if (not feature_list_loaded) and features_max > 0:
        # 先用“全特征”构造 FeatureType，加载一次数据做打分，然后再用筛选后的特征重建 FeatureType。
        _install_dynamic_feature_type(feature_space.feature_cols)
        from AlphaQCM.alphagen_qlib.crypto_data import CryptoData

        print(f"计算特征 IC 以做预筛选: topK={features_max}, prune_corr={prune_corr}")
        score_data = CryptoData(
            symbols=SYMBOLS,
            start_time=START_TIME,
            end_time=TRAIN_END,
            timeframe="1h",
            data_dir=DATA_DIR,
            max_backtrack_periods=100,
            max_future_periods=30,
            features=None,
            device=torch.device("cpu"),
        )
        selected_cols = _select_top_features_by_ic(
            data=score_data.data.detach().cpu(),
            feature_cols=feature_space.feature_cols,
            k=features_max,
            corr_threshold=prune_corr,
            ensure_cols=["close"],
        )
        feature_space = FeatureSpace(feature_cols=selected_cols)
        print(f"预筛选后特征数: {len(feature_space.feature_cols)}")
        print(f"预筛选特征列表: {feature_space.feature_cols}")
        # 记录到输出目录，方便复现实验
        try:
            out_dir = Path("./alphagen_output")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "selected_features.json").write_text(
                json.dumps({"features": feature_space.feature_cols}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"✓ 预筛选结果已保存: {out_dir / 'selected_features.json'}")
        except Exception as e:
            print(f"⚠ 保存预筛选结果失败: {e}")

    _install_dynamic_feature_type(feature_space.feature_cols)
    subexprs_max = int(os.environ.get("ALPHAGEN_SUBEXPRS_MAX", "0").strip() or 0)
    if subexprs_max < 0:
        subexprs_max = 0

    # resume 时同样需要固定 subexpr 映射：优先加载 alphagen_output/subexprs.json 以保证动作索引一致。
    # 这里先读取“原始字符串列表”，真正 parse 在完成动态 FeatureType 安装之后进行（见后文）。
    subexprs_loaded_texts = None
    force_load_subexpr_raw = os.environ.get("ALPHAGEN_SUBEXPRS_LOAD", "auto").strip().lower()
    force_load_subexpr = force_load_subexpr_raw in {"1", "true", "yes", "y", "on"}
    auto_load_subexpr = force_load_subexpr_raw in {"auto", ""}
    default_subexprs_path = Path("./alphagen_output") / "subexprs.json"
    if resume_step_early is not None:
        cand = Path("./alphagen_output") / "checkpoints" / f"subexprs_step_{resume_step_early}.json"
        if cand.exists():
            default_subexprs_path = cand
    subexprs_path = Path(os.environ.get("ALPHAGEN_SUBEXPRS_PATH", str(default_subexprs_path)).strip() or str(default_subexprs_path))
    if (force_load_subexpr or (auto_load_subexpr and resume_flag_early)) and subexprs_path.exists():
        try:
            obj = json.loads(subexprs_path.read_text(encoding="utf-8"))
            raw = obj.get("subexprs", [])
            if isinstance(raw, list):
                raw = [str(x).strip() for x in raw if str(x).strip()]
                if raw:
                    subexprs_loaded_texts = raw
                    # resume 时必须保证动作空间一致：优先使用文件长度
                    env_subexprs_max = int(os.environ.get("ALPHAGEN_SUBEXPRS_MAX", "0").strip() or 0)
                    if resume_flag_early:
                        if env_subexprs_max > 0 and env_subexprs_max != len(raw):
                            print(
                                f"⚠ 检测到 resume + 子表达式数量不一致："
                                f"ALPHAGEN_SUBEXPRS_MAX={env_subexprs_max} vs file={len(raw)}，"
                                f"将强制使用 file={len(raw)} 以保证空间一致。"
                            )
                        subexprs_max = len(raw)
                    else:
                        # 非 resume：若外部没显式指定 subexprs_max，则以文件长度为准
                        if env_subexprs_max <= 0:
                            subexprs_max = len(raw)
                    print(f"✓ 已从文件恢复子表达式列表: {subexprs_path}（n={len(raw)}）")
        except Exception as e:
            print(f"⚠ 读取子表达式列表失败（将忽略）：{subexprs_path} err={e}")
    # alphagen wrapper 的 state dtype 默认是 uint8，因此 action_space 不能超过 255
    # action_space 大小约等于 len(features) + 常量/算子开销（约 42）+ subexprs
    if len(feature_space.feature_cols) + 42 + subexprs_max > 255:
        raise RuntimeError(
            f"action_space 过大：features={len(feature_space.feature_cols)}, subexprs_max={subexprs_max}。"
            f"这会导致 AlphaGen action_space>255（uint8 溢出）。请减少特征列/子表达式，或改造 alphagen wrapper dtype。"
        )

    # 现在再 import alphagen（确保 action space 读到的是动态 FeatureType）
    # 注意：当前 alphagen 版本没有 Close()/Open() 这类快捷构造器，使用 Feature(FeatureType.X) 即可。
    from alphagen.data.expression import Corr, Cov, Delta, EMA, Feature, Mean, Rank, Ref, Std, WMA
    # 兼容：alphagen 上游 rolling Std/Var 在窗口=1 时会触发 dof<=0 警告并产生 NaN（unbiased=True 的默认行为）。
    # 这里做一次运行时 monkey patch，避免需要修改 submodule 指针（否则会导致他人无法拉取特定 commit）。
    import alphagen.data.expression as _expr_mod
    def _std_apply_unbiased_false(self, operand):  # type: ignore[no-redef]
        return operand.std(dim=-1, unbiased=False)
    def _var_apply_unbiased_false(self, operand):  # type: ignore[no-redef]
        return operand.var(dim=-1, unbiased=False)
    _expr_mod.Std._apply = _std_apply_unbiased_false  # type: ignore[assignment]
    _expr_mod.Var._apply = _var_apply_unbiased_false  # type: ignore[assignment]
    from alphagen.models.linear_alpha_pool import MeanStdAlphaPool, MseAlphaPool
    from alphagen.rl.env.wrapper import AlphaEnv
    import alphagen.rl.env.wrapper as env_wrapper
    from alphagen.rl.policy import LSTMSharedNet
    from alphagen.utils import reseed_everything
    from sb3_contrib.ppo_mask import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.callbacks import CallbackList

    from AlphaQCM.alphagen_qlib.crypto_data import CryptoData
    from AlphaQCM.alphagen_qlib.calculator import QLibStockDataCalculator, TensorQLibStockDataCalculator
    import alphagen_qlib.stock_data as sd

    import alphagen as alphagen_pkg
    alphagen_file = getattr(alphagen_pkg, "__file__", None)
    alphagen_path = list(getattr(alphagen_pkg, "__path__", []))
    # namespace package 场景下 __file__ 可能为 None，用 __path__ 更可靠
    print(f"alphagen 包路径: {alphagen_file if alphagen_file else alphagen_path}")

    def _parse_int_list(raw: str, default: List[int]) -> List[int]:
        s = str(raw).strip()
        if not s:
            return default
        out: List[int] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(part))
            except Exception:
                continue
        return out or default

    def _build_subexpr_library(max_n: int) -> List["Expression"]:
        """
        构建一个“子表达式库”，作为额外动作（ExpressionToken）提供给 agent 复用。

        核心目的：减少逐 token 生成的难度，让 agent 更容易组合出有效结构，从而突破 IC 平台期。
        """
        if max_n <= 0:
            return []

        windows = _parse_int_list(os.environ.get("ALPHAGEN_SUBEXPRS_WINDOWS", "5,10,20,40"), [5, 10, 20, 40])
        dts = _parse_int_list(os.environ.get("ALPHAGEN_SUBEXPRS_DTS", "1,2,4,8"), [1, 2, 4, 8])
        windows = [w for w in windows if w > 1]
        dts = [d for d in dts if d > 0]
        enable_rank_raw = os.environ.get("ALPHAGEN_SUBEXPRS_ENABLE_RANK", "1").strip().lower()
        enable_rank = enable_rank_raw in {"1", "true", "yes", "y", "on"}
        enable_pair_raw = os.environ.get("ALPHAGEN_SUBEXPRS_ENABLE_PAIRROLL", "1").strip().lower()
        enable_pair = enable_pair_raw in {"1", "true", "yes", "y", "on"}
        pair_max = int(os.environ.get("ALPHAGEN_SUBEXPRS_PAIR_MAX", "3").strip() or 3)
        pair_max = max(2, pair_max)

        # 子表达式库的“构成”非常关键：
        # 之前如果简单按顺序 add(Feature(...)) 再 add(Mean/Std/...)，
        # 一旦 max_n 较小（例如 30），就会被“原子 feature”占满，导致库里几乎没有任何可复用结构，
        # 实际上对探索帮助有限（看起来“开了 subexprs”，但效果接近没开）。
        #
        # 新策略：预算拆分
        # - 先放少量原子 feature（保证最基本可用）
        # - 优先放常用的 rolling/Ref/Delta 等结构（才是能“复用”的东西）
        # - 最后如果还有预算，再补更多原子 feature
        raw_max = int(os.environ.get("ALPHAGEN_SUBEXPRS_RAW_MAX", "10").strip() or 10)
        raw_max = max(0, raw_max)

        exprs: List["Expression"] = []
        seen = set()

        def add(e):
            k = str(e)
            if k in seen:
                return
            seen.add(k)
            exprs.append(e)

        # 1) 选择一小批“核心原子 feature”：close + 前 raw_max-1 个
        cols = list(feature_space.feature_cols)
        core_cols: List[str] = []
        if "close" in cols:
            core_cols.append("close")
        for c in cols:
            if c == "close":
                continue
            if len(core_cols) >= max(1, raw_max):
                break
            core_cols.append(c)

        core_indices: List[int] = []
        for c in core_cols:
            try:
                core_indices.append(cols.index(c))
            except ValueError:
                continue

        # 2) 先放少量原子 feature（避免被原子塞满：只放 core）
        for i in core_indices:
            add(Feature(sd.FeatureType(i)))

        # 3) 优先放“可复用结构”：对 core feature 做 Ref/Delta/rolling
        for i in core_indices:
            base = Feature(sd.FeatureType(i))
            for dt in dts:
                add(Ref(base, dt))
                add(Delta(base, dt))
            for w in windows:
                add(Mean(base, w))
                add(Std(base, w))
                add(EMA(base, w))
                add(WMA(base, w))
                if enable_rank:
                    add(Rank(base, w))

        # 3.5) 结构增强：少量 PairRolling（Corr/Cov）子表达式（对突破平台期很关键）
        # 控制规模：只取前 pair_max 个 core feature，两两组合，并且只用前 2 个 window（避免爆炸）
        if enable_pair and (len(core_indices) >= 2) and windows:
            use_core = core_indices[: min(len(core_indices), int(pair_max))]
            use_windows = windows[:2]
            for ai in range(len(use_core)):
                for bi in range(ai + 1, len(use_core)):
                    lhs = Feature(sd.FeatureType(use_core[ai]))
                    rhs = Feature(sd.FeatureType(use_core[bi]))
                    for w in use_windows:
                        add(Corr(lhs, rhs, w))
                        add(Cov(lhs, rhs, w))

        # 4) 若还有预算，再补其它 feature 的少量 rolling（提升表达力，但避免爆炸）
        if len(exprs) < max_n:
            extra_indices = [i for i in range(len(cols)) if i not in set(core_indices)]
            extra_budget = max_n - len(exprs)
            # 每个 feature 最多补 2 个结构（优先 Mean/Std）
            per_feat = 2
            for i in extra_indices:
                if extra_budget <= 0:
                    break
                base = Feature(sd.FeatureType(i))
                if windows:
                    add(Mean(base, windows[0]))
                    extra_budget -= 1
                    if extra_budget <= 0:
                        break
                    add(Std(base, windows[0]))
                    extra_budget -= 1
                if extra_budget <= 0:
                    break
                # 兜底：如果 windows 为空，就补原子
                if not windows:
                    add(base)
                    extra_budget -= 1

        # 5) 最后，如果还有预算，再把原子 feature 补齐（用来提升组合多样性）
        if len(exprs) < max_n:
            for i in range(len(cols)):
                add(Feature(sd.FeatureType(i)))
                if len(exprs) >= max_n:
                    break

        # 截断到 max_n（保持确定性顺序）
        return exprs[:max_n]

    class TensorboardCallback(BaseCallback):
        """记录训练指标到TensorBoard"""

        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            return True

    class AlphaCacheStatsCallback(BaseCallback):
        """
        把 alpha 评估缓存命中率写入 TensorBoard（用于定位“越跑越慢”）。
        只有使用我们这边的 QLibStockDataCalculator / TensorQLibStockDataCalculator 才生效。
        """

        def __init__(self, calculator_obj, update_every: int = 2048, verbose: int = 0):
            super().__init__(verbose=verbose)
            self._calc = calculator_obj
            self._update_every = max(1, int(update_every))
            self._last_step = 0

        def _on_step(self) -> bool:
            if (int(self.num_timesteps) - int(self._last_step)) < int(self._update_every):
                return True
            self._last_step = int(self.num_timesteps)
            fn = getattr(self._calc, "alpha_cache_stats", None)
            if fn is None:
                return True
            try:
                st = fn()
            except Exception:
                return True
            self.logger.record("cache/alpha_cache_size", float(st.get("cache_size", 0)))
            self.logger.record("cache/alpha_cache_len", float(st.get("cache_len", 0)))
            self.logger.record("cache/alpha_cache_hits", float(st.get("hits", 0)))
            self.logger.record("cache/alpha_cache_misses", float(st.get("misses", 0)))
            denom = float(st.get("hits", 0) + st.get("misses", 0))
            hit_rate = float(st.get("hits", 0)) / denom if denom > 0 else 0.0
            self.logger.record("cache/alpha_cache_hit_rate", hit_rate)
            return True

    class PeriodicCheckpointCallback(BaseCallback):
        """
        周期性保存 checkpoint（模型 + alpha_pool），用于中断后恢复训练。

        注意：
        - SB3 的 `model.save(path)` 会自动添加 `.zip` 后缀。
        - 同步保存 `alpha_pool_step_{step}.json`，便于同时恢复 pool（否则 reward 会非平稳）。
        """

        def __init__(
            self,
            output_dir: Path,
            pool_obj,
            every_steps: int,
            keep_last: int = 3,
            feature_cols: Optional[List[str]] = None,
            subexpr_texts: Optional[List[str]] = None,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self._output_dir = Path(output_dir)
            self._pool = pool_obj
            self._every = max(1, int(every_steps))
            self._keep = max(1, int(keep_last))
            self._feature_cols = list(feature_cols) if feature_cols else None
            self._subexpr_texts = list(subexpr_texts) if subexpr_texts else None
            self._ckpt_dir = self._output_dir / "checkpoints"
            self._ckpt_dir.mkdir(parents=True, exist_ok=True)
            self._last_step = 0

        @staticmethod
        def _parse_step_from_name(name: str) -> Optional[int]:
            # model_step_12345.zip / alpha_pool_step_12345.json
            for prefix in ("model_step_", "alpha_pool_step_"):
                if name.startswith(prefix):
                    s = name[len(prefix):]
                    s = s.split(".", 1)[0]
                    try:
                        return int(s)
                    except Exception:
                        return None
            return None

        def _gc(self) -> None:
            items = []
            for fp in self._ckpt_dir.glob("model_step_*.zip"):
                step = self._parse_step_from_name(fp.name)
                if step is not None:
                    items.append((step, fp))
            items.sort(key=lambda x: x[0])
            if len(items) <= self._keep:
                return
            to_delete = items[:-self._keep]
            for step, model_fp in to_delete:
                try:
                    model_fp.unlink(missing_ok=True)
                except Exception:
                    pass
                pool_fp = self._ckpt_dir / f"alpha_pool_step_{step}.json"
                try:
                    pool_fp.unlink(missing_ok=True)
                except Exception:
                    pass
                # 附属快照一起清理（避免目录混杂导致误用）
                for extra in (
                    self._ckpt_dir / f"features_step_{step}.json",
                    self._ckpt_dir / f"subexprs_step_{step}.json",
                ):
                    try:
                        extra.unlink(missing_ok=True)
                    except Exception:
                        pass

        def _on_step(self) -> bool:
            # 注意：SB3 的 num_timesteps 通常按 n_steps 的倍数递增，
            # 如果用 `% every == 0`，当 every 不是 n_steps 的倍数时可能“永远不触发”。
            # 这里改成“跨过阈值就保存一次”，保证近似每 every_steps 保存。
            if (int(self.num_timesteps) - int(self._last_step)) < int(self._every):
                return True
            self._last_step = int(self.num_timesteps)
            step = int(self.num_timesteps)
            model_base = self._ckpt_dir / f"model_step_{step}"
            pool_fp = self._ckpt_dir / f"alpha_pool_step_{step}.json"
            try:
                self.model.save(str(model_base))
                _dump_json(pool_fp, self._pool.to_json_dict())
                # 同步保存 feature/subexpr 快照，便于未来严格 resume（避免 144 vs 142 这种空间不一致）
                if self._feature_cols is not None:
                    _dump_json(self._ckpt_dir / f"features_step_{step}.json", {"features": self._feature_cols})
                if self._subexpr_texts is not None:
                    _dump_json(self._ckpt_dir / f"subexprs_step_{step}.json", {"subexprs": self._subexpr_texts})
                self.logger.record("checkpoint/last_step", float(step))
            except Exception:
                # checkpoint 失败不应中断训练
                return True
            self._gc()
            return True

    class FastGateStatsCallback(BaseCallback):
        """
        把 FastGate 的粗筛统计写入 TensorBoard，便于判断：
        - 是否真的在“pool 满后”帮你省掉了大量评估开销
        - 阈值是否设得过高（skips 过多可能导致学习信号变稀疏）
        """

        def __init__(self, pool_obj, update_every: int = 2048, verbose: int = 0):
            super().__init__(verbose=verbose)
            self._pool = pool_obj
            self._update_every = max(1, int(update_every))
            self._last = {"calls": 0, "skips": 0, "time_s": 0.0}
            self._last_step = 0

        def _on_step(self) -> bool:
            if (int(self.num_timesteps) - int(self._last_step)) < int(self._update_every):
                return True
            self._last_step = int(self.num_timesteps)
            st = getattr(self._pool, "_fast_gate_stats", None)
            if not isinstance(st, dict):
                return True
            calls = int(st.get("calls", 0))
            skips = int(st.get("skips", 0))
            time_s = float(st.get("time_s", 0.0))

            dc = calls - int(self._last.get("calls", 0))
            ds = skips - int(self._last.get("skips", 0))
            dt = time_s - float(self._last.get("time_s", 0.0))

            self.logger.record("perf/fast_gate_calls", float(dc))
            self.logger.record("perf/fast_gate_skips", float(ds))
            self.logger.record("perf/fast_gate_time_s", float(dt))
            if dc > 0:
                self.logger.record("perf/fast_gate_ms_per_call", 1000.0 * float(dt) / float(dc))
                self.logger.record("perf/fast_gate_skip_rate", float(ds) / float(dc))

            self._last = {"calls": calls, "skips": skips, "time_s": time_s}
            return True

    class ValGateStatsCallback(BaseCallback):
        """
        把 ValGate（验证集小样本门控）的统计写入 TensorBoard，便于判断：
        - 是否真的减少了昂贵的 full-eval / optimize
        - 阈值是否过严（skips 过多会导致学习信号变稀疏）
        """

        def __init__(self, pool_obj, update_every: int = 2048, verbose: int = 0):
            super().__init__(verbose=verbose)
            self._pool = pool_obj
            self._update_every = max(1, int(update_every))
            self._last_step = 0
            self._last = {"calls": 0, "skips": 0, "time_s": 0.0}

        def _on_step(self) -> bool:
            if (int(self.num_timesteps) - int(self._last_step)) < int(self._update_every):
                return True
            self._last_step = int(self.num_timesteps)
            st = getattr(self._pool, "_val_gate_stats", None)
            if not isinstance(st, dict):
                return True
            calls = int(st.get("calls", 0))
            skips = int(st.get("skips", 0))
            time_s = float(st.get("time_s", 0.0))

            dc = calls - int(self._last.get("calls", 0))
            ds = skips - int(self._last.get("skips", 0))
            dt = time_s - float(self._last.get("time_s", 0.0))

            self.logger.record("perf/val_gate_calls", float(dc))
            self.logger.record("perf/val_gate_skips", float(ds))
            self.logger.record("perf/val_gate_time_s", float(dt))
            if dc > 0:
                self.logger.record("perf/val_gate_ms_per_call", 1000.0 * float(dt) / float(dc))
                self.logger.record("perf/val_gate_skip_rate", float(ds) / float(dc))

            self._last = {"calls": calls, "skips": skips, "time_s": time_s}
            return True

    class FastGateThresholdScheduleCallback(BaseCallback):
        """
        动态调整 FastGate 的粗筛阈值（min_abs_ic）：
        - 只影响“是否跳过完整评估”，不改变真实训练目标/IC 口径；
        - 常用做法：前期阈值低（保证探索），后期阈值升高（提升 skip_rate，缓解 fps 衰减）。
        """

        def __init__(
            self,
            total_timesteps: int,
            start_thr: float,
            end_thr: float,
            update_every: int,
            holder: dict,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self.total_timesteps = max(1, int(total_timesteps))
            self.start_thr = float(max(0.0, float(start_thr)))
            self.end_thr = float(max(0.0, float(end_thr)))
            self.update_every = max(1, int(update_every))
            self.holder = holder
            self._last_step = 0

        def _compute_thr(self) -> float:
            frac = min(1.0, float(self.num_timesteps) / float(self.total_timesteps))
            return self.start_thr + frac * (self.end_thr - self.start_thr)

        def _on_step(self) -> bool:
            if (int(self.num_timesteps) - int(self._last_step)) < int(self.update_every):
                return True
            self._last_step = int(self.num_timesteps)
            v = float(self._compute_thr())
            self.holder["value"] = float(v)
            self.logger.record("perf/fast_gate_min_abs_ic", float(v))
            return True

    class FastGateAutoTuneCallback(BaseCallback):
        """
        FastGate 自动调阈值：让粗筛 skip_rate 收敛到一个目标区间，从而“稳定” full-eval 频率。
        这通常比手工调阈值更稳，特别是当训练分布随时间变化时。
        """

        def __init__(
            self,
            pool_obj,
            holder: dict,
            target_skip_rate: float,
            adjust_mul: float,
            min_thr: float,
            max_thr: float,
            update_every: int = 2048,
            warmup_calls: int = 50,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self._pool = pool_obj
            self._holder = holder
            self._target = float(max(0.0, min(0.95, target_skip_rate)))
            self._mul = float(max(1.01, adjust_mul))
            self._min_thr = float(max(0.0, min_thr))
            self._max_thr = float(max(self._min_thr, max_thr))
            self._update_every = max(1, int(update_every))
            self._warmup_calls = max(10, int(warmup_calls))
            self._last = {"calls": 0, "skips": 0}
            self._last_step = 0

        def _on_step(self) -> bool:
            if (int(self.num_timesteps) - int(self._last_step)) < int(self._update_every):
                return True
            self._last_step = int(self.num_timesteps)
            st = getattr(self._pool, "_fast_gate_stats", None)
            if not isinstance(st, dict):
                return True
            calls = int(st.get("calls", 0))
            skips = int(st.get("skips", 0))
            dc = calls - int(self._last.get("calls", 0))
            ds = skips - int(self._last.get("skips", 0))
            self._last = {"calls": calls, "skips": skips}
            if dc < int(self._warmup_calls):
                return True

            skip_rate = float(ds) / float(dc) if dc > 0 else 0.0
            thr = float(self._holder.get("value", 0.0))

            # 简单比例控制：低于目标 => 提高阈值；明显高于目标 => 降低阈值
            if skip_rate < self._target:
                thr = thr * self._mul if thr > 0 else max(self._min_thr, 1e-4)
            elif skip_rate > (self._target * 1.6):
                thr = thr / self._mul if thr > 0 else 0.0

            thr = float(max(self._min_thr, min(self._max_thr, thr)))
            self._holder["value"] = thr
            self.logger.record("perf/fast_gate_autotune_skip_rate", float(skip_rate))
            self.logger.record("perf/fast_gate_autotune_thr", float(thr))
            return True

    class PoolPerfStatsCallback(BaseCallback):
        """
        记录 pool 关键函数耗时（用于定位 fps 衰减热点）：
        - pool.try_new_expr 总耗时（包含 mutual/optimize）
        - pool.optimize 耗时
        - pool._calc_ics 耗时（单 IC + mutual IC）
        """

        def __init__(self, pool_obj, update_every: int = 2048, verbose: int = 0):
            super().__init__(verbose=verbose)
            self._pool = pool_obj
            self._update_every = max(1, int(update_every))
            self._last_step = 0
            self._last = {
                "try_calls": 0,
                "try_time_s": 0.0,
                "opt_calls": 0,
                "opt_time_s": 0.0,
                "ics_calls": 0,
                "ics_time_s": 0.0,
            }

        def _on_step(self) -> bool:
            if (int(self.num_timesteps) - int(self._last_step)) < int(self._update_every):
                return True
            self._last_step = int(self.num_timesteps)
            st = getattr(self._pool, "_perf_stats", None)
            if not isinstance(st, dict):
                return True

            def _delta_int(key: str) -> int:
                v = int(st.get(key, 0))
                dv = v - int(self._last.get(key, 0))
                self._last[key] = v
                return dv

            def _delta_float(key: str) -> float:
                v = float(st.get(key, 0.0))
                dv = v - float(self._last.get(key, 0.0))
                self._last[key] = v
                return dv

            try_calls = _delta_int("try_calls")
            try_time_s = _delta_float("try_time_s")
            self.logger.record("perf/pool_try_new_expr_calls", float(try_calls))
            self.logger.record("perf/pool_try_new_expr_time_s", float(try_time_s))
            if try_calls > 0:
                self.logger.record("perf/pool_try_new_expr_ms_per_call", 1000.0 * float(try_time_s) / float(try_calls))

            opt_calls = _delta_int("opt_calls")
            opt_time_s = _delta_float("opt_time_s")
            self.logger.record("perf/pool_optimize_calls", float(opt_calls))
            self.logger.record("perf/pool_optimize_time_s", float(opt_time_s))
            if opt_calls > 0:
                self.logger.record("perf/pool_optimize_ms_per_call", 1000.0 * float(opt_time_s) / float(opt_calls))

            ics_calls = _delta_int("ics_calls")
            ics_time_s = _delta_float("ics_time_s")
            self.logger.record("perf/pool_calc_ics_calls", float(ics_calls))
            self.logger.record("perf/pool_calc_ics_time_s", float(ics_time_s))
            if ics_calls > 0:
                self.logger.record("perf/pool_calc_ics_ms_per_call", 1000.0 * float(ics_time_s) / float(ics_calls))

            return True

    class PeriodicValTestEvalCallback(BaseCallback):
        """
        周期性在验证/测试集评估当前因子池（IC/RankIC），并写入 TensorBoard 标量。

        说明：
        - 评估会额外消耗算力与内存（需要加载 Val/Test 数据到内存），默认关闭；
        - 通过环境变量 `ALPHAGEN_EVAL_EVERY_STEPS` 启用（>0）。
        """

        def __init__(
            self,
            pool,
            val_calculator_obj,
            test_calculator_obj,
            eval_every_steps: int,
            eval_test: bool,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self._pool = pool
            self._val_calc = val_calculator_obj
            self._test_calc = test_calculator_obj
            self._every = max(1, int(eval_every_steps))
            self._eval_test = bool(eval_test)
            self._last_step = 0

        @staticmethod
        def _get_target_tensor(calc_obj):
            # QLibStockDataCalculator: target_value
            tv = getattr(calc_obj, "target_value", None)
            if tv is not None:
                return tv
            # TensorQLibStockDataCalculator: target property from TensorAlphaCalculator
            try:
                return getattr(calc_obj, "target")
            except Exception:
                return None

        @staticmethod
        def _calc_icir_and_turnover(calc_obj, exprs, weights):
            """
            额外指标（不参与训练目标，仅用于观测）：
            - ICIR：按天 IC 序列的 mean/std
            - turnover：mean(|alpha_t - alpha_{t-1}|)（对组合 alpha）
            """
            try:
                from alphagen.utils.correlation import batch_pearsonr  # 延迟导入
            except Exception:
                return {}

            if not hasattr(calc_obj, "make_ensemble_alpha"):
                return {}
            target = PeriodicValTestEvalCallback._get_target_tensor(calc_obj)
            if target is None:
                return {}

            with torch.no_grad():
                alpha = calc_obj.make_ensemble_alpha(exprs, weights)
                ics = batch_pearsonr(alpha, target)  # (days,)
                ic_mean = ics.mean()
                ic_std = ics.std()
                icir = (ic_mean / ic_std).item() if ic_std.item() > 0 else float("nan")

                a0 = alpha[:-1]
                a1 = alpha[1:]
                mask = torch.isfinite(a0) & torch.isfinite(a1)
                diff = torch.abs(a1 - a0)
                diff[~mask] = torch.nan
                turnover = torch.nanmean(diff).item()

            return {
                "ic_std": float(ic_std.item()),
                "icir": float(icir),
                "turnover": float(turnover),
            }

        def _eval_pool(self, calc_obj):
            if getattr(self._pool, "size", 0) <= 0:
                return {"ic": 0.0, "ric": 0.0}
            exprs = [e for e in self._pool.exprs[: self._pool.size] if e is not None]
            weights = list(self._pool.weights)
            if len(exprs) == 0:
                return {"ic": 0.0, "ric": 0.0}
            ic, ric = calc_obj.calc_pool_all_ret(exprs, weights)
            out = {"ic": float(ic), "ric": float(ric)}
            out.update(self._calc_icir_and_turnover(calc_obj, exprs, weights))
            return out

        def _on_step(self) -> bool:
            # 注意：SB3 的 num_timesteps 往往按 n_steps 的倍数跳变；
            # 用 `% every == 0` 会导致当 every 不是 n_steps 的倍数时“永远不触发”。
            if (int(self.num_timesteps) - int(self._last_step)) < int(self._every):
                return True
            self._last_step = int(self.num_timesteps)

            # pool 基本信息（便于在 TB 里看 pool 是否在增长）
            self.logger.record("pool/size", float(getattr(self._pool, "size", 0)))
            self.logger.record("pool/best_ic_ret", float(getattr(self._pool, "best_ic_ret", 0.0)))

            try:
                v = self._eval_pool(self._val_calc)
                self.logger.record("eval/val_ic", float(v.get("ic", 0.0)))
                self.logger.record("eval/val_rank_ic", float(v.get("ric", 0.0)))
                if "icir" in v:
                    self.logger.record("eval/val_icir", float(v["icir"]))
                if "ic_std" in v:
                    self.logger.record("eval/val_ic_std", float(v["ic_std"]))
                if "turnover" in v:
                    self.logger.record("eval/val_turnover", float(v["turnover"]))
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Val 评估失败：{e}")

            if self._eval_test and (self._test_calc is not None):
                try:
                    t = self._eval_pool(self._test_calc)
                    self.logger.record("eval/test_ic", float(t.get("ic", 0.0)))
                    self.logger.record("eval/test_rank_ic", float(t.get("ric", 0.0)))
                    if "icir" in t:
                        self.logger.record("eval/test_icir", float(t["icir"]))
                    if "ic_std" in t:
                        self.logger.record("eval/test_ic_std", float(t["ic_std"]))
                    if "turnover" in t:
                        self.logger.record("eval/test_turnover", float(t["turnover"]))
                except Exception as e:
                    if self.verbose:
                        print(f"⚠ Test 评估失败：{e}")

            return True

    class IcLowerBoundScheduleCallback(BaseCallback):
        """
        动态 IC lower bound：
        - 训练初期放宽（更容易把“尚可”的表达式塞进 pool）
        - 训练后期收紧（逼迫更强的 alpha 进入 pool）
        """

        def __init__(
            self,
            pool,
            total_timesteps: int,
            start_lb: float,
            end_lb: float,
            update_every: int,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self.pool = pool
            self.total_timesteps = max(1, int(total_timesteps))
            self.start_lb = float(start_lb)
            self.end_lb = float(end_lb)
            self.update_every = max(1, int(update_every))
            self._last_lb: Optional[float] = None
            self._last_step = 0

        def _compute_lb(self) -> float:
            frac = min(1.0, float(self.num_timesteps) / float(self.total_timesteps))
            return self.start_lb + frac * (self.end_lb - self.start_lb)

        def _on_step(self) -> bool:
            if (int(self.num_timesteps) - int(self._last_step)) < int(self.update_every):
                return True
            self._last_step = int(self.num_timesteps)
            lb = float(self._compute_lb())
            # LinearAlphaPool 内部用的是 `_ic_lower_bound`（float），这里直接更新即可。
            setattr(self.pool, "_ic_lower_bound", lb)
            self.logger.record("pool/ic_lower_bound", lb)
            self._last_lb = lb
            return True

    class PoolLcbBetaScheduleCallback(BaseCallback):
        """
        动态调整 MeanStdAlphaPool 的 LCB beta（LCB = mean - beta * std）。

        直觉：
        - beta < 0：等价于“UCB”(mean + |beta|*std)，更偏探索不确定性大的表达式（类似 AlphaQCM 的 variance bonus）
        - beta > 0：更偏“保守稳健”(惩罚 std)，提升泛化稳定性（val/test 更不容易 flat）

        常用策略：beta 从负到正（先探索后稳健），例如 -0.5 -> +0.5。
        """

        def __init__(
            self,
            pool,
            total_timesteps: int,
            start_beta: float,
            end_beta: float,
            update_every: int,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self.pool = pool
            self.total_timesteps = max(1, int(total_timesteps))
            self.start_beta = float(start_beta)
            self.end_beta = float(end_beta)
            self.update_every = max(1, int(update_every))
            self._last: Optional[float] = None
            self._last_step = 0

        def _compute_beta(self) -> float:
            frac = min(1.0, float(self.num_timesteps) / float(self.total_timesteps))
            return self.start_beta + frac * (self.end_beta - self.start_beta)

        def _on_step(self) -> bool:
            if (int(self.num_timesteps) - int(self._last_step)) < int(self.update_every):
                return True
            self._last_step = int(self.num_timesteps)
            beta = float(self._compute_beta())
            setattr(self.pool, "_lcb_beta", beta)
            self.logger.record("pool/lcb_beta", beta)
            self._last = beta
            return True

    class MinExprLenScheduleCallback(BaseCallback):
        """
        动态最小表达式长度（MIN_EXPR_LEN）：
        - 训练初期允许更短表达式，先把“生成合法表达式/会 SEP”学会；
        - 训练后期逐步抬高最小长度，降低“每步都在评估”的频率，缓解越跑越慢。

        说明：
        - 该值通过运行时 monkey patch 影响 AlphaEnvCore._valid_action_types，从而控制 SEP 的可用性；
        - 这是性能/稳定性的工程性约束，不改变 pool 的 IC 口径。
        """

        def __init__(
            self,
            total_timesteps: int,
            start_len: int,
            end_len: int,
            update_every: int,
            holder: dict,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self.total_timesteps = max(1, int(total_timesteps))
            self.start_len = max(1, int(start_len))
            self.end_len = max(1, int(end_len))
            self.update_every = max(1, int(update_every))
            self.holder = holder
            self._last: Optional[int] = None
            self._last_step = 0

        def _compute_len(self) -> int:
            frac = min(1.0, float(self.num_timesteps) / float(self.total_timesteps))
            v = self.start_len + frac * (self.end_len - self.start_len)
            return max(1, int(round(v)))

        def _on_step(self) -> bool:
            if (int(self.num_timesteps) - int(self._last_step)) < int(self.update_every):
                return True
            self._last_step = int(self.num_timesteps)
            v = int(self._compute_len())
            self.holder["value"] = v
            self.logger.record("env/min_expr_len", float(v))
            self._last = v
            return True

    class NaNFriendlyMeanStdAlphaPool(MeanStdAlphaPool):
        """
        MeanStdAlphaPool 的 NaN 友好版本：
        - 单个因子缺失视为 0
        - 只有“全因子都缺失”的位置才保留 NaN

        目的：避免加权求和时 NaN 传播，导致组合 alpha 大面积 NaN，从而把 IC/ICIR 压成 0。
        """

        def _calc_obj_impl(self, alpha_values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            from alphagen.utils.correlation import batch_pearsonr  # 延迟导入避免循环

            target_value = self.calculator.target
            all_nan = torch.isnan(alpha_values).all(dim=0)
            weighted = (weights[:, None, None] * torch.nan_to_num(alpha_values, nan=0.0)).sum(dim=0)
            weighted[all_nan] = torch.nan
            ics = batch_pearsonr(weighted, target_value)
            mean, std = ics.mean(), ics.std()
            if getattr(self, "_lcb_beta", None) is not None:
                return mean - float(getattr(self, "_lcb_beta")) * std
            return mean / std

    # 训练配置
    SEED = int(os.environ.get("ALPHAGEN_SEED", "42"))
    BATCH_SIZE = int(os.environ.get("ALPHAGEN_BATCH_SIZE", "128"))
    TOTAL_TIMESTEPS = int(os.environ.get("ALPHAGEN_TOTAL_TIMESTEPS", "100000"))
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_obj = torch.device(DEVICE)

    # 输出配置
    OUTPUT_DIR = Path('./alphagen_output')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Alpha Pool配置 - 针对高维特征优化
    POOL_CAPACITY = int(os.environ.get("ALPHAGEN_POOL_CAPACITY", "10"))
    IC_LOWER_BOUND = float(os.environ.get("ALPHAGEN_IC_LOWER_BOUND", "0.01"))
    L1_ALPHA = float(os.environ.get("ALPHAGEN_POOL_L1_ALPHA", "0.005"))
    POOL_TYPE = os.environ.get("ALPHAGEN_POOL_TYPE", "mse").strip().lower()  # mse / meanstd
    pool_lcb_beta_raw = os.environ.get("ALPHAGEN_POOL_LCB_BETA", "none").strip().lower()
    POOL_LCB_BETA: Optional[float]
    if pool_lcb_beta_raw in {"none", "null", ""}:
        POOL_LCB_BETA = None
    else:
        POOL_LCB_BETA = float(pool_lcb_beta_raw)

    # LCB beta schedule（仅对 POOL_TYPE=meanstd 生效；用于“先 UCB 探索、后 LCB 稳健”）
    pool_lcb_beta_start_raw = os.environ.get("ALPHAGEN_POOL_LCB_BETA_START", str(pool_lcb_beta_raw)).strip().lower()
    pool_lcb_beta_end_raw = os.environ.get("ALPHAGEN_POOL_LCB_BETA_END", str(pool_lcb_beta_raw)).strip().lower()
    POOL_LCB_BETA_START: Optional[float]
    POOL_LCB_BETA_END: Optional[float]
    if pool_lcb_beta_start_raw in {"none", "null", ""}:
        POOL_LCB_BETA_START = None
    else:
        POOL_LCB_BETA_START = float(pool_lcb_beta_start_raw)
    if pool_lcb_beta_end_raw in {"none", "null", ""}:
        POOL_LCB_BETA_END = None
    else:
        POOL_LCB_BETA_END = float(pool_lcb_beta_end_raw)
    pool_lcb_beta_update_every = int(os.environ.get("ALPHAGEN_POOL_LCB_BETA_UPDATE_EVERY", "10000").strip() or 10000)

    # 动态 threshold：start/end 任意一个被设置就启用（默认与 IC_LOWER_BOUND 相同 => 等价于关闭）
    ic_lb_start = float(os.environ.get("ALPHAGEN_IC_LOWER_BOUND_START", str(IC_LOWER_BOUND)))
    ic_lb_end = float(os.environ.get("ALPHAGEN_IC_LOWER_BOUND_END", str(IC_LOWER_BOUND)))
    ic_lb_update_every = int(os.environ.get("ALPHAGEN_IC_LOWER_BOUND_UPDATE_EVERY", "2048"))

    # 模型/训练超参（允许通过环境变量配置，便于 alphagen_config.sh 生效）
    LSTM_LAYERS = int(os.environ.get("ALPHAGEN_LSTM_LAYERS", "2"))
    LSTM_DIM = int(os.environ.get("ALPHAGEN_LSTM_DIM", "128"))
    LSTM_DROPOUT = float(os.environ.get("ALPHAGEN_LSTM_DROPOUT", "0.1"))
    LEARNING_RATE = float(os.environ.get("ALPHAGEN_LEARNING_RATE", "3e-4"))
    N_STEPS = int(os.environ.get("ALPHAGEN_N_STEPS", "2048"))
    N_EPOCHS = int(os.environ.get("ALPHAGEN_N_EPOCHS", "10"))
    GAE_LAMBDA = float(os.environ.get("ALPHAGEN_GAE_LAMBDA", "0.95"))
    CLIP_RANGE = float(os.environ.get("ALPHAGEN_CLIP_RANGE", "0.2"))
    # 默认不额外加熵正则（保持与 PPO 默认一致，避免在大 action space 下冷启动卡死）
    ENT_COEF = float(os.environ.get("ALPHAGEN_ENT_COEF", "0.0"))
    # 默认关闭 KL early-stop（更贴近最初的训练脚本行为；需要时可显式设置数值开启）
    target_kl_raw = os.environ.get("ALPHAGEN_TARGET_KL", "none").strip().lower()
    TARGET_KL: Optional[float]
    if target_kl_raw in {"none", "null", ""}:
        TARGET_KL = None
    else:
        v = float(target_kl_raw)
        TARGET_KL = None if v <= 0 else v

    # 每步惩罚（鼓励更短表达式/更早 SEP），默认 0
    reward_per_step = float(os.environ.get("ALPHAGEN_REWARD_PER_STEP", "0").strip() or 0.0)
    env_wrapper.REWARD_PER_STEP = reward_per_step

    # 性能/探索控制：最小表达式长度（防止策略学会“超早 SEP”导致评估次数爆炸，从而越跑越慢）
    # - 设为 1 表示不限制（默认）
    # - 建议可先试 6~10（会明显减少 pool.try_new_expr 的调用频率）
    # 课程化建议：先用较小的 start（例如 1~3）让模型学会生成“可评估表达式”，再逐步拉到 end（例如 8~12）
    MIN_EXPR_LEN = int(os.environ.get("ALPHAGEN_MIN_EXPR_LEN", "1").strip() or 1)
    if MIN_EXPR_LEN < 1:
        MIN_EXPR_LEN = 1
    MIN_EXPR_LEN_START = int(os.environ.get("ALPHAGEN_MIN_EXPR_LEN_START", str(MIN_EXPR_LEN)).strip() or MIN_EXPR_LEN)
    MIN_EXPR_LEN_END = int(os.environ.get("ALPHAGEN_MIN_EXPR_LEN_END", str(MIN_EXPR_LEN)).strip() or MIN_EXPR_LEN)
    MIN_EXPR_LEN_START = max(1, MIN_EXPR_LEN_START)
    MIN_EXPR_LEN_END = max(1, MIN_EXPR_LEN_END)
    MIN_EXPR_LEN_UPDATE_EVERY = int(os.environ.get("ALPHAGEN_MIN_EXPR_LEN_UPDATE_EVERY", "2048").strip() or 2048)
    MIN_EXPR_LEN_UPDATE_EVERY = max(256, MIN_EXPR_LEN_UPDATE_EVERY)
    min_expr_len_holder = {"value": int(MIN_EXPR_LEN_START)}
    # schedule_steps 控制“多久从 start ramp 到 end”（默认=TOTAL_TIMESTEPS）。
    # 设小一点会更快抬高 min_expr_len，从而减少评估次数，显著缓解 fps 衰减。
    MIN_EXPR_LEN_SCHEDULE_STEPS = int(
        os.environ.get("ALPHAGEN_MIN_EXPR_LEN_SCHEDULE_STEPS", str(TOTAL_TIMESTEPS)).strip() or TOTAL_TIMESTEPS
    )
    MIN_EXPR_LEN_SCHEDULE_STEPS = max(1, MIN_EXPR_LEN_SCHEDULE_STEPS)
    stack_guard_raw = os.environ.get("ALPHAGEN_STACK_GUARD", "1").strip().lower()
    STACK_GUARD = stack_guard_raw in {"1", "true", "yes", "y", "on"}
    force_sep_raw = os.environ.get("ALPHAGEN_FORCE_SEP_WHEN_VALID", "0").strip().lower()
    FORCE_SEP_WHEN_VALID = force_sep_raw in {"1", "true", "yes", "y", "on"}

    # 性能控制：pool 权重优化上限（MseAlphaPool 的 Adam 优化默认 max_steps=10000 很重）
    # 仅对 POOL_TYPE=mse 生效；MeanStdAlphaPool 有自己的一套优化。
    POOL_OPT_LR = float(os.environ.get("ALPHAGEN_POOL_OPT_LR", "5e-4"))
    POOL_OPT_MAX_STEPS = int(os.environ.get("ALPHAGEN_POOL_OPT_MAX_STEPS", "10000"))
    POOL_OPT_TOLERANCE = int(os.environ.get("ALPHAGEN_POOL_OPT_TOLERANCE", "500"))
    POOL_OPT_MAX_STEPS = max(50, POOL_OPT_MAX_STEPS)
    POOL_OPT_TOLERANCE = max(10, POOL_OPT_TOLERANCE)

    # 性能/多样性控制：mutual IC 的“相似度早退”阈值
    # - 越小越严格（更早拒绝高相似表达式，提速+增强多样性，但也可能让学习信号变稀疏）
    # - 默认 0.99：基本等价于“不启用”（只过滤几乎完全重复）
    POOL_IC_MUT_THRESHOLD = float(os.environ.get("ALPHAGEN_POOL_IC_MUT_THRESHOLD", "0.99").strip() or 0.99)
    if not np.isfinite(POOL_IC_MUT_THRESHOLD):
        POOL_IC_MUT_THRESHOLD = 0.99
    POOL_IC_MUT_THRESHOLD = float(max(0.0, min(0.9999, POOL_IC_MUT_THRESHOLD)))

    # 可选：额外性能日志（写入 TensorBoard），用于定位 fps 衰减热点
    perf_log_raw = os.environ.get("ALPHAGEN_PERF_LOG", "0").strip().lower()
    PERF_LOG = perf_log_raw in {"1", "true", "yes", "y", "on"}

    # 周期性评估（默认关闭，>0 开启）
    eval_every_steps = int(os.environ.get("ALPHAGEN_EVAL_EVERY_STEPS", "0").strip() or 0)
    eval_test_flag = os.environ.get("ALPHAGEN_EVAL_TEST", "1").strip().lower() in {"1", "true", "yes", "y"}

    # 周期性保存 checkpoint（默认关闭，>0 开启）
    ckpt_every_steps = int(os.environ.get("ALPHAGEN_CHECKPOINT_EVERY_STEPS", "0").strip() or 0)
    ckpt_keep_last = int(os.environ.get("ALPHAGEN_CHECKPOINT_KEEP", "3").strip() or 3)
    ckpt_every_steps = max(0, ckpt_every_steps)
    ckpt_keep_last = max(1, ckpt_keep_last)

    # 近似评估（FastGate）：pool 已满时先用“小样本数据”估计 single-IC，
    # 不达标则跳过完整评估（节省 mutual IC 与 pool optimize 的开销）。
    #
    # 设计原则：
    # - 默认关闭（避免对训练动态产生不可预期影响）
    # - 默认只在 pool 满时启用（尽量不影响冷启动/早期探索）
    fast_gate_raw = os.environ.get("ALPHAGEN_FAST_GATE", "0").strip().lower()
    FAST_GATE = fast_gate_raw in {"1", "true", "yes", "y", "on"}
    fast_gate_only_full_raw = os.environ.get("ALPHAGEN_FAST_GATE_ONLY_WHEN_FULL", "1").strip().lower()
    FAST_GATE_ONLY_WHEN_FULL = fast_gate_only_full_raw in {"1", "true", "yes", "y", "on"}
    FAST_GATE_SYMBOLS = int(os.environ.get("ALPHAGEN_FAST_GATE_SYMBOLS", "20").strip() or 20)
    FAST_GATE_PERIODS = int(os.environ.get("ALPHAGEN_FAST_GATE_PERIODS", "4000").strip() or 4000)  # 1h bars
    FAST_GATE_MIN_ABS_IC = float(os.environ.get("ALPHAGEN_FAST_GATE_MIN_ABS_IC", "0.0").strip() or 0.0)
    FAST_GATE_SYMBOLS = max(4, FAST_GATE_SYMBOLS)
    FAST_GATE_PERIODS = max(256, FAST_GATE_PERIODS)
    FAST_GATE_MIN_ABS_IC = max(0.0, FAST_GATE_MIN_ABS_IC)
    # FastGate 阈值也支持“先松后紧”的 schedule（只影响粗筛，不改变真实 pool 目标）
    FAST_GATE_MIN_ABS_IC_START = float(
        os.environ.get("ALPHAGEN_FAST_GATE_MIN_ABS_IC_START", str(FAST_GATE_MIN_ABS_IC)).strip() or FAST_GATE_MIN_ABS_IC
    )
    FAST_GATE_MIN_ABS_IC_END = float(
        os.environ.get("ALPHAGEN_FAST_GATE_MIN_ABS_IC_END", str(FAST_GATE_MIN_ABS_IC)).strip() or FAST_GATE_MIN_ABS_IC
    )
    FAST_GATE_MIN_ABS_IC_UPDATE_EVERY = int(os.environ.get("ALPHAGEN_FAST_GATE_MIN_ABS_IC_UPDATE_EVERY", "2048").strip() or 2048)
    FAST_GATE_MIN_ABS_IC_SCHEDULE_STEPS = int(
        os.environ.get("ALPHAGEN_FAST_GATE_MIN_ABS_IC_SCHEDULE_STEPS", str(TOTAL_TIMESTEPS)).strip() or TOTAL_TIMESTEPS
    )
    FAST_GATE_MIN_ABS_IC_START = max(0.0, float(FAST_GATE_MIN_ABS_IC_START))
    FAST_GATE_MIN_ABS_IC_END = max(0.0, float(FAST_GATE_MIN_ABS_IC_END))
    FAST_GATE_MIN_ABS_IC_UPDATE_EVERY = max(256, int(FAST_GATE_MIN_ABS_IC_UPDATE_EVERY))
    FAST_GATE_MIN_ABS_IC_SCHEDULE_STEPS = max(1, int(FAST_GATE_MIN_ABS_IC_SCHEDULE_STEPS))
    fast_gate_thr_holder = {"value": float(FAST_GATE_MIN_ABS_IC_START)}
    # 可选：FastGate 阈值至少为 ic_lower_bound 的一部分（在后期能显著提升 skip_rate）
    FAST_GATE_THR_LB_RATIO = float(os.environ.get("ALPHAGEN_FAST_GATE_THR_LB_RATIO", "0").strip() or 0.0)
    if not np.isfinite(FAST_GATE_THR_LB_RATIO):
        FAST_GATE_THR_LB_RATIO = 0.0
    FAST_GATE_THR_LB_RATIO = float(max(0.0, FAST_GATE_THR_LB_RATIO))

    # 可选：FastGate 自动调阈值（目标：把 skip_rate 拉到一个区间，从而维持可接受的 fps）
    fast_gate_autotune_raw = os.environ.get("ALPHAGEN_FAST_GATE_AUTOTUNE", "0").strip().lower()
    FAST_GATE_AUTOTUNE = fast_gate_autotune_raw in {"1", "true", "yes", "y", "on"}
    FAST_GATE_AUTOTUNE_TARGET_SKIP = float(os.environ.get("ALPHAGEN_FAST_GATE_AUTOTUNE_TARGET_SKIP", "0.3").strip() or 0.3)
    FAST_GATE_AUTOTUNE_ADJUST_MUL = float(os.environ.get("ALPHAGEN_FAST_GATE_AUTOTUNE_ADJUST_MUL", "1.15").strip() or 1.15)
    FAST_GATE_AUTOTUNE_MIN_THR = float(os.environ.get("ALPHAGEN_FAST_GATE_AUTOTUNE_MIN_THR", "0.0").strip() or 0.0)
    FAST_GATE_AUTOTUNE_MAX_THR = float(os.environ.get("ALPHAGEN_FAST_GATE_AUTOTUNE_MAX_THR", "0.02").strip() or 0.02)
    if not np.isfinite(FAST_GATE_AUTOTUNE_TARGET_SKIP):
        FAST_GATE_AUTOTUNE_TARGET_SKIP = 0.3
    if not np.isfinite(FAST_GATE_AUTOTUNE_ADJUST_MUL) or FAST_GATE_AUTOTUNE_ADJUST_MUL <= 1.0:
        FAST_GATE_AUTOTUNE_ADJUST_MUL = 1.15
    if not np.isfinite(FAST_GATE_AUTOTUNE_MIN_THR):
        FAST_GATE_AUTOTUNE_MIN_THR = 0.0
    if not np.isfinite(FAST_GATE_AUTOTUNE_MAX_THR):
        FAST_GATE_AUTOTUNE_MAX_THR = 0.02
    FAST_GATE_AUTOTUNE_TARGET_SKIP = float(max(0.0, min(0.95, FAST_GATE_AUTOTUNE_TARGET_SKIP)))
    FAST_GATE_AUTOTUNE_MIN_THR = float(max(0.0, FAST_GATE_AUTOTUNE_MIN_THR))
    FAST_GATE_AUTOTUNE_MAX_THR = float(max(FAST_GATE_AUTOTUNE_MIN_THR, FAST_GATE_AUTOTUNE_MAX_THR))

    # 近似评估（ValGate）：用“验证集小样本 single-IC”做二次门控，目标是提升 OOS（val）质量，
    # 同时减少会触发昂贵 pool.optimize 的候选数量（帮助稳定 fps）。
    val_gate_raw = os.environ.get("ALPHAGEN_VAL_GATE", "0").strip().lower()
    VAL_GATE = val_gate_raw in {"1", "true", "yes", "y", "on"}
    val_gate_only_full_raw = os.environ.get("ALPHAGEN_VAL_GATE_ONLY_WHEN_FULL", "1").strip().lower()
    VAL_GATE_ONLY_WHEN_FULL = val_gate_only_full_raw in {"1", "true", "yes", "y", "on"}
    VAL_GATE_SYMBOLS = int(os.environ.get("ALPHAGEN_VAL_GATE_SYMBOLS", "20").strip() or 20)
    VAL_GATE_PERIODS = int(os.environ.get("ALPHAGEN_VAL_GATE_PERIODS", "4000").strip() or 4000)
    VAL_GATE_MIN_ABS_IC = float(os.environ.get("ALPHAGEN_VAL_GATE_MIN_ABS_IC", "0.0").strip() or 0.0)
    VAL_GATE_SYMBOLS = max(4, VAL_GATE_SYMBOLS)
    VAL_GATE_PERIODS = max(256, VAL_GATE_PERIODS)
    VAL_GATE_MIN_ABS_IC = max(0.0, VAL_GATE_MIN_ABS_IC)

    print("=" * 60)
    print("AlphaGen Crypto Factor Mining")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Symbols: {SYMBOLS}")
    print(f"Train: {START_TIME} -> {TRAIN_END}")
    print(f"Val: {TRAIN_END} -> {VAL_END}")
    print(f"Test: {VAL_END} -> {END_TIME}")
    if MIN_EXPR_LEN_START != MIN_EXPR_LEN_END:
        print(
            f"Min expr len schedule: {MIN_EXPR_LEN_START}->{MIN_EXPR_LEN_END} "
            f"(every {MIN_EXPR_LEN_UPDATE_EVERY} steps, schedule_steps={MIN_EXPR_LEN_SCHEDULE_STEPS})"
        )
    elif MIN_EXPR_LEN > 1:
        print(f"Min expr len: {MIN_EXPR_LEN}（将延迟允许 SEP，减少评估次数以提速）")
    print(f"Stack guard: {'ON' if STACK_GUARD else 'OFF'}（避免栈过深导致最终表达式无效 => reward=-1）")
    print(f"Force SEP when valid: {'ON' if FORCE_SEP_WHEN_VALID else 'OFF'}（一旦表达式有效，强制下一步只能 SEP，避免越生成越无效）")
    print(f"Pool optimize: lr={POOL_OPT_LR}, max_steps={POOL_OPT_MAX_STEPS}, tol={POOL_OPT_TOLERANCE}")
    if ckpt_every_steps > 0:
        print(f"Checkpoint: every {ckpt_every_steps} steps, keep_last={ckpt_keep_last}")
    print(f"PPO: n_steps={N_STEPS}, batch_size={BATCH_SIZE}, n_epochs={N_EPOCHS}")
    print(f"Features (dynamic): {len(feature_space.feature_cols)}")
    print()

    # ==================== 设置随机种子 ====================
    reseed_everything(SEED)

    # ==================== 加载训练数据 ====================
    print("Loading training data...")
    train_data = CryptoData(
        symbols=SYMBOLS,
        start_time=START_TIME,
        end_time=TRAIN_END,
        timeframe='1h',
        data_dir=DATA_DIR,
        max_backtrack_periods=100,
        max_future_periods=30,
        features=None,  # 使用动态 FeatureType 的全集
        device=device_obj
    )

    print(f"Train data: {train_data.n_days} days, {train_data.n_stocks} symbols, {train_data.n_features} features")

    # ==================== 定义目标 ====================
    # 预测1小时后的收益率
    if "close" not in feature_space.feature_cols:
        raise RuntimeError(
            f"特征列中未找到 close，无法构造目标。请检查 {DATA_DIR} 下 *_train.csv 表头。"
        )
    close_idx = feature_space.feature_cols.index("close")
    close_expr = Feature(sd.FeatureType(close_idx))
    target = Ref(close_expr, -1) / close_expr - 1

    print(f"Target: 1-hour forward return")

    # FastGate：构造一个更小的“训练子集”，仅用于 single-IC 粗筛（不改变最终 IC 口径）
    fast_gate_calc = None
    if FAST_GATE:
        try:
            class _StockDataView:
                def __init__(self, base, data_tensor: torch.Tensor):
                    self.data = data_tensor
                    self.max_backtrack_days = int(getattr(base, "max_backtrack_days"))
                    self.max_future_days = int(getattr(base, "max_future_days"))

                @property
                def n_features(self) -> int:
                    return int(self.data.shape[1])

                @property
                def n_stocks(self) -> int:
                    return int(self.data.shape[2])

                @property
                def n_days(self) -> int:
                    return int(self.data.shape[0] - self.max_backtrack_days - self.max_future_days)

            total_symbols = int(getattr(train_data, "n_stocks", 0))
            take_symbols = min(int(FAST_GATE_SYMBOLS), max(1, total_symbols))
            rng = np.random.default_rng(SEED)
            idx = np.array(sorted(rng.choice(total_symbols, size=take_symbols, replace=False).tolist()), dtype=np.int64)

            base_tensor = train_data.data
            need_len = int(FAST_GATE_PERIODS + train_data.max_backtrack_days + train_data.max_future_days + 1)
            start = max(0, int(base_tensor.shape[0]) - need_len)
            fast_tensor = base_tensor[start:, :, :].index_select(2, torch.tensor(idx, device=base_tensor.device))
            fast_view = _StockDataView(train_data, fast_tensor)

            # 注意：这里用的是“同口径 calculator”，但数据更小，速度更快
            if POOL_TYPE == "meanstd":
                fast_gate_calc = TensorQLibStockDataCalculator(fast_view, target)
            else:
                fast_gate_calc = QLibStockDataCalculator(fast_view, target)
            print(
                f"✓ FastGate 已启用：symbols={take_symbols}/{total_symbols}, periods≈{FAST_GATE_PERIODS}, "
                f"min_abs_ic={FAST_GATE_MIN_ABS_IC}, only_when_full={FAST_GATE_ONLY_WHEN_FULL}"
            )
        except Exception as e:
            print(f"⚠ FastGate 初始化失败（将禁用）：{e}")
            fast_gate_calc = None

    # ==================== 创建Calculator ====================
    print("\nInitializing calculator...")
    if POOL_TYPE == "meanstd":
        calculator = TensorQLibStockDataCalculator(train_data, target)
    else:
        calculator = QLibStockDataCalculator(train_data, target)

    # 可选：提前加载 Val/Test 数据与 calculator，用于周期性评估
    val_calculator_periodic = None
    test_calculator_periodic = None
    if eval_every_steps > 0:
        print("\n" + "=" * 60)
        print(f"周期性评估已开启：每 {eval_every_steps} steps 评估一次（将额外加载 Val/Test 数据）")
        print("=" * 60)

        val_data_periodic = CryptoData(
            symbols=SYMBOLS,
            start_time=TRAIN_END,
            end_time=VAL_END,
            timeframe='1h',
            data_dir=DATA_DIR,
            max_backtrack_periods=100,
            max_future_periods=30,
            features=None,
            device=device_obj,
        )
        test_data_periodic = CryptoData(
            symbols=SYMBOLS,
            start_time=VAL_END,
            end_time=END_TIME,
            timeframe='1h',
            data_dir=DATA_DIR,
            max_backtrack_periods=100,
            max_future_periods=30,
            features=None,
            device=device_obj,
        )

        if POOL_TYPE == "meanstd":
            val_calculator_periodic = TensorQLibStockDataCalculator(val_data_periodic, target)
            test_calculator_periodic = TensorQLibStockDataCalculator(test_data_periodic, target)
        else:
            val_calculator_periodic = QLibStockDataCalculator(val_data_periodic, target)
            test_calculator_periodic = QLibStockDataCalculator(test_data_periodic, target)

    # 可选：ValGate（验证集小样本 single-IC 门控）
    val_gate_calc = None
    if VAL_GATE:
        try:
            # 复用 periodic val_data（如果有），避免额外加载
            base_val_data = None
            if eval_every_steps > 0:
                base_val_data = val_data_periodic  # type: ignore[name-defined]
            else:
                base_val_data = CryptoData(
                    symbols=SYMBOLS,
                    start_time=TRAIN_END,
                    end_time=VAL_END,
                    timeframe="1h",
                    data_dir=DATA_DIR,
                    max_backtrack_periods=100,
                    max_future_periods=30,
                    features=None,
                    device=device_obj,
                )

            class _StockDataView:
                def __init__(self, base, data_tensor: torch.Tensor):
                    self.data = data_tensor
                    self.max_backtrack_days = int(getattr(base, "max_backtrack_days"))
                    self.max_future_days = int(getattr(base, "max_future_days"))

                @property
                def n_features(self) -> int:
                    return int(self.data.shape[1])

                @property
                def n_stocks(self) -> int:
                    return int(self.data.shape[2])

                @property
                def n_days(self) -> int:
                    return int(self.data.shape[0] - self.max_backtrack_days - self.max_future_days)

            total_symbols = int(getattr(base_val_data, "n_stocks", 0))
            take_symbols = min(int(VAL_GATE_SYMBOLS), max(1, total_symbols))
            rng = np.random.default_rng(SEED + 7)
            idx = np.array(sorted(rng.choice(total_symbols, size=take_symbols, replace=False).tolist()), dtype=np.int64)

            base_tensor = base_val_data.data
            need_len = int(VAL_GATE_PERIODS + base_val_data.max_backtrack_days + base_val_data.max_future_days + 1)
            start = max(0, int(base_tensor.shape[0]) - need_len)
            val_tensor = base_tensor[start:, :, :].index_select(2, torch.tensor(idx, device=base_tensor.device))
            val_view = _StockDataView(base_val_data, val_tensor)

            if POOL_TYPE == "meanstd":
                val_gate_calc = TensorQLibStockDataCalculator(val_view, target)
            else:
                val_gate_calc = QLibStockDataCalculator(val_view, target)
            print(
                f"✓ ValGate 已启用：symbols={take_symbols}/{total_symbols}, periods≈{VAL_GATE_PERIODS}, "
                f"min_abs_ic={VAL_GATE_MIN_ABS_IC}, only_when_full={VAL_GATE_ONLY_WHEN_FULL}"
            )
        except Exception as e:
            print(f"⚠ ValGate 初始化失败（将禁用）：{e}")
            val_gate_calc = None

    # ==================== 创建Alpha Pool ====================
    # 如果启用动态阈值，初始化用 start（避免刚开始就被“后期阈值”卡死）
    init_ic_lb = ic_lb_start if (ic_lb_start != ic_lb_end) else IC_LOWER_BOUND
    print(
        f"Creating alpha pool (type={POOL_TYPE}, capacity={POOL_CAPACITY}, IC threshold={init_ic_lb})..."
        + (f" [schedule {ic_lb_start}->{ic_lb_end}]" if ic_lb_start != ic_lb_end else "")
    )
    if POOL_TYPE == "meanstd":
        class TunableNaNFriendlyMeanStdAlphaPool(NaNFriendlyMeanStdAlphaPool):
            def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:  # type: ignore[override]
                # MeanStdAlphaPool 默认 optimize(max_steps=10000) 在 days*stocks 很大时非常慢。
                # 这里复用 ALPHAGEN_POOL_OPT_* 作为统一的“优化预算阀门”，方便在脚本里提速/稳住。
                return super().optimize(lr=POOL_OPT_LR, max_steps=POOL_OPT_MAX_STEPS, tolerance=POOL_OPT_TOLERANCE)

        pool = TunableNaNFriendlyMeanStdAlphaPool(
            capacity=POOL_CAPACITY,
            calculator=calculator,  # type: ignore[arg-type]
            ic_lower_bound=init_ic_lb,
            l1_alpha=L1_ALPHA,
            lcb_beta=POOL_LCB_BETA_START if (POOL_LCB_BETA_START is not None) else POOL_LCB_BETA,
            device=device_obj,
        )
    else:
        class TunableMseAlphaPool(MseAlphaPool):
            def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:  # type: ignore[override]
                return super().optimize(lr=POOL_OPT_LR, max_steps=POOL_OPT_MAX_STEPS, tolerance=POOL_OPT_TOLERANCE)

        pool = TunableMseAlphaPool(
            capacity=POOL_CAPACITY,
            calculator=calculator,
            ic_lower_bound=init_ic_lb,
            l1_alpha=L1_ALPHA,
            device=device_obj,
        )

    # mutual IC “相似度早退”阈值（用于提速/增强多样性；见 LinearAlphaPool.try_new_expr）
    try:
        setattr(pool, "_ic_mut_threshold", float(POOL_IC_MUT_THRESHOLD))
        print(f"Pool mutual IC early-exit threshold: {float(POOL_IC_MUT_THRESHOLD):.4f}")
    except Exception:
        pass

    # 互相关计算加速：把 mutual IC 比对顺序改为“按权重绝对值从大到小”，并允许用更严格阈值早退。
    # 设计目标：
    # - 后期 pool 满时，mutual IC 计算是 fps 衰减最大来源；
    # - 越早与“重要 alpha”比对，越容易触发 early-exit，从而减少无意义 mutual 计算；
    # - 不改变最终 mutual IC 数值（只改变计算顺序），因此不会引入指标口径偏差。
    mutual_order_raw = os.environ.get("ALPHAGEN_POOL_MUTUAL_ORDER", "1").strip().lower()
    MUTUAL_ORDER = mutual_order_raw in {"1", "true", "yes", "y", "on"}
    if MUTUAL_ORDER:
        try:
            import types

            _orig_calc_ics_impl = getattr(pool, "_calc_ics", None)

            def _calc_ics_weight_order(self, expr, ic_mut_threshold=None):  # type: ignore[no-redef]
                # 兼容上游：try_new_expr 固定传 ic_mut_threshold=0.99；
                # 这里允许外部通过 `_ic_mut_threshold` 把阈值收紧（取 min），以便更早 early-exit。
                thr = ic_mut_threshold
                cfg_thr = getattr(self, "_ic_mut_threshold", None)
                try:
                    if cfg_thr is not None and np.isfinite(float(cfg_thr)):
                        thr = float(cfg_thr) if thr is None else min(float(thr), float(cfg_thr))
                except Exception:
                    pass

                single_ic = self.calculator.calc_single_IC_ret(expr)
                try:
                    under = bool(getattr(self, "_under_thres_alpha"))
                except Exception:
                    under = False
                lb = getattr(self, "_ic_lower_bound", None)
                if (lb is not None) and (not under) and (single_ic < float(lb)):
                    return single_ic, None
                if getattr(self, "size", 0) <= 0:
                    return single_ic, []

                order = list(range(int(self.size)))
                if thr is not None:
                    try:
                        order.sort(key=lambda i: -float(abs(getattr(self, "_weights")[i])))
                    except Exception:
                        pass

                mutual_ics = [0.0] * int(self.size)
                for i in order:
                    mutual_ic = self.calculator.calc_mutual_IC(expr, self.exprs[i])  # type: ignore[index]
                    if thr is not None and mutual_ic > thr:
                        return single_ic, None
                    mutual_ics[i] = mutual_ic
                return single_ic, mutual_ics

            # 仅在 LinearAlphaPool 场景下替换；否则回退原实现
            if _orig_calc_ics_impl is not None:
                pool._calc_ics = types.MethodType(_calc_ics_weight_order, pool)  # type: ignore[attr-defined]
        except Exception:
            pass

    # 可选：恢复历史 alpha_pool（重要：ALPHAGEN_RESUME 默认只恢复 PPO 模型，不会恢复 pool）
    # 用法示例：
    #   ALPHAGEN_POOL_RESUME=1 \
    #   ALPHAGEN_POOL_RESUME_PATH=alphagen_output/alpha_pool.json \
    #   ALPHAGEN_POOL_RESUME_WEIGHTS=1 \
    #   ./run_training.sh ...
    pool_resume_raw = os.environ.get("ALPHAGEN_POOL_RESUME", "0").strip().lower()
    POOL_RESUME = pool_resume_raw in {"1", "true", "yes", "y", "on"}
    pool_resume_path = Path(
        os.environ.get("ALPHAGEN_POOL_RESUME_PATH", str(OUTPUT_DIR / "alpha_pool.json")).strip()
        or str(OUTPUT_DIR / "alpha_pool.json")
    )
    pool_resume_weights_raw = os.environ.get("ALPHAGEN_POOL_RESUME_WEIGHTS", "1").strip().lower()
    POOL_RESUME_WEIGHTS = pool_resume_weights_raw in {"1", "true", "yes", "y", "on"}
    if POOL_RESUME and pool_resume_path.exists():
        try:
            from alphagen.data.parser import parse_expression

            obj = json.loads(pool_resume_path.read_text(encoding="utf-8"))
            raw_exprs = obj.get("exprs", [])
            raw_weights = obj.get("weights", None)
            expr_pairs = []
            if isinstance(raw_exprs, list):
                if isinstance(raw_weights, list) and POOL_RESUME_WEIGHTS:
                    for s, w in zip(raw_exprs, raw_weights):
                        expr_pairs.append((s, w))
                else:
                    for s in raw_exprs:
                        expr_pairs.append((s, None))

            parsed_exprs = []
            parsed_weights = []
            for s, w in expr_pairs:
                if not isinstance(s, str) or not s.strip():
                    continue
                try:
                    e = parse_expression(s.strip())
                except Exception:
                    continue
                parsed_exprs.append(e)
                if w is not None:
                    try:
                        parsed_weights.append(float(w))
                    except Exception:
                        parsed_weights.append(0.0)

            # force_load_exprs 会重新计算 mutual IC 等信息；
            # 为了确保能完整恢复，不让 ic_lower_bound 影响加载，加载时临时关闭阈值。
            if parsed_exprs:
                orig_lb = getattr(pool, "_ic_lower_bound", None)
                try:
                    if orig_lb is not None:
                        setattr(pool, "_ic_lower_bound", -1.0)
                    if parsed_weights and len(parsed_weights) == len(parsed_exprs):
                        pool.force_load_exprs(parsed_exprs, weights=parsed_weights)
                        print(f"✓ 恢复 alpha_pool（含 weights）: {pool_resume_path}（loaded={pool.size}）")
                    else:
                        pool.force_load_exprs(parsed_exprs, weights=None)
                        print(f"✓ 恢复 alpha_pool（重算 weights）: {pool_resume_path}（loaded={pool.size}）")
                finally:
                    if orig_lb is not None:
                        setattr(pool, "_ic_lower_bound", orig_lb)
        except Exception as e:
            print(f"⚠ alpha_pool 恢复失败（将忽略）：{e}")

    # 可选：FastGate/ValGate（近似 single-IC 粗筛）
    # 说明：返回 0.0 表示“这次不值得做完整评估”，不改变 pool 状态。
    if (fast_gate_calc is not None) or (val_gate_calc is not None):
        import types
        import time as _time

        if fast_gate_calc is not None:
            pool._fast_gate_stats = {"calls": 0, "skips": 0, "time_s": 0.0}  # type: ignore[attr-defined]
        if val_gate_calc is not None:
            pool._val_gate_stats = {"calls": 0, "skips": 0, "time_s": 0.0}  # type: ignore[attr-defined]

        _orig_try_new_expr = pool.__class__.try_new_expr

        def _try_new_expr_gates(self, expr):  # type: ignore[no-redef]
            # 1) FastGate：训练集小样本 single-IC 粗筛（主要为了省 mutual/optimize）
            if fast_gate_calc is not None:
                only_full = bool(FAST_GATE_ONLY_WHEN_FULL)
                if (not only_full) or (getattr(self, "size", 0) >= getattr(self, "capacity", 0)):
                    t0g = _time.perf_counter()
                    try:
                        ic_fast = float(fast_gate_calc.calc_single_IC_ret(expr))
                    except Exception:
                        ic_fast = float("nan")
                    dtg = _time.perf_counter() - t0g

                    st = getattr(self, "_fast_gate_stats", None)
                    if isinstance(st, dict):
                        st["calls"] = int(st.get("calls", 0)) + 1
                        st["time_s"] = float(st.get("time_s", 0.0)) + float(dtg)

                    thr = float(fast_gate_thr_holder.get("value", FAST_GATE_MIN_ABS_IC))
                    if FAST_GATE_THR_LB_RATIO > 0:
                        try:
                            lb = getattr(self, "_ic_lower_bound", None)
                            if lb is not None:
                                thr = max(thr, float(FAST_GATE_THR_LB_RATIO) * float(lb))
                        except Exception:
                            pass
                    if not (np.isfinite(ic_fast) and (abs(ic_fast) >= thr)):
                        if isinstance(st, dict):
                            st["skips"] = int(st.get("skips", 0)) + 1
                        return 0.0

            # 2) ValGate：验证集小样本 single-IC 二次筛（对齐 val_ic）
            if val_gate_calc is not None:
                only_full = bool(VAL_GATE_ONLY_WHEN_FULL)
                if (not only_full) or (getattr(self, "size", 0) >= getattr(self, "capacity", 0)):
                    t0v = _time.perf_counter()
                    try:
                        ic_val = float(val_gate_calc.calc_single_IC_ret(expr))
                    except Exception:
                        ic_val = float("nan")
                    dtv = _time.perf_counter() - t0v

                    st = getattr(self, "_val_gate_stats", None)
                    if isinstance(st, dict):
                        st["calls"] = int(st.get("calls", 0)) + 1
                        st["time_s"] = float(st.get("time_s", 0.0)) + float(dtv)

                    thr = float(VAL_GATE_MIN_ABS_IC)
                    if not (np.isfinite(ic_val) and (abs(ic_val) >= thr)):
                        if isinstance(st, dict):
                            st["skips"] = int(st.get("skips", 0)) + 1
                        return 0.0

            return _orig_try_new_expr(self, expr)

        pool.try_new_expr = types.MethodType(_try_new_expr_gates, pool)  # type: ignore[assignment]

    # 可选：额外性能日志（用于解释“为什么后面 fps 会掉到 80/50”）
    # 通过 monkey patch 统计 pool 关键函数耗时，再由 PoolPerfStatsCallback 写入 TensorBoard。
    if PERF_LOG:
        try:
            import time as _time

            setattr(
                pool,
                "_perf_stats",
                {
                    "try_calls": 0,
                    "try_time_s": 0.0,
                    "opt_calls": 0,
                    "opt_time_s": 0.0,
                    "ics_calls": 0,
                    "ics_time_s": 0.0,
                },
            )

            _orig_optimize = getattr(pool, "optimize")

            def _optimize_perf(self, *args, **kwargs):  # type: ignore[no-redef]
                t0 = _time.perf_counter()
                out = _orig_optimize(*args, **kwargs)
                dt = _time.perf_counter() - t0
                st = getattr(self, "_perf_stats", None)
                if isinstance(st, dict):
                    st["opt_calls"] = int(st.get("opt_calls", 0)) + 1
                    st["opt_time_s"] = float(st.get("opt_time_s", 0.0)) + float(dt)
                return out

            pool.optimize = types.MethodType(_optimize_perf, pool)  # type: ignore[assignment]

            _orig_calc_ics = getattr(pool, "_calc_ics", None)
            if _orig_calc_ics is not None:

                def _calc_ics_perf(self, *args, **kwargs):  # type: ignore[no-redef]
                    t0 = _time.perf_counter()
                    out = _orig_calc_ics(*args, **kwargs)
                    dt = _time.perf_counter() - t0
                    st = getattr(self, "_perf_stats", None)
                    if isinstance(st, dict):
                        st["ics_calls"] = int(st.get("ics_calls", 0)) + 1
                        st["ics_time_s"] = float(st.get("ics_time_s", 0.0)) + float(dt)
                    return out

                pool._calc_ics = types.MethodType(_calc_ics_perf, pool)  # type: ignore[attr-defined]

            _orig_try = getattr(pool, "try_new_expr")

            def _try_new_expr_perf(self, expr):  # type: ignore[no-redef]
                t0 = _time.perf_counter()
                out = _orig_try(expr)
                dt = _time.perf_counter() - t0
                st = getattr(self, "_perf_stats", None)
                if isinstance(st, dict):
                    st["try_calls"] = int(st.get("try_calls", 0)) + 1
                    st["try_time_s"] = float(st.get("try_time_s", 0.0)) + float(dt)
                return out

            pool.try_new_expr = types.MethodType(_try_new_expr_perf, pool)  # type: ignore[assignment]
            print("✓ PERF_LOG 已启用：将记录 pool.try_new_expr/optimize/_calc_ics 耗时到 TensorBoard")
        except Exception as e:
            print(f"⚠ PERF_LOG 初始化失败（将忽略）：{e}")

    # 可选：限制最小表达式长度/启用 stack guard（通过 monkey patch core 的 stop 有效性）
    # - MIN_EXPR_LEN：控制 SEP 何时可用（减少评估频率 => 提速）
    # - STACK_GUARD：避免栈过深导致最终表达式无效（reward=-1），提升冷启动可训练性
    if (MIN_EXPR_LEN_START > 1) or (MIN_EXPR_LEN_END > 1) or STACK_GUARD:
        try:
            import alphagen.rl.env.core as _core_mod

            _orig_valid_action_types = _core_mod.AlphaEnvCore._valid_action_types

            def _valid_action_types_with_min_len(self):  # type: ignore[no-redef]
                ret = _orig_valid_action_types(self)
                tokens = getattr(self, "_tokens", [])
                # self._tokens 包含 BEG_TOKEN，真实 token 数 = len(_tokens)-1
                min_len = int(min_expr_len_holder.get("value", 1))
                if (len(tokens) - 1) < min_len:
                    ret["select"][4] = False  # SEP
                else:
                    # 关键：很多 -1/15 冷启动来自“表达式在中途一度有效，但策略继续堆 token，
                    # 直到 MAX_EXPR_LENGTH 结束时又变无效”。这里提供可选的“强制收敛”：
                    # 一旦表达式已经有效，就强制下一步只能 SEP，让学习信号不再稀疏。
                    if FORCE_SEP_WHEN_VALID and bool(ret["select"][4]):
                        # 只保留 SEP，禁用所有其它动作类型
                        ret["select"][0] = False
                        ret["select"][1] = False
                        ret["select"][2] = False
                        ret["select"][3] = False
                        for k in list(ret.get("op", {}).keys()):
                            ret["op"][k] = False

                # Stack guard：当栈太深且剩余 token 太少时，禁止继续 push（特征/常量/dt/子表达式），
                # 强制策略优先选择 Operator 来“收栈”，避免最终长度到顶时表达式无效 => reward=-1。
                #
                # 这是一个启发式约束（默认开启，可用 ALPHAGEN_STACK_GUARD=0 关闭），
                # 目标是让训练更快进入“可评估表达式”的轨道，并减少无意义的 -1 episode。
                if STACK_GUARD:
                    try:
                        stack_size = len(getattr(getattr(self, "_builder", None), "stack", []))
                        remaining = int(MAX_EXPR_LENGTH) - int(len(tokens))
                        if remaining < 0:
                            remaining = 0
                        # 只有当当前确实存在可选 Operator 时才收紧（避免把 action mask 收到全 False）
                        if bool(ret["select"][0]) and (stack_size > (remaining + 1)):
                            ret["select"][1] = False  # Features / Sub-expressions
                            ret["select"][2] = False  # Constants
                            ret["select"][3] = False  # Delta time
                    except Exception:
                        pass
                return ret

            _core_mod.AlphaEnvCore._valid_action_types = _valid_action_types_with_min_len  # type: ignore[assignment]
        except Exception as e:
            print(f"⚠ 设置 MIN_EXPR_LEN 失败（将忽略）：{e}")

    # ==================== 创建RL环境 ====================
    print("Setting up RL environment...")
    subexprs = None
    if subexprs_loaded_texts:
        # 解析文件中的 subexpr（保证动作映射与 checkpoint 一致）
        try:
            from alphagen.data.parser import parse_expression

            parsed = []
            for s in subexprs_loaded_texts:
                try:
                    parsed.append(parse_expression(s))
                except Exception:
                    continue
            subexprs = parsed[: int(subexprs_max)] if int(subexprs_max) > 0 else parsed
            print(f"✓ 子表达式库已从文件加载并解析: n={len(subexprs)}")
        except Exception as e:
            print(f"⚠ 解析子表达式列表失败（将回退为自动构建）：{e}")
            subexprs = None
    if subexprs is None:
        subexprs = _build_subexpr_library(subexprs_max)
    if subexprs:
        try:
            out_dir = Path("./alphagen_output")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "subexprs.json").write_text(
                json.dumps({"subexprs": [str(e) for e in subexprs]}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"✓ 子表达式库已生成: {out_dir / 'subexprs.json'}（n={len(subexprs)}）")
        except Exception as e:
            print(f"⚠ 子表达式库保存失败: {e}")

    env = AlphaEnv(pool=pool, device=device_obj, subexprs=subexprs)

    # ==================== 创建PPO模型 ====================
    print("Creating PPO model...")
    policy_kwargs = dict(
        features_extractor_class=LSTMSharedNet,
        features_extractor_kwargs=dict(
            n_layers=LSTM_LAYERS,
            d_model=LSTM_DIM,
            dropout=LSTM_DROPOUT,
            device=device_obj,
        )
    )

    resume_flag = os.environ.get("ALPHAGEN_RESUME", "0").strip() in {"1", "true", "yes", "y"}
    resume_path = os.environ.get("ALPHAGEN_RESUME_PATH", str(OUTPUT_DIR / "model_final.zip")).strip()
    resumed = bool(resume_flag and Path(resume_path).exists())
    if resumed:
        print(f"Resuming PPO model from: {resume_path}")
        model = MaskablePPO.load(
            resume_path,
            env=env,
            device=DEVICE,
        )
        # 允许在恢复训练时微调部分超参（不改网络结构）
        model.ent_coef = ENT_COEF
        model.target_kl = TARGET_KL
        model.n_epochs = N_EPOCHS
        # learning_rate 在 SB3 里可能是 schedule，这里不强行覆盖，避免产生误解
    else:
        if resume_flag and not Path(resume_path).exists():
            print(f"⚠ ALPHAGEN_RESUME=1 但未找到模型文件：{resume_path}（将从头训练）")
        model = MaskablePPO(
            'MlpPolicy',
            env,
            policy_kwargs=policy_kwargs,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            n_epochs=N_EPOCHS,
            gamma=0.99,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            target_kl=TARGET_KL,
            device=DEVICE,
            verbose=1,
            tensorboard_log=str(OUTPUT_DIR / 'tensorboard')
        )

    # ==================== 开始训练 ====================
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"Total timesteps: {TOTAL_TIMESTEPS}")
    print(f"Monitor training: tensorboard --logdir={OUTPUT_DIR / 'tensorboard'}")
    print()

    callbacks = [TensorboardCallback()]
    callbacks.append(AlphaCacheStatsCallback(calculator_obj=calculator, update_every=2048))
    if fast_gate_calc is not None:
        callbacks.append(FastGateStatsCallback(pool_obj=pool, update_every=2048))
        if float(FAST_GATE_MIN_ABS_IC_START) != float(FAST_GATE_MIN_ABS_IC_END):
            callbacks.append(
                FastGateThresholdScheduleCallback(
                    total_timesteps=FAST_GATE_MIN_ABS_IC_SCHEDULE_STEPS,
                    start_thr=float(FAST_GATE_MIN_ABS_IC_START),
                    end_thr=float(FAST_GATE_MIN_ABS_IC_END),
                    update_every=int(FAST_GATE_MIN_ABS_IC_UPDATE_EVERY),
                    holder=fast_gate_thr_holder,
                    verbose=0,
                )
            )
        if FAST_GATE_AUTOTUNE:
            callbacks.append(
                FastGateAutoTuneCallback(
                    pool_obj=pool,
                    holder=fast_gate_thr_holder,
                    target_skip_rate=float(FAST_GATE_AUTOTUNE_TARGET_SKIP),
                    adjust_mul=float(FAST_GATE_AUTOTUNE_ADJUST_MUL),
                    min_thr=float(FAST_GATE_AUTOTUNE_MIN_THR),
                    max_thr=float(FAST_GATE_AUTOTUNE_MAX_THR),
                    update_every=2048,
                    warmup_calls=50,
                    verbose=0,
                )
            )
    if val_gate_calc is not None:
        callbacks.append(ValGateStatsCallback(pool_obj=pool, update_every=2048))
    if PERF_LOG:
        callbacks.append(PoolPerfStatsCallback(pool_obj=pool, update_every=2048))
    if ckpt_every_steps > 0:
        callbacks.append(
            PeriodicCheckpointCallback(
                output_dir=OUTPUT_DIR,
                pool_obj=pool,
                every_steps=ckpt_every_steps,
                keep_last=ckpt_keep_last,
                feature_cols=list(feature_space.feature_cols),
                subexpr_texts=[str(e) for e in subexprs] if subexprs else None,
                verbose=0,
            )
        )
    # 动态最小长度（可选）：帮助解决“后期越跑越慢（评估过频）”以及“冷启动全 -1（表达式无效）”
    if (MIN_EXPR_LEN_START != MIN_EXPR_LEN_END) or (MIN_EXPR_LEN_START > 1):
        callbacks.append(
            MinExprLenScheduleCallback(
                total_timesteps=MIN_EXPR_LEN_SCHEDULE_STEPS,
                start_len=MIN_EXPR_LEN_START,
                end_len=MIN_EXPR_LEN_END,
                update_every=MIN_EXPR_LEN_UPDATE_EVERY,
                holder=min_expr_len_holder,
                verbose=0,
            )
        )
    if eval_every_steps > 0 and val_calculator_periodic is not None:
        callbacks.append(
            PeriodicValTestEvalCallback(
                pool=pool,
                val_calculator_obj=val_calculator_periodic,
                test_calculator_obj=test_calculator_periodic,
                eval_every_steps=eval_every_steps,
                eval_test=eval_test_flag,
                verbose=0,
            )
        )
    # LCB beta schedule（仅 meanstd）：beta 从负到正 ≈ “先探索不确定性、后追求稳定”
    if POOL_TYPE == "meanstd" and (POOL_LCB_BETA_START is not None) and (POOL_LCB_BETA_END is not None):
        if float(POOL_LCB_BETA_START) != float(POOL_LCB_BETA_END):
            callbacks.append(
                PoolLcbBetaScheduleCallback(
                    pool=pool,
                    total_timesteps=TOTAL_TIMESTEPS,
                    start_beta=float(POOL_LCB_BETA_START),
                    end_beta=float(POOL_LCB_BETA_END),
                    update_every=int(max(256, pool_lcb_beta_update_every)),
                    verbose=0,
                )
            )
    if ic_lb_start != ic_lb_end:
        callbacks.append(
            IcLowerBoundScheduleCallback(
                pool=pool,
                total_timesteps=TOTAL_TIMESTEPS,
                start_lb=ic_lb_start,
                end_lb=ic_lb_end,
                update_every=ic_lb_update_every,
            )
        )
    callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]

    # resume 时不要重置 timesteps（否则各种 schedule/eval 会从 0 重新计数，造成“看起来像新跑”）
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        reset_num_timesteps=(not resumed),
    )

    # ==================== 保存结果 ====================
    print("\n" + "=" * 60)
    print("Training complete! Saving results...")
    print("=" * 60)

    model_path = OUTPUT_DIR / 'model_final'
    pool_path = OUTPUT_DIR / 'alpha_pool.json'

    model.save(str(model_path))
    _dump_json(pool_path, pool.to_json_dict())

    print(f"✓ Model saved: {model_path}")
    print(f"✓ Alpha pool saved: {pool_path}")
    print(f"✓ Best alphas: {pool.size}")

    # ==================== 验证集评估 ====================
    print("\n" + "=" * 60)
    print("Evaluating on validation set...")
    print("=" * 60)

    val_data = CryptoData(
        symbols=SYMBOLS,
        start_time=TRAIN_END,
        end_time=VAL_END,
        timeframe='1h',
        data_dir=DATA_DIR,
        max_backtrack_periods=100,
        max_future_periods=30,
        features=None,
        device=device_obj,
    )

    if POOL_TYPE == "meanstd":
        val_calculator = TensorQLibStockDataCalculator(val_data, target)
    else:
        val_calculator = QLibStockDataCalculator(val_data, target)

    if pool.size > 0:
        exprs = [e for e in pool.exprs[:pool.size] if e is not None]
        weights = list(pool.weights)
        ic, ric = val_calculator.calc_pool_all_ret(exprs, weights)
        print(f"Validation IC: {ic:.4f}")
        print(f"Validation Rank IC: {ric:.4f}")

        # 保存验证结果
        val_results = {
            'ic': float(ic),
            'rank_ic': float(ric),
            'n_factors': len(exprs),
            'factors': [str(expr) for expr in exprs],
            'weights': [float(w) for w in weights],
            'n_features_total': int(train_data.n_features),
            'feature_columns': feature_space.feature_cols,
        }

        with open(OUTPUT_DIR / 'validation_results.json', 'w', encoding="utf-8") as f:
            json.dump(val_results, f, indent=2)

        print(f"✓ Validation results saved: {OUTPUT_DIR / 'validation_results.json'}")
    else:
        print("⚠ No factors in pool")

    # ==================== 测试集评估 ====================
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    test_data = CryptoData(
        symbols=SYMBOLS,
        start_time=VAL_END,
        end_time=END_TIME,
        timeframe='1h',
        data_dir=DATA_DIR,
        max_backtrack_periods=100,
        max_future_periods=30,
        features=None,
        device=device_obj,
    )

    if POOL_TYPE == "meanstd":
        test_calculator = TensorQLibStockDataCalculator(test_data, target)
    else:
        test_calculator = QLibStockDataCalculator(test_data, target)

    if pool.size > 0:
        exprs = [e for e in pool.exprs[:pool.size] if e is not None]
        weights = list(pool.weights)
        ic, ric = test_calculator.calc_pool_all_ret(exprs, weights)
        print(f"Test IC: {ic:.4f}")
        print(f"Test Rank IC: {ric:.4f}")

        test_results = {
            'ic': float(ic),
            'rank_ic': float(ric),
            'n_factors': len(exprs),
            'factors': [str(expr) for expr in exprs],
            'weights': [float(w) for w in weights],
            'n_features_total': int(train_data.n_features),
            'feature_columns': feature_space.feature_cols,
            'test_start': str(VAL_END),
            'test_end': str(END_TIME),
        }

        with open(OUTPUT_DIR / 'test_results.json', 'w', encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)

        print(f"✓ Test results saved: {OUTPUT_DIR / 'test_results.json'}")
    else:
        print("⚠ No factors in pool (skip test eval)")

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Review factors: cat {pool_path}")
    print(f"2. View training logs: tensorboard --logdir={OUTPUT_DIR / 'tensorboard'}")
    print(f"3. Backtest on test set: {VAL_END} -> {END_TIME}")


if __name__ == '__main__':
    main()
