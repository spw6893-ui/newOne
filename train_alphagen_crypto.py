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
import random
import re
import sys
import zlib
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


def _dump_json_atomic(path: Path, obj: dict) -> None:
    """
    原子写入 JSON（先写临时文件再 replace），用于 checkpoint 场景避免中断产生半文件。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _infer_resume_step(resume_path: Path, ckpt_dir: Path) -> Optional[int]:
    """
    推断 resume 的 step（用于对齐 features/subexprs 快照，避免 observation_space mismatch）。
    优先级：
    1) 文件名包含 model_step_{step}.zip
    2) checkpoints/latest.json 里的 step
    """
    try:
        m = re.search(r"model_step_(\\d+)\\.zip$", str(resume_path.name))
        if m:
            step = int(m.group(1))
            if step > 0:
                return step
    except Exception:
        pass

    try:
        latest = ckpt_dir / "latest.json"
        if latest.exists():
            obj = json.loads(latest.read_text(encoding="utf-8"))
            step = int(obj.get("step", 0) or 0)
            if step > 0:
                return step
    except Exception:
        pass

    return None


def _pick_latest_step_snapshot_leq(ckpt_dir: Path, prefix: str, step: int) -> Optional[Path]:
    """
    从 ckpt_dir 下选择 `prefix_{n}.json` 中 n<=step 的最近一份快照。

    例：
      - prefix=features_step, step=420000 -> features_step_400000.json（若 420000 不存在）
    """
    try:
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            return None
        step = int(step)
        if step <= 0:
            return None
    except Exception:
        return None

    best: Optional[tuple[int, Path]] = None
    pat = re.compile(rf"^{re.escape(prefix)}_(\\d+)\\.json$")
    for fp in ckpt_dir.glob(f"{prefix}_*.json"):
        m = pat.match(fp.name)
        if not m:
            continue
        try:
            s = int(m.group(1))
        except Exception:
            continue
        if s <= step and (best is None or s > best[0]):
            best = (s, fp)
    return best[1] if best else None


def _load_resume_feature_cols(output_dir: Path, ckpt_dir: Path, resume_step: Optional[int]) -> Optional[List[str]]:
    """
    尝试加载“训练时实际使用的特征列顺序”，用于 resume 时保持 observation/action space 一致。
    """
    # 1) checkpoints/features_step_{<=step}.json
    if resume_step is not None and resume_step > 0:
        fp = _pick_latest_step_snapshot_leq(ckpt_dir, "features_step", int(resume_step))
        if fp and fp.exists():
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
                feats = obj.get("features", obj.get("feature_cols", []))
                feats = [str(x).strip() for x in (feats or []) if str(x).strip()]
                if feats:
                    return feats
            except Exception:
                pass

    # 2) alphagen_output/selected_features.json（特征预筛选输出）
    fp2 = output_dir / "selected_features.json"
    if fp2.exists():
        try:
            obj = json.loads(fp2.read_text(encoding="utf-8"))
            feats = obj.get("features", obj.get("feature_cols", []))
            feats = [str(x).strip() for x in (feats or []) if str(x).strip()]
            if feats:
                return feats
        except Exception:
            pass

    return None


def _load_resume_subexpr_strs(output_dir: Path, ckpt_dir: Path, resume_step: Optional[int]) -> Optional[List[str]]:
    """
    尝试加载“训练时实际使用的子表达式库”（字符串形式），用于 resume 时保持 action space 一致。
    """
    # 1) checkpoints/subexprs_step_{<=step}.json
    if resume_step is not None and resume_step > 0:
        fp = _pick_latest_step_snapshot_leq(ckpt_dir, "subexprs_step", int(resume_step))
        if fp and fp.exists():
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
                subs = obj.get("subexprs", [])
                subs = [str(x).strip() for x in (subs or []) if str(x).strip()]
                if subs:
                    return subs
            except Exception:
                pass

    # 2) alphagen_output/subexprs.json（当前 run 自动生成的库）
    fp2 = output_dir / "subexprs.json"
    if fp2.exists():
        try:
            obj = json.loads(fp2.read_text(encoding="utf-8"))
            subs = obj.get("subexprs", [])
            subs = [str(x).strip() for x in (subs or []) if str(x).strip()]
            if subs:
                return subs
        except Exception:
            pass

    return None


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
    output_dir = Path("./alphagen_output")
    ckpt_dir_for_resume = Path(os.environ.get("ALPHAGEN_CHECKPOINT_DIR", str(output_dir / "checkpoints")).strip() or str(output_dir / "checkpoints"))

    resume_flag_raw = os.environ.get("ALPHAGEN_RESUME", "0").strip().lower()
    RESUME_FLAG = resume_flag_raw in {"1", "true", "yes", "y", "on"}
    resume_path = Path(os.environ.get("ALPHAGEN_RESUME_PATH", str(output_dir / "model_final.zip")).strip() or str(output_dir / "model_final.zip"))
    resume_step_env = int(os.environ.get("ALPHAGEN_RESUME_STEP", "0").strip() or 0)
    resume_step = int(resume_step_env) if resume_step_env > 0 else None
    if RESUME_FLAG and resume_step is None:
        resume_step = _infer_resume_step(resume_path=resume_path, ckpt_dir=ckpt_dir_for_resume)

    feature_space = _detect_feature_space(Path(DATA_DIR))
    resume_feature_cols = _load_resume_feature_cols(output_dir=output_dir, ckpt_dir=ckpt_dir_for_resume, resume_step=resume_step) if RESUME_FLAG else None
    resume_subexpr_strs = _load_resume_subexpr_strs(output_dir=output_dir, ckpt_dir=ckpt_dir_for_resume, resume_step=resume_step) if RESUME_FLAG else None

    features_max = int(os.environ.get("ALPHAGEN_FEATURES_MAX", "0").strip() or 0)
    prune_corr = float(os.environ.get("ALPHAGEN_FEATURES_PRUNE_CORR", "0.95").strip() or 0.95)
    if resume_feature_cols:
        # resume 时优先使用历史快照，避免“同样的 features_max 但选择结果不同”导致 action/observation space 不一致。
        if len(resume_feature_cols) != len(feature_space.feature_cols):
            # 只做提示，不强制一致：旧 run 可能用了 intersection/union 等不同 schema
            print(
                f"✓ RESUME: 使用历史 features 快照（n={len(resume_feature_cols)}）以保持空间一致；"
                f"当前数据推断 n={len(feature_space.feature_cols)}"
            )
        feature_space = FeatureSpace(feature_cols=list(resume_feature_cols))
        # 关键：不要在 resume 时重新跑 IC 预筛选（否则选择顺序变化就会导致 mismatch）
        features_max = 0
    if features_max > 0:
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
            out_dir = output_dir
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

    # 现在再 import alphagen（确保 action space 读到的是动态 FeatureType）
    # 注意：当前 alphagen 版本没有 Close()/Open() 这类快捷构造器，使用 Feature(FeatureType.X) 即可。
    #
    # 兼容：alphagen 上游 rolling Std/Var 在窗口=1 时会触发 dof<=0 警告并产生 NaN（unbiased=True 的默认行为）。
    # 这里做一次运行时 monkey patch，避免需要修改 submodule 指针（否则会导致他人无法拉取特定 commit）。
    #
    # 架构提升（可选，默认关闭）：
    # - 增加“截面算子”到 action space：CSRank / CSZScore
    # - 这会改变 action_space 大小：旧模型无法 resume（必须从头训练）
    enable_cs_ops_raw = os.environ.get("ALPHAGEN_ENABLE_CS_OPS", "0").strip().lower()
    ENABLE_CS_OPS = enable_cs_ops_raw in {"1", "true", "yes", "y", "on"}

    import alphagen.data.expression as _expr_mod

    def _std_apply_unbiased_false(self, operand):  # type: ignore[no-redef]
        return operand.std(dim=-1, unbiased=False)

    def _var_apply_unbiased_false(self, operand):  # type: ignore[no-redef]
        return operand.var(dim=-1, unbiased=False)

    _expr_mod.Std._apply = _std_apply_unbiased_false  # type: ignore[assignment]
    _expr_mod.Var._apply = _var_apply_unbiased_false  # type: ignore[assignment]

    # 注册 CSZScore（用于 parse_expression + 可选 action space）
    if not hasattr(_expr_mod, "CSZScore"):

        class CSZScore(_expr_mod.UnaryOperator):
            """
            截面 ZScore：对每个时点在“币种维度”做标准化（忽略 NaN/inf）。
            输出保留 NaN 语义：该时点该币种输入不可用 => 输出为 NaN。
            """

            def _apply(self, operand: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                x = operand
                mask = (~torch.isfinite(x)) | torch.isnan(x)
                n = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
                x0 = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                x0 = x0 * (~mask)
                mean = x0.sum(dim=1, keepdim=True) / n
                xc = (x0 - mean) * (~mask)
                var = (xc * xc).sum(dim=1, keepdim=True) / n
                std = torch.sqrt(var)
                std_safe = torch.where(std > 1e-6, std, torch.ones_like(std))
                out = (torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std_safe
                out[mask] = torch.nan
                return out

        _expr_mod.CSZScore = CSZScore  # type: ignore[attr-defined]

    # 让 parser 认识 CSZScore（不改变 action space）
    try:
        ops_list = getattr(_expr_mod, "Operators", None)
        if isinstance(ops_list, list) and _expr_mod.CSZScore not in ops_list:  # type: ignore[attr-defined]
            ops_list.insert(0, _expr_mod.CSZScore)  # type: ignore[attr-defined]
    except Exception:
        pass

    # 可选：把截面算子加入 action space（会改变 action_space => 不兼容旧模型 resume）
    import alphagen.config as _cfg
    if ENABLE_CS_OPS:
        resume_raw = os.environ.get("ALPHAGEN_RESUME", "0").strip().lower()
        if resume_raw in {"1", "true", "yes", "y", "on"}:
            print("⚠ 已开启 ALPHAGEN_ENABLE_CS_OPS=1：这会改变 action_space，旧模型无法 resume，请从头训练。")

        # 让 RL 能直接生成截面算子（默认 alphagen.config.OPERATORS 不包含它们）
        extra_ops = []
        if hasattr(_expr_mod, "CSRank"):
            extra_ops.append(_expr_mod.CSRank)  # type: ignore[attr-defined]
        extra_ops.append(_expr_mod.CSZScore)  # type: ignore[attr-defined]

        # 插到 Unary 区域（Abs/Log 后），保持 token 语义相对稳定
        insert_pos = 2 if len(_cfg.OPERATORS) >= 2 else len(_cfg.OPERATORS)
        for op in extra_ops:
            if op in _cfg.OPERATORS:
                continue
            _cfg.OPERATORS.insert(insert_pos, op)
            insert_pos += 1

    # alphagen wrapper 的 state dtype 默认是 uint8，因此 action_space 不能超过 255（否则会溢出）
    import alphagen_qlib.stock_data as sd
    base_action = len(_cfg.OPERATORS) + len(sd.FeatureType) + len(_cfg.DELTA_TIMES) + len(_cfg.CONSTANTS) + 1
    total_action = base_action + subexprs_max
    if total_action > 255:
        raise RuntimeError(
            f"action_space 过大：base={base_action}, subexprs_max={subexprs_max}, total={total_action}。"
            "这会导致 AlphaGen action_space>255（uint8 溢出）。请减少特征列/子表达式，或关闭 ALPHAGEN_ENABLE_CS_OPS。"
        )

    from alphagen.data.expression import Delta, EMA, Feature, Mean, Ref, Std, WMA
    from alphagen.data.pool_update import AddRemoveAlphas
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

    class Sb3LoggerMaxLengthCallback(BaseCallback):
        """
        规避 SB3 logger 的 key 截断冲突导致训练中断。

        背景：HumanOutputFormat 会把过长 key 截断到 max_length；
        截断后若发生重名，SB3 会抛 ValueError 并终止训练。
        """

        def __init__(self, max_length: int = 120, verbose: int = 0):
            super().__init__(verbose=verbose)
            self._max_length = int(max(36, max_length))

        def _on_training_start(self) -> None:
            try:
                fmts = getattr(self.logger, "output_formats", None)
                if not fmts:
                    return
                for fmt in fmts:
                    if hasattr(fmt, "max_length"):
                        try:
                            setattr(fmt, "max_length", self._max_length)
                        except Exception:
                            pass
            except Exception:
                return

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

        def _on_step(self) -> bool:
            if (self.num_timesteps % self._update_every) != 0:
                return True
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
            # 可选性能打点（默认关闭，设置 ALPHAGEN_PERF_LOG=1 开启）
            if "alpha_time_s" in st:
                alpha_calls = float(st.get("alpha_calls", 0))
                alpha_hit_calls = float(st.get("alpha_hit_calls", 0))
                alpha_miss_calls = float(st.get("alpha_miss_calls", 0))
                alpha_time_s = float(st.get("alpha_time_s", 0.0))
                alpha_hit_time_s = float(st.get("alpha_hit_time_s", 0.0))
                alpha_miss_time_s = float(st.get("alpha_miss_time_s", 0.0))
                self.logger.record("perf/alpha_time_s_total", alpha_time_s)
                if alpha_calls > 0:
                    self.logger.record("perf/alpha_ms_per_call", 1000.0 * alpha_time_s / alpha_calls)
                if alpha_hit_calls > 0:
                    self.logger.record("perf/alpha_hit_ms_per_call", 1000.0 * alpha_hit_time_s / alpha_hit_calls)
                if alpha_miss_calls > 0:
                    self.logger.record("perf/alpha_miss_ms_per_call", 1000.0 * alpha_miss_time_s / alpha_miss_calls)
            if "ic_time_s" in st:
                ic_calls = float(st.get("ic_calls", 0))
                ic_time_s = float(st.get("ic_time_s", 0.0))
                self.logger.record("perf/ic_time_s_total", ic_time_s)
                if ic_calls > 0:
                    self.logger.record("perf/ic_ms_per_call", 1000.0 * ic_time_s / ic_calls)
            return True

    class PoolPerfStatsCallback(BaseCallback):
        """
        记录 pool 侧的关键状态与耗时（用于定位 fps 衰减的根因）。

        说明：
        - 默认只记录轻量指标（pool/size, pool/eval_cnt）；
        - 若 pool 实例提供 perf_stats()（本脚本会在创建 pool 时按需注入），则额外记录耗时。
        """

        def __init__(self, pool_obj, update_every: int = 2048, verbose: int = 0):
            super().__init__(verbose=verbose)
            self._pool = pool_obj
            self._every = max(1, int(update_every))
            self._last = None

        @staticmethod
        def _safe_float(v) -> float:
            try:
                return float(v)
            except Exception:
                return 0.0

        def _on_step(self) -> bool:
            if (self.num_timesteps % self._every) != 0:
                return True
            # 让 pool 知道当前训练步数（用于 trial log / 代理 gate 的可追溯标注）
            try:
                setattr(self._pool, "_current_step", int(self.num_timesteps))
            except Exception:
                pass
            try:
                self.logger.record("pool/size", float(getattr(self._pool, "size", 0)))
                self.logger.record("pool/eval_cnt", float(getattr(self._pool, "eval_cnt", 0)))
            except Exception:
                pass

            fn = getattr(self._pool, "perf_stats", None)
            if fn is None:
                return True
            try:
                st = fn()
            except Exception:
                return True
            if not isinstance(st, dict):
                return True
            if self._last is None:
                self._last = dict(st)
                return True

            def delta(key: str) -> float:
                return self._safe_float(st.get(key, 0.0)) - self._safe_float(self._last.get(key, 0.0))

            # delta over interval
            te_calls = delta("try_new_expr_calls")
            te_time = delta("try_new_expr_time_s")
            opt_calls = delta("optimize_calls")
            opt_time = delta("optimize_time_s")
            ics_calls = delta("calc_ics_calls")
            ics_time = delta("calc_ics_time_s")
            fg_calls = delta("fast_gate_calls")
            fg_skips = delta("fast_gate_skips")
            fg_time = delta("fast_gate_time_s")
            sg_calls = delta("surrogate_calls")
            sg_skips = delta("surrogate_skips")
            sg_time = delta("surrogate_time_s")

            self.logger.record("perf/pool_try_new_expr_calls", te_calls)
            self.logger.record("perf/pool_try_new_expr_time_s", te_time)
            if te_calls > 0:
                self.logger.record("perf/pool_try_new_expr_ms_per_call", 1000.0 * te_time / te_calls)

            self.logger.record("perf/pool_optimize_calls", opt_calls)
            self.logger.record("perf/pool_optimize_time_s", opt_time)
            if opt_calls > 0:
                self.logger.record("perf/pool_optimize_ms_per_call", 1000.0 * opt_time / opt_calls)

            self.logger.record("perf/pool_calc_ics_calls", ics_calls)
            self.logger.record("perf/pool_calc_ics_time_s", ics_time)
            if ics_calls > 0:
                self.logger.record("perf/pool_calc_ics_ms_per_call", 1000.0 * ics_time / ics_calls)

            # FastGate
            self.logger.record("perf/fast_gate_calls", fg_calls)
            self.logger.record("perf/fast_gate_skips", fg_skips)
            self.logger.record("perf/fast_gate_time_s", fg_time)
            if fg_calls > 0:
                self.logger.record("perf/fast_gate_ms_per_call", 1000.0 * fg_time / fg_calls)

            # ValGate
            vg_calls = delta("val_gate_calls")
            vg_skips = delta("val_gate_skips")
            vg_time = delta("val_gate_time_s")
            self.logger.record("perf/val_gate_calls", vg_calls)
            self.logger.record("perf/val_gate_skips", vg_skips)
            self.logger.record("perf/val_gate_time_s", vg_time)
            if vg_calls > 0:
                self.logger.record("perf/val_gate_ms_per_call", 1000.0 * vg_time / vg_calls)

            # Surrogate Gate
            self.logger.record("perf/surrogate_calls", sg_calls)
            self.logger.record("perf/surrogate_skips", sg_skips)
            self.logger.record("perf/surrogate_time_s", sg_time)
            if sg_calls > 0:
                self.logger.record("perf/surrogate_ms_per_call", 1000.0 * sg_time / sg_calls)

            self._last = dict(st)
            return True

    class PeriodicPoolValPruneCallback(BaseCallback):
        """
        架构级改动：pool 满后进入平台期时，仅靠 try_new_expr 的“局部替换”很容易卡死。

        这里增加一个“周期性重筛/腾位”机制：
        - 每隔 N steps，用小样本验证集（val_gate_calc）快速估计当前 pool 每个因子的 single-IC；
        - 只保留 top-K（按 abs(val_ic_fast)）的因子，其余直接移除，让 pool 重新有空间容纳新因子；
        - 目的：强制 pool 持续更新，避免弱/同质因子长期占坑导致 val_ic 上限上不去。

        注意：
        - 该操作默认只在 pool 满时触发；
        - 评估使用小样本 val 子集，成本低，且触发频率低，对平均 fps 影响很小。
        """

        def __init__(
            self,
            pool,
            val_calc,
            every_steps: int,
            keep_top_k: int,
            min_abs_ic: float = 0.0,
            only_when_full: bool = True,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self._pool = pool
            self._val_calc = val_calc
            self._every = max(256, int(every_steps))
            self._keep_k = max(1, int(keep_top_k))
            self._min_abs_ic = float(max(0.0, min_abs_ic))
            self._only_full = bool(only_when_full)

        @staticmethod
        def _finite_abs(v: float) -> float:
            try:
                if not np.isfinite(v):
                    return -1.0
                return float(abs(v))
            except Exception:
                return -1.0

        def _on_step(self) -> bool:
            if (self.num_timesteps % self._every) != 0:
                return True
            if self._val_calc is None:
                return True
            try:
                size = int(getattr(self._pool, "size", 0) or 0)
                cap = int(getattr(self._pool, "capacity", 0) or 0)
            except Exception:
                return True
            if size <= 0:
                return True
            if self._only_full and cap > 0 and size < cap:
                return True

            import time as _time
            t0 = _time.perf_counter()

            vals = []
            for i in range(size):
                try:
                    expr = self._pool.exprs[i]  # type: ignore[index]
                except Exception:
                    expr = None
                if expr is None:
                    vals.append((i, float("nan")))
                    continue
                try:
                    ic = float(self._val_calc.calc_single_IC_ret(expr))
                except Exception:
                    ic = float("nan")
                vals.append((i, ic))

            order = sorted(vals, key=lambda x: self._finite_abs(x[1]), reverse=True)
            keep = [i for i, _ in order[: min(self._keep_k, len(order))]]
            if self._min_abs_ic > 0:
                extra = [i for i, ic in order if self._finite_abs(ic) >= self._min_abs_ic]
                for i in extra:
                    if i not in keep:
                        keep.append(i)
            keep = sorted(set(keep))
            if len(keep) >= size:
                return True

            try:
                self._pool.leave_only(keep)
                try:
                    setattr(self._pool, "_did_first_full_optimize", False)
                    setattr(self._pool, "_lazy_updates_since_opt", 0)
                except Exception:
                    pass
            except Exception:
                return True
            dt = _time.perf_counter() - t0

            self.logger.record("pool/prune_kept", float(len(keep)))
            self.logger.record("pool/prune_removed", float(max(0, size - len(keep))))
            self.logger.record("perf/pool_prune_time_s", float(dt))
            if self.verbose:
                print(f"✓ pool prune by val: kept={len(keep)}/{size}, dt={dt:.3f}s")
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
            print_on_test_error: bool = False,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self._pool = pool
            self._val_calc = val_calculator_obj
            self._test_calc = test_calculator_obj
            self._every = max(1, int(eval_every_steps))
            self._eval_test = bool(eval_test)
            self._print_on_test_error = bool(print_on_test_error)

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
            if (self.num_timesteps % self._every) != 0:
                return True

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
                    # 便于 TensorBoard 里判断“test 评估是否真的跑了”
                    self.logger.record("eval/test_eval_failed", 0.0)
                except Exception as e:
                    # 重要：test 评估若一直失败，旧实现会导致完全没有 eval/test_* tags，
                    # 用户容易误判为“ALPHAGEN_EVAL_TEST 没生效”。这里显式记录失败标记。
                    self.logger.record("eval/test_eval_failed", 1.0)
                    if self.verbose or self._print_on_test_error:
                        print(f"⚠ Test 评估失败：{e}")

            return True

    class PeriodicCheckpointCallback(BaseCallback):
        """
        周期性保存 checkpoint（模型 + alpha_pool）。

        用法：
        - ALPHAGEN_CHECKPOINT_EVERY_STEPS=100000  # 每 N step 保存一次（>0 开启）
        - ALPHAGEN_CHECKPOINT_KEEP=3             # 只保留最近 N 个
        - ALPHAGEN_CHECKPOINT_DIR=...            # 可选，自定义输出目录
        """

        def __init__(
            self,
            pool,
            ckpt_dir: Path,
            every_steps: int,
            feature_cols: Optional[Sequence[str]] = None,
            subexpr_strs: Optional[Sequence[str]] = None,
            keep_last: int = 3,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self._pool = pool
            self._dir = Path(ckpt_dir)
            self._dir.mkdir(parents=True, exist_ok=True)
            self._every = max(1, int(every_steps))
            self._keep = max(1, int(keep_last))
            self._last_saved_step = -1
            self._feature_cols = [str(x).strip() for x in (feature_cols or []) if str(x).strip()]
            self._subexpr_strs = [str(x).strip() for x in (subexpr_strs or []) if str(x).strip()]

        @staticmethod
        def _step_from_model_zip(p: Path) -> int:
            # model_step_123456.zip
            stem = p.stem
            try:
                return int(stem.split("_")[-1])
            except Exception:
                return -1

        def _cleanup(self) -> None:
            zips = list(self._dir.glob("model_step_*.zip"))
            if len(zips) <= self._keep:
                return
            zips.sort(key=self._step_from_model_zip)
            for zp in zips[: max(0, len(zips) - self._keep)]:
                st = self._step_from_model_zip(zp)
                try:
                    zp.unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    (self._dir / f"alpha_pool_step_{st}.json").unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    (self._dir / f"features_step_{st}.json").unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    (self._dir / f"subexprs_step_{st}.json").unlink(missing_ok=True)
                except Exception:
                    pass

        def _save(self, step: int, reason: str) -> None:
            if step <= 0 or step == self._last_saved_step:
                return
            try:
                model_base = self._dir / f"model_step_{step}"
                # SB3 会写成 model_step_{step}.zip
                self.model.save(str(model_base))  # type: ignore[attr-defined]
                _dump_json_atomic(self._dir / f"alpha_pool_step_{step}.json", self._pool.to_json_dict())
                # 关键：保存“当时的空间定义”（features/subexprs），保证后续 resume 不会因为 action/obs space 变化而失败
                if self._feature_cols:
                    _dump_json_atomic(self._dir / f"features_step_{step}.json", {"features": list(self._feature_cols)})
                if self._subexpr_strs:
                    _dump_json_atomic(self._dir / f"subexprs_step_{step}.json", {"subexprs": list(self._subexpr_strs)})
                _dump_json_atomic(
                    self._dir / "latest.json",
                    {
                        "step": int(step),
                        "reason": str(reason),
                        "model": f"model_step_{step}.zip",
                        "pool": f"alpha_pool_step_{step}.json",
                    },
                )
                self._cleanup()
                self._last_saved_step = int(step)
                self.logger.record("checkpoint/last_step", float(step))
            except Exception as e:
                # checkpoint 失败不应中断训练
                if self.verbose:
                    print(f"⚠ checkpoint 保存失败: step={step}, reason={reason}, err={e}")

        def _on_step(self) -> bool:
            if (self.num_timesteps % self._every) != 0:
                return True
            self._save(int(self.num_timesteps), reason="periodic")
            return True

        def save_now(self, reason: str = "manual") -> None:
            step = int(getattr(self.model, "num_timesteps", 0) or self.num_timesteps)  # type: ignore[attr-defined]
            self._save(step, reason=reason)

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
            schedule_steps: Optional[int] = None,
            warmup_steps: Optional[int] = None,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self.pool = pool
            self.total_timesteps = max(1, int(total_timesteps))
            if schedule_steps is None:
                self.schedule_steps = self.total_timesteps
            else:
                self.schedule_steps = max(1, int(schedule_steps))
            if warmup_steps is None:
                self.warmup_steps = 0
            else:
                self.warmup_steps = max(0, int(warmup_steps))
            self.start_lb = float(start_lb)
            self.end_lb = float(end_lb)
            self.update_every = max(1, int(update_every))
            self._last_lb: Optional[float] = None

        def _compute_lb(self) -> float:
            if self.num_timesteps <= self.warmup_steps:
                frac = 0.0
            else:
                frac = min(1.0, float(self.num_timesteps - self.warmup_steps) / float(self.schedule_steps))
            return self.start_lb + frac * (self.end_lb - self.start_lb)

        def _on_step(self) -> bool:
            if (self.num_timesteps % self.update_every) != 0:
                return True
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
            schedule_steps: Optional[int] = None,
            warmup_steps: Optional[int] = None,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self.pool = pool
            self.total_timesteps = max(1, int(total_timesteps))
            if schedule_steps is None:
                self.schedule_steps = self.total_timesteps
            else:
                self.schedule_steps = max(1, int(schedule_steps))
            if warmup_steps is None:
                self.warmup_steps = 0
            else:
                self.warmup_steps = max(0, int(warmup_steps))
            self.start_beta = float(start_beta)
            self.end_beta = float(end_beta)
            self.update_every = max(1, int(update_every))
            self._last: Optional[float] = None

        def _compute_beta(self) -> float:
            if self.num_timesteps <= self.warmup_steps:
                frac = 0.0
            else:
                frac = min(1.0, float(self.num_timesteps - self.warmup_steps) / float(self.schedule_steps))
            return self.start_beta + frac * (self.end_beta - self.start_beta)

        def _on_step(self) -> bool:
            if (self.num_timesteps % self.update_every) != 0:
                return True
            beta = float(self._compute_beta())
            setattr(self.pool, "_lcb_beta", beta)
            self.logger.record("pool/lcb_beta", beta)
            self._last = beta
            return True

    class MutualIcThresholdScheduleCallback(BaseCallback):
        """
        动态 mutual IC 阈值（越小越严格）：
        - 早期更宽松：更容易把 pool 填起来
        - 后期更严格：增强多样性，缓解 best_ic_ret/val_ic 平台期
        """

        def __init__(
            self,
            pool,
            start_thr: float,
            end_thr: float,
            update_every: int,
            schedule_steps: int,
            warmup_steps: int = 0,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self.pool = pool
            self.start_thr = float(start_thr)
            self.end_thr = float(end_thr)
            self.update_every = max(256, int(update_every))
            self.schedule_steps = max(1, int(schedule_steps))
            self.warmup_steps = max(0, int(warmup_steps))
            self._last: Optional[float] = None

        def _compute_thr(self) -> float:
            if self.num_timesteps <= self.warmup_steps:
                frac = 0.0
            else:
                frac = min(1.0, float(self.num_timesteps - self.warmup_steps) / float(self.schedule_steps))
            v = self.start_thr + frac * (self.end_thr - self.start_thr)
            if not np.isfinite(v):
                v = self.start_thr
            return float(max(0.0, min(0.9999, v)))

        def _on_step(self) -> bool:
            if (self.num_timesteps % self.update_every) != 0:
                return True
            thr = float(self._compute_thr())
            try:
                setattr(self.pool, "_mutual_ic_threshold", thr)
            except Exception:
                pass
            self.logger.record("pool/mutual_ic_threshold", thr)
            self._last = thr
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
            schedule_steps: Optional[int] = None,
            warmup_steps: Optional[int] = None,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self.total_timesteps = max(1, int(total_timesteps))
            # 允许把“长度课程”压缩到更短步数内完成（更快降低评估频率/提升 fps）
            if schedule_steps is None:
                self.schedule_steps = self.total_timesteps
            else:
                self.schedule_steps = max(1, int(schedule_steps))
            if warmup_steps is None:
                self.warmup_steps = 0
            else:
                self.warmup_steps = max(0, int(warmup_steps))
            self.start_len = max(1, int(start_len))
            self.end_len = max(1, int(end_len))
            self.update_every = max(1, int(update_every))
            self.holder = holder
            self._last: Optional[int] = None

        def _compute_len(self) -> int:
            if self.num_timesteps <= self.warmup_steps:
                frac = 0.0
            else:
                frac = min(1.0, float(self.num_timesteps - self.warmup_steps) / float(self.schedule_steps))
            v = self.start_len + frac * (self.end_len - self.start_len)
            return max(1, int(round(v)))

        def _on_step(self) -> bool:
            if (self.num_timesteps % self.update_every) != 0:
                return True
            v = int(self._compute_len())
            self.holder["value"] = v
            self.logger.record("env/min_expr_len", float(v))
            self._last = v
            return True

    class PoolOptimizeEveryScheduleCallback(BaseCallback):
        """
        动态调整 Pool 的“权重优化频率”（optimize_every_updates）：
        - 训练早期：每次有新 alpha 入池都优化（更强训练信号，避免冷启动训不动）
        - 训练后期：降低优化频率（显著缓解 MeanStdAlphaPool.optimize() 带来的 fps 衰减）
        """

        def __init__(
            self,
            pool,
            total_timesteps: int,
            start_every: int,
            end_every: int,
            update_every: int,
            verbose: int = 0,
        ):
            super().__init__(verbose=verbose)
            self.pool = pool
            self.total_timesteps = max(1, int(total_timesteps))
            self.start_every = max(1, int(start_every))
            self.end_every = max(1, int(end_every))
            self.update_every = max(1, int(update_every))
            self._last: Optional[int] = None

        def _compute_every(self) -> int:
            frac = min(1.0, float(self.num_timesteps) / float(self.total_timesteps))
            v = self.start_every + frac * (self.end_every - self.start_every)
            return max(1, int(round(v)))

        def _on_step(self) -> bool:
            if (self.num_timesteps % self.update_every) != 0:
                return True
            v = int(self._compute_every())
            try:
                setattr(self.pool, "_optimize_every_updates", v)
            except Exception:
                return True
            self.logger.record("pool/optimize_every_updates", float(v))
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

    # ==================== Trial Log / Surrogate Gate（可选） ====================
    # 目标：
    # - 采集 expr -> single_ic 的训练样本（JSONL），用于训练代理模型
    # - （可选）在 pool 满后用代理模型做候选 gate，减少无效评估，把搜索预算用在更有希望的表达式上
    trial_log_raw = os.environ.get("ALPHAGEN_TRIAL_LOG", "0").strip().lower()
    TRIAL_LOG = trial_log_raw in {"1", "true", "yes", "y", "on"}
    TRIAL_LOG_PATH = os.environ.get("ALPHAGEN_TRIAL_LOG_PATH", str(OUTPUT_DIR / "expr_trials.jsonl")).strip()
    TRIAL_LOG_FLUSH_EVERY = int(os.environ.get("ALPHAGEN_TRIAL_LOG_FLUSH_EVERY", "256").strip() or 256)

    surrogate_gate_raw = os.environ.get("ALPHAGEN_SURROGATE_GATE", "0").strip().lower()
    SURROGATE_GATE = surrogate_gate_raw in {"1", "true", "yes", "y", "on"}
    SURROGATE_MODEL_PATH = os.environ.get("ALPHAGEN_SURROGATE_MODEL_PATH", str(OUTPUT_DIR / "surrogate_model.npz")).strip()
    SURROGATE_SCORE_THRESHOLD = float(os.environ.get("ALPHAGEN_SURROGATE_SCORE_THRESHOLD", "0.0").strip() or 0.0)
    SURROGATE_RANDOM_ACCEPT_PROB = float(os.environ.get("ALPHAGEN_SURROGATE_RANDOM_ACCEPT_PROB", "0.05").strip() or 0.05)
    surrogate_only_full_raw = os.environ.get("ALPHAGEN_SURROGATE_ONLY_WHEN_FULL", "1").strip().lower()
    SURROGATE_ONLY_WHEN_FULL = surrogate_only_full_raw in {"1", "true", "yes", "y", "on"}

    # ==================== Reward shaping（可选） ====================
    # 目标：把“新表达式对 pool 目标的边际贡献”对齐为 RL 回报信号，
    # 避免出现“pool 很快满，但 best 不再提升”的平台期。
    # - abs：保持历史行为（返回 new_obj）
    # - delta_best：返回 best_obj 的边际增量（<=0 记为 0；信号更稀疏但更对齐）
    REWARD_MODE = os.environ.get("ALPHAGEN_REWARD_MODE", "abs").strip().lower()
    if REWARD_MODE not in {"abs", "delta_best", "delta_obj"}:
        print(f"⚠ 未知 ALPHAGEN_REWARD_MODE={REWARD_MODE}（将回退到 abs）")
        REWARD_MODE = "abs"
    try:
        REWARD_SCALE = float(os.environ.get("ALPHAGEN_REWARD_SCALE", "1.0").strip() or 1.0)
    except Exception:
        REWARD_SCALE = 1.0

    # delta_best 的“延迟启用”条件（默认：pool 满后再启用；避免前期回报过稀疏导致训歪）
    try:
        DELTA_BEST_MIN_POOL_SIZE = int(os.environ.get("ALPHAGEN_DELTA_BEST_MIN_POOL_SIZE", "-1").strip() or -1)
    except Exception:
        DELTA_BEST_MIN_POOL_SIZE = -1
    try:
        DELTA_BEST_WARMUP_STEPS = int(os.environ.get("ALPHAGEN_DELTA_BEST_WARMUP_STEPS", "0").strip() or 0)
    except Exception:
        DELTA_BEST_WARMUP_STEPS = 0
    raw_beta = os.environ.get("ALPHAGEN_DELTA_BEST_MIN_LCB_BETA", "").strip()
    try:
        DELTA_BEST_MIN_LCB_BETA = float(raw_beta) if raw_beta else None
    except Exception:
        DELTA_BEST_MIN_LCB_BETA = None

    # Lazy 模式超容量淘汰策略：
    # - weight：按 |weight| 最小淘汰（更接近 LinearAlphaPool 的“边际贡献小则踢出”直觉）
    # - single_ic：按 |single_ic| 最小淘汰（旧行为；更快但更容易留下冗余、导致平台期）
    LAZY_REMOVE_BY = os.environ.get("ALPHAGEN_POOL_LAZY_REMOVE_BY", "weight").strip().lower()
    if LAZY_REMOVE_BY not in {"weight", "single_ic"}:
        print(f"⚠ 未知 ALPHAGEN_POOL_LAZY_REMOVE_BY={LAZY_REMOVE_BY}（将回退到 weight）")
        LAZY_REMOVE_BY = "weight"

    # SB3 logger key 最大长度（避免截断冲突导致训练中断）
    try:
        SB3_LOGGER_MAX_LENGTH = int(os.environ.get("ALPHAGEN_SB3_LOGGER_MAX_LENGTH", "120").strip() or 120)
    except Exception:
        SB3_LOGGER_MAX_LENGTH = 120

    class TrialLogger:
        def __init__(self, path: str, flush_every: int = 256):
            self._path = Path(path)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._buf: List[dict] = []
            self._flush_every = max(1, int(flush_every))

        def log(self, row: dict) -> None:
            self._buf.append(row)
            if len(self._buf) >= self._flush_every:
                self.flush()

        def flush(self) -> None:
            if not self._buf:
                return
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    for r in self._buf:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            except Exception:
                # 采集失败不应中断训练
                pass
            finally:
                self._buf.clear()

    _RE_OP = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(")
    _RE_FEAT = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
    _RE_INT = re.compile(r"(?:,|\()\s*(\d{1,5})\s*(?:\)|,)")

    def _stable_hash(s: str) -> int:
        return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF

    def _expr_to_ids(expr: str, dim: int) -> List[int]:
        ops = _RE_OP.findall(expr)
        feats = _RE_FEAT.findall(expr)
        ints = _RE_INT.findall(expr)
        f: List[str] = []
        f.append(f"len:{min(64, max(1, len(expr)//8))}")
        f.append(f"ops_cnt:{min(32, len(ops))}")
        f.append(f"feats_cnt:{min(32, len(feats))}")
        for op in ops[:64]:
            f.append(f"op:{op}")
        for ft in feats[:64]:
            f.append(f"feat:{ft}")
        for x in ints[:64]:
            f.append(f"int:{x}")
        for op in ops[:8]:
            for ft in feats[:8]:
                f.append(f"x:{op}|{ft}")
        idx = [int(_stable_hash(s) % dim) for s in f]
        return sorted(set(idx))

    class SurrogateScorer:
        def __init__(self, model_path: str):
            self._path = Path(model_path)
            self._dim = 0
            self._w = None
            self._bias = 0.0
            self._ok = False
            self._load()

        def _load(self) -> None:
            try:
                if not self._path.exists():
                    return
                d = np.load(self._path, allow_pickle=False)
                self._dim = int(d["dim"][0])
                self._w = d["w"].astype(np.float32, copy=False)
                self._bias = float(d["bias"][0])
                self._ok = bool(self._dim > 0 and self._w is not None)
            except Exception:
                self._ok = False

        def score(self, expr: str) -> float:
            if not self._ok or self._w is None:
                return float("nan")
            ids = _expr_to_ids(expr, int(self._dim))
            if not ids:
                return float(self._bias)
            return float(self._bias + float(self._w[ids].sum()))

        @property
        def enabled(self) -> bool:
            return bool(self._ok)

    trial_logger = TrialLogger(TRIAL_LOG_PATH, flush_every=TRIAL_LOG_FLUSH_EVERY) if TRIAL_LOG else None
    surrogate = SurrogateScorer(SURROGATE_MODEL_PATH) if SURROGATE_GATE else None
    if surrogate is not None and not surrogate.enabled:
        print(f"⚠ Surrogate gate 已请求开启，但未找到/无法加载模型：{SURROGATE_MODEL_PATH}（将忽略）")
        surrogate = None

    # Alpha Pool配置 - 针对高维特征优化
    POOL_CAPACITY = int(os.environ.get("ALPHAGEN_POOL_CAPACITY", "10"))
    IC_LOWER_BOUND = float(os.environ.get("ALPHAGEN_IC_LOWER_BOUND", "0.01"))
    # 默认沿用 alphagen 的单因子筛选逻辑：只接受 single_ic >= lower_bound（会拒绝负 IC）。
    # 对于 crypto/可做空组合，更合理的是接受 abs(single_ic) >= lower_bound，让 optimize 自己学符号。
    ic_lb_abs_raw = os.environ.get("ALPHAGEN_IC_LOWER_BOUND_ABS", "0").strip().lower()
    IC_LOWER_BOUND_ABS = ic_lb_abs_raw in {"1", "true", "yes", "y", "on"}
    # mutual IC 阈值：越大越“宽松”（更不容易因为相似被拒），但会让 mutual 计算更常走完（略慢）
    MUTUAL_IC_THRESHOLD = float(os.environ.get("ALPHAGEN_MUTUAL_IC_THRESHOLD", "0.99").strip() or 0.99)
    # 可选：mutual 阈值课程（早期宽松，后期收紧以增强多样性与泛化）
    mutual_thr_start = float(os.environ.get("ALPHAGEN_MUTUAL_IC_THRESHOLD_START", str(MUTUAL_IC_THRESHOLD)).strip() or MUTUAL_IC_THRESHOLD)
    mutual_thr_end = float(os.environ.get("ALPHAGEN_MUTUAL_IC_THRESHOLD_END", str(MUTUAL_IC_THRESHOLD)).strip() or MUTUAL_IC_THRESHOLD)
    mutual_thr_update_every = int(os.environ.get("ALPHAGEN_MUTUAL_IC_THRESHOLD_UPDATE_EVERY", "20000").strip() or 20000)
    mutual_thr_schedule_steps = int(
        os.environ.get("ALPHAGEN_MUTUAL_IC_THRESHOLD_SCHEDULE_STEPS", str(TOTAL_TIMESTEPS)).strip() or TOTAL_TIMESTEPS
    )
    mutual_thr_warmup_steps = int(os.environ.get("ALPHAGEN_MUTUAL_IC_THRESHOLD_WARMUP_STEPS", "0").strip() or 0)
    mutual_thr_update_every = max(256, int(mutual_thr_update_every))
    mutual_thr_schedule_steps = max(1, int(mutual_thr_schedule_steps))
    mutual_thr_warmup_steps = max(0, int(mutual_thr_warmup_steps))
    mutual_thr_start = float(max(0.0, min(0.9999, mutual_thr_start)))
    mutual_thr_end = float(max(0.0, min(0.9999, mutual_thr_end)))
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
    # 允许把 beta schedule 压缩到更短步数内完成（更早从 UCB 过渡到 LCB，提升 val/test 稳健性）
    pool_lcb_beta_schedule_steps = int(
        os.environ.get("ALPHAGEN_POOL_LCB_BETA_SCHEDULE_STEPS", str(TOTAL_TIMESTEPS)).strip() or TOTAL_TIMESTEPS
    )
    pool_lcb_beta_schedule_steps = max(1, int(pool_lcb_beta_schedule_steps))
    pool_lcb_beta_warmup_steps = int(os.environ.get("ALPHAGEN_POOL_LCB_BETA_WARMUP_STEPS", "0").strip() or 0)
    pool_lcb_beta_warmup_steps = max(0, int(pool_lcb_beta_warmup_steps))

    # 动态 threshold：start/end 任意一个被设置就启用（默认与 IC_LOWER_BOUND 相同 => 等价于关闭）
    ic_lb_start = float(os.environ.get("ALPHAGEN_IC_LOWER_BOUND_START", str(IC_LOWER_BOUND)))
    ic_lb_end = float(os.environ.get("ALPHAGEN_IC_LOWER_BOUND_END", str(IC_LOWER_BOUND)))
    ic_lb_update_every = int(os.environ.get("ALPHAGEN_IC_LOWER_BOUND_UPDATE_EVERY", "2048"))
    # 允许把 IC lower bound schedule 压缩到更短步数内完成（更早收紧质量门槛）
    ic_lb_schedule_steps = int(
        os.environ.get("ALPHAGEN_IC_LOWER_BOUND_SCHEDULE_STEPS", str(TOTAL_TIMESTEPS)).strip() or TOTAL_TIMESTEPS
    )
    ic_lb_schedule_steps = max(1, int(ic_lb_schedule_steps))
    ic_lb_warmup_steps = int(os.environ.get("ALPHAGEN_IC_LOWER_BOUND_WARMUP_STEPS", "0").strip() or 0)
    ic_lb_warmup_steps = max(0, int(ic_lb_warmup_steps))

    # 把关键运行参数落盘，便于确认“你实际跑的是什么配置”（避免命令行覆盖失败却不自知）
    try:
        run_cfg = {
            "seed": SEED,
            "total_timesteps": TOTAL_TIMESTEPS,
            "batch_size": BATCH_SIZE,
            "n_steps": int(os.environ.get("ALPHAGEN_N_STEPS", "2048")),
            "n_epochs": int(os.environ.get("ALPHAGEN_N_EPOCHS", "10")),
            "learning_rate": float(os.environ.get("ALPHAGEN_LEARNING_RATE", "3e-4")),
            "clip_range": float(os.environ.get("ALPHAGEN_CLIP_RANGE", "0.2")),
            "target_kl": TARGET_KL,
            "pool": {
                "type": POOL_TYPE,
                "capacity": POOL_CAPACITY,
                "l1_alpha": L1_ALPHA,
                "ic_lower_bound": IC_LOWER_BOUND,
                "ic_lower_bound_start": ic_lb_start,
                "ic_lower_bound_end": ic_lb_end,
                "ic_lower_bound_update_every": ic_lb_update_every,
                "ic_lower_bound_schedule_steps": int(ic_lb_schedule_steps),
                "ic_lower_bound_warmup_steps": int(ic_lb_warmup_steps),
                "ic_lower_bound_abs": bool(IC_LOWER_BOUND_ABS),
                "mutual_ic_threshold": float(MUTUAL_IC_THRESHOLD),
                "mutual_ic_threshold_start": float(mutual_thr_start),
                "mutual_ic_threshold_end": float(mutual_thr_end),
                "mutual_ic_threshold_update_every": int(mutual_thr_update_every),
                "mutual_ic_threshold_schedule_steps": int(mutual_thr_schedule_steps),
                "mutual_ic_threshold_warmup_steps": int(mutual_thr_warmup_steps),
                "lcb_beta": POOL_LCB_BETA,
                "lcb_beta_start": POOL_LCB_BETA_START,
                "lcb_beta_end": POOL_LCB_BETA_END,
                "lcb_beta_update_every": int(pool_lcb_beta_update_every),
                "lcb_beta_schedule_steps": int(pool_lcb_beta_schedule_steps),
                "lcb_beta_warmup_steps": int(pool_lcb_beta_warmup_steps),
            },
            "trial_log": {
                "enabled": bool(TRIAL_LOG),
                "path": str(TRIAL_LOG_PATH),
            },
            "surrogate": {
                "enabled": bool(SURROGATE_GATE and (surrogate is not None)),
                "model_path": str(SURROGATE_MODEL_PATH),
                "score_threshold": float(SURROGATE_SCORE_THRESHOLD),
                "random_accept_prob": float(SURROGATE_RANDOM_ACCEPT_PROB),
                "only_when_full": bool(SURROGATE_ONLY_WHEN_FULL),
            },
            "val_gate": {
                "enabled": bool(VAL_GATE and (val_gate_calc is not None)),
                "min_abs_ic": float(VAL_GATE_MIN_ABS_IC),
                "only_when_full": bool(VAL_GATE_ONLY_WHEN_FULL),
                "symbols": int(VAL_GATE_SYMBOLS),
                "periods": int(VAL_GATE_PERIODS),
            },
        }
        (OUTPUT_DIR / "run_config.json").write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ 运行配置已保存: {OUTPUT_DIR / 'run_config.json'}")
    except Exception:
        pass

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
    # 可选：把“最小长度课程”在更短 steps 内完成（例如 150k 内从 1->12）
    # 目的：更快降低评估频率，避免 200k 后 fps 过低。
    min_expr_len_sched_steps_raw = os.environ.get("ALPHAGEN_MIN_EXPR_LEN_SCHEDULE_STEPS", "").strip()
    MIN_EXPR_LEN_SCHEDULE_STEPS: Optional[int]
    if not min_expr_len_sched_steps_raw:
        MIN_EXPR_LEN_SCHEDULE_STEPS = None
    else:
        try:
            v = int(float(min_expr_len_sched_steps_raw))
            MIN_EXPR_LEN_SCHEDULE_STEPS = None if v <= 0 else v
        except Exception:
            MIN_EXPR_LEN_SCHEDULE_STEPS = None

    # 可选：长度课程 warmup（前 N steps 固定为 start_len，用于解决冷启动 -1/15 卡死）
    min_expr_len_warmup_raw = os.environ.get("ALPHAGEN_MIN_EXPR_LEN_WARMUP_STEPS", "").strip()
    MIN_EXPR_LEN_WARMUP_STEPS: Optional[int]
    if not min_expr_len_warmup_raw:
        MIN_EXPR_LEN_WARMUP_STEPS = None
    else:
        try:
            v = int(float(min_expr_len_warmup_raw))
            MIN_EXPR_LEN_WARMUP_STEPS = None if v <= 0 else v
        except Exception:
            MIN_EXPR_LEN_WARMUP_STEPS = None
    min_expr_len_holder = {"value": int(MIN_EXPR_LEN_START)}
    stack_guard_raw = os.environ.get("ALPHAGEN_STACK_GUARD", "1").strip().lower()
    STACK_GUARD = stack_guard_raw in {"1", "true", "yes", "y", "on"}

    # 性能控制：pool 权重优化上限（MseAlphaPool 的 Adam 优化默认 max_steps=10000 很重）
    # 仅对 POOL_TYPE=mse 生效；MeanStdAlphaPool 有自己的一套优化。
    POOL_OPT_LR = float(os.environ.get("ALPHAGEN_POOL_OPT_LR", "5e-4"))
    POOL_OPT_MAX_STEPS = int(os.environ.get("ALPHAGEN_POOL_OPT_MAX_STEPS", "10000"))
    POOL_OPT_TOLERANCE = int(os.environ.get("ALPHAGEN_POOL_OPT_TOLERANCE", "500"))
    POOL_OPT_MAX_STEPS = max(50, POOL_OPT_MAX_STEPS)
    POOL_OPT_TOLERANCE = max(10, POOL_OPT_TOLERANCE)

    # 性能控制：pool 权重优化频率（默认=1，保持原始行为：每次入池都 optimize）
    # 建议：MeanStdAlphaPool 在后期 fps 掉得厉害时，逐步拉到 4~12（能显著改善）
    POOL_OPT_EVERY_UPDATES = int(os.environ.get("ALPHAGEN_POOL_OPT_EVERY_UPDATES", "1").strip() or 1)
    POOL_OPT_EVERY_UPDATES = max(1, POOL_OPT_EVERY_UPDATES)
    POOL_OPT_EVERY_UPDATES_START = int(os.environ.get("ALPHAGEN_POOL_OPT_EVERY_UPDATES_START", str(POOL_OPT_EVERY_UPDATES)).strip() or POOL_OPT_EVERY_UPDATES)
    POOL_OPT_EVERY_UPDATES_END = int(os.environ.get("ALPHAGEN_POOL_OPT_EVERY_UPDATES_END", str(POOL_OPT_EVERY_UPDATES)).strip() or POOL_OPT_EVERY_UPDATES)
    POOL_OPT_EVERY_UPDATES_START = max(1, POOL_OPT_EVERY_UPDATES_START)
    POOL_OPT_EVERY_UPDATES_END = max(1, POOL_OPT_EVERY_UPDATES_END)
    POOL_OPT_EVERY_UPDATES_UPDATE_EVERY = int(os.environ.get("ALPHAGEN_POOL_OPT_EVERY_UPDATES_UPDATE_EVERY", "20000").strip() or 20000)
    POOL_OPT_EVERY_UPDATES_UPDATE_EVERY = max(256, POOL_OPT_EVERY_UPDATES_UPDATE_EVERY)

    # 周期性评估（默认关闭，>0 开启）
    eval_every_steps = int(os.environ.get("ALPHAGEN_EVAL_EVERY_STEPS", "0").strip() or 0)
    eval_test_flag = os.environ.get("ALPHAGEN_EVAL_TEST", "1").strip().lower() in {"1", "true", "yes", "y"}
    eval_test_print_err = os.environ.get("ALPHAGEN_EVAL_TEST_PRINT_ERROR", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    # 近似评估（Fast Gate）：在 pool 已满时，先用“更小的训练子集”计算 single-IC，
    # 不达标则跳过完整评估（节省大量 mutual IC / optimize 的开销）。
    fast_gate_raw = os.environ.get("ALPHAGEN_FAST_GATE", "0").strip().lower()
    FAST_GATE = fast_gate_raw in {"1", "true", "yes", "y", "on"}
    fast_gate_only_full_raw = os.environ.get("ALPHAGEN_FAST_GATE_ONLY_WHEN_FULL", "1").strip().lower()
    FAST_GATE_ONLY_WHEN_FULL = fast_gate_only_full_raw in {"1", "true", "yes", "y", "on"}
    FAST_GATE_SYMBOLS = int(os.environ.get("ALPHAGEN_FAST_GATE_SYMBOLS", "20").strip() or 20)
    FAST_GATE_PERIODS = int(os.environ.get("ALPHAGEN_FAST_GATE_PERIODS", "4000").strip() or 4000)  # 1h bars
    FAST_GATE_MIN_ABS_IC = float(os.environ.get("ALPHAGEN_FAST_GATE_MIN_ABS_IC", "0.003").strip() or 0.003)
    FAST_GATE_SYMBOLS = max(4, FAST_GATE_SYMBOLS)
    FAST_GATE_PERIODS = max(256, FAST_GATE_PERIODS)
    FAST_GATE_MIN_ABS_IC = max(0.0, FAST_GATE_MIN_ABS_IC)

    # ValGate（方法替换的关键）：用“小样本验证集”估计 single-IC，引导搜索朝 val 泛化方向走
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

    # Pool prune（架构增强）：周期性用 val 小样本重筛现有 pool，主动腾位避免平台期
    pool_prune_raw = os.environ.get("ALPHAGEN_POOL_PRUNE_BY_VAL", "0").strip().lower()
    POOL_PRUNE_BY_VAL = pool_prune_raw in {"1", "true", "yes", "y", "on"}
    pool_prune_only_full_raw = os.environ.get("ALPHAGEN_POOL_PRUNE_ONLY_WHEN_FULL", "1").strip().lower()
    POOL_PRUNE_ONLY_WHEN_FULL = pool_prune_only_full_raw in {"1", "true", "yes", "y", "on"}
    POOL_PRUNE_EVERY_STEPS = int(os.environ.get("ALPHAGEN_POOL_PRUNE_EVERY_STEPS", "50000").strip() or 50000)
    POOL_PRUNE_KEEP_TOP_K = int(os.environ.get("ALPHAGEN_POOL_PRUNE_KEEP_TOP_K", "20").strip() or 20)
    POOL_PRUNE_MIN_ABS_IC = float(os.environ.get("ALPHAGEN_POOL_PRUNE_MIN_ABS_IC", "0.0").strip() or 0.0)
    POOL_PRUNE_EVERY_STEPS = max(256, int(POOL_PRUNE_EVERY_STEPS))
    POOL_PRUNE_KEEP_TOP_K = max(1, int(POOL_PRUNE_KEEP_TOP_K))
    POOL_PRUNE_MIN_ABS_IC = max(0.0, float(POOL_PRUNE_MIN_ABS_IC))

    # 周期性保存 checkpoint（默认关闭，>0 开启）
    ckpt_every_steps = int(os.environ.get("ALPHAGEN_CHECKPOINT_EVERY_STEPS", "0").strip() or 0)
    ckpt_keep_last = int(os.environ.get("ALPHAGEN_CHECKPOINT_KEEP", "3").strip() or 3)
    ckpt_dir_raw = os.environ.get("ALPHAGEN_CHECKPOINT_DIR", str(OUTPUT_DIR / "checkpoints")).strip()
    ckpt_dir = Path(ckpt_dir_raw) if ckpt_dir_raw else (OUTPUT_DIR / "checkpoints")

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
            f"(every {MIN_EXPR_LEN_UPDATE_EVERY} steps)"
        )
        if MIN_EXPR_LEN_SCHEDULE_STEPS is not None:
            print(f"Min expr len schedule steps: {MIN_EXPR_LEN_SCHEDULE_STEPS}")
        if MIN_EXPR_LEN_WARMUP_STEPS is not None:
            print(f"Min expr len warmup steps: {MIN_EXPR_LEN_WARMUP_STEPS}")
    elif MIN_EXPR_LEN > 1:
        print(f"Min expr len: {MIN_EXPR_LEN}（将延迟允许 SEP，减少评估次数以提速）")
    print(f"Stack guard: {'ON' if STACK_GUARD else 'OFF'}（避免栈过深导致最终表达式无效 => reward=-1）")
    print(f"Pool optimize: lr={POOL_OPT_LR}, max_steps={POOL_OPT_MAX_STEPS}, tol={POOL_OPT_TOLERANCE}")
    if POOL_OPT_EVERY_UPDATES_START != POOL_OPT_EVERY_UPDATES_END or POOL_OPT_EVERY_UPDATES_START != 1:
        if POOL_OPT_EVERY_UPDATES_START != POOL_OPT_EVERY_UPDATES_END:
            print(
                "Pool optimize schedule: "
                f"every_updates {POOL_OPT_EVERY_UPDATES_START}->{POOL_OPT_EVERY_UPDATES_END} "
                f"(every {POOL_OPT_EVERY_UPDATES_UPDATE_EVERY} env steps)"
            )
        else:
            print(f"Pool optimize every_updates: {POOL_OPT_EVERY_UPDATES_START}")
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

    # Fast Gate：构造一个更小的训练子集，仅用于 single-IC 粗筛（不改变最终 IC 口径）
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

    # ValGate：优先用 val 子集估计 single-IC，避免“train 很好但 val 很差”的过拟合候选占用计算预算
    # 同时：若启用 POOL_PRUNE_BY_VAL，也会复用同一份 val_gate_calc 作为快速重筛依据。
    val_gate_calc = None
    if VAL_GATE or POOL_PRUNE_BY_VAL:
        try:
            # 如果用户开启了周期 eval，则 val_data_periodic 后面会创建；这里为避免双份加载，优先复用该对象。
            # 若 eval 未开启，则此处单独加载一份 val 数据（仅用于 gate）。
            if eval_every_steps > 0:
                val_data_for_gate = None  # 先占位，等 val_data_periodic 初始化后再补齐
            else:
                val_data_for_gate = CryptoData(
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
            # 后续若 val_data_periodic 不为 None，会覆盖 val_data_for_gate
            _val_gate_need_init = True
        except Exception as e:
            print(f"⚠ ValGate 预初始化失败（将禁用）：{e}")
            val_gate_calc = None
            _val_gate_need_init = False
    else:
        _val_gate_need_init = False

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

    # 在 val 数据加载完成后，再初始化 ValGate（避免重复加载）
    if _val_gate_need_init and val_gate_calc is None:
        try:
            if eval_every_steps > 0:
                val_data_for_gate = val_data_periodic
            if val_data_for_gate is None:
                raise RuntimeError("ValGate 缺少 val 数据（val_data_for_gate=None）")

            total_symbols = int(getattr(val_data_for_gate, "n_stocks", 0))
            take_symbols = min(int(VAL_GATE_SYMBOLS), max(1, total_symbols))
            rng = np.random.default_rng(SEED + 7)
            idx = np.array(sorted(rng.choice(total_symbols, size=take_symbols, replace=False).tolist()), dtype=np.int64)

            base_tensor = val_data_for_gate.data
            need_len = int(
                VAL_GATE_PERIODS
                + val_data_for_gate.max_backtrack_days
                + val_data_for_gate.max_future_days
                + 1
            )
            start = max(0, int(base_tensor.shape[0]) - need_len)
            val_tensor = base_tensor[start:, :, :].index_select(2, torch.tensor(idx, device=base_tensor.device))
            val_view = _StockDataView(val_data_for_gate, val_tensor)

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
        perf_raw = os.environ.get("ALPHAGEN_PERF_LOG", "0").strip().lower()
        PERF_LOG = perf_raw in {"1", "true", "yes", "y", "on"}

        class TunableNaNFriendlyMeanStdAlphaPool(NaNFriendlyMeanStdAlphaPool):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._perf_enabled = bool(PERF_LOG)
                self._perf = {
                    "try_new_expr_calls": 0,
                    "try_new_expr_time_s": 0.0,
                    "calc_ics_calls": 0,
                    "calc_ics_time_s": 0.0,
                    "optimize_calls": 0,
                    "optimize_time_s": 0.0,
                    "fast_gate_calls": 0,
                    "fast_gate_skips": 0,
                    "fast_gate_time_s": 0.0,
                    "val_gate_calls": 0,
                    "val_gate_skips": 0,
                    "val_gate_time_s": 0.0,
                    "surrogate_calls": 0,
                    "surrogate_skips": 0,
                    "surrogate_time_s": 0.0,
                }
                self._optimize_every_updates = int(POOL_OPT_EVERY_UPDATES_START)
                self._lazy_updates_since_opt = 0
                self._did_first_full_optimize = False
                self._fast_gate_calc = fast_gate_calc
                self._fast_gate_only_full = bool(FAST_GATE_ONLY_WHEN_FULL)
                self._fast_gate_min_abs_ic = float(FAST_GATE_MIN_ABS_IC)
                self._trial_logger = trial_logger
                self._val_gate_calc = val_gate_calc
                self._val_gate_only_full = bool(VAL_GATE_ONLY_WHEN_FULL)
                self._val_gate_min_abs_ic = float(VAL_GATE_MIN_ABS_IC)
                self._surrogate = surrogate
                self._surrogate_thr = float(SURROGATE_SCORE_THRESHOLD)
                self._surrogate_rand = float(SURROGATE_RANDOM_ACCEPT_PROB)
                self._surrogate_only_full = bool(SURROGATE_ONLY_WHEN_FULL)
                self._last_single_ic = float("nan")
                self._last_mutual_max = float("nan")
                self._last_val_gate_ic = float("nan")
                self._reward_mode = str(REWARD_MODE)
                self._reward_scale = float(REWARD_SCALE)
                self._reward_prev_obj: Optional[float] = None
                self._delta_best_min_pool_size = int(DELTA_BEST_MIN_POOL_SIZE)
                self._delta_best_warmup_steps = int(DELTA_BEST_WARMUP_STEPS)
                self._delta_best_min_lcb_beta = DELTA_BEST_MIN_LCB_BETA
                self._lazy_remove_by = str(LAZY_REMOVE_BY)

            def perf_stats(self) -> dict:
                return dict(self._perf) if getattr(self, "_perf_enabled", False) else {}

            def _shape_reward(self, old_best_obj: float, out_obj: float) -> float:
                mode = str(getattr(self, "_reward_mode", "abs"))
                scale = float(getattr(self, "_reward_scale", 1.0) or 1.0)
                base_obj = float(out_obj) if np.isfinite(out_obj) else 0.0
                if mode == "delta_obj":
                    prev = getattr(self, "_reward_prev_obj", None)
                    if prev is None or (not np.isfinite(prev)):
                        setattr(self, "_reward_prev_obj", float(base_obj))
                        return 0.0
                    d = float(base_obj) - float(prev)
                    setattr(self, "_reward_prev_obj", float(base_obj))
                    r = d * scale
                    return float(r) if np.isfinite(r) else 0.0
                if mode == "delta_best":
                    # 前期保留稠密回报，避免 delta 信号过稀疏导致训歪
                    min_pool = int(getattr(self, "_delta_best_min_pool_size", -1))
                    if min_pool < 0:
                        min_pool = int(getattr(self, "capacity", 0) or 0)
                    if int(getattr(self, "size", 0) or 0) < min_pool:
                        r = float(base_obj)
                        r *= scale
                        return float(r) if np.isfinite(r) else 0.0
                    warm = int(getattr(self, "_delta_best_warmup_steps", 0) or 0)
                    if warm > 0:
                        step = int(getattr(self, "_current_step", 0) or 0)
                        if step < warm:
                            r = float(base_obj)
                            r *= scale
                            return float(r) if np.isfinite(r) else 0.0
                    min_beta = getattr(self, "_delta_best_min_lcb_beta", None)
                    if min_beta is not None:
                        try:
                            beta = float(getattr(self, "_lcb_beta"))
                        except Exception:
                            beta = 0.0
                        if beta < float(min_beta):
                            r = float(base_obj)
                            r *= scale
                            return float(r) if np.isfinite(r) else 0.0
                    new_best_obj = float(getattr(self, "best_obj", old_best_obj))
                    d = new_best_obj - float(old_best_obj)
                    if not np.isfinite(d) or d <= 0:
                        return 0.0
                    r = d
                else:
                    r = float(base_obj)
                r *= scale
                if not np.isfinite(r):
                    return 0.0
                return float(r)

            @staticmethod
            def _safe_abs_min_idx(x: np.ndarray) -> int:
                if x.size <= 0:
                    return 0
                return int(np.argmin(np.abs(x)))

            def _try_new_expr_lazy(self, expr):
                """
                Lazy 模式：不是每次入池都做 optimize（MeanStd 的 optimize 很重，会导致后期 fps 崩）。
                - 入池/互相关筛选仍然严格；
                - 超容量时用 |single IC| 的启发式做一次快速剔除；
                - 仅当累计 N 次“成功入池/替换”后才做一次 optimize 来更新 weights。
                """
                ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=0.99)
                if ic_ret is None or ic_mut is None or np.isnan(ic_ret) or np.isnan(ic_mut).any():
                    return 0.0
                if str(expr) in self._failure_cache:
                    return self.best_obj

                self.eval_cnt += 1
                old_pool = self.exprs[: self.size]  # type: ignore
                self._add_factor(expr, ic_ret, ic_mut)

                worst_idx = None
                if self.size > self.capacity:
                    remove_by = str(getattr(self, "_lazy_remove_by", "weight"))
                    if remove_by == "single_ic":
                        worst_idx = self._safe_abs_min_idx(self.single_ics[: self.size])
                    else:
                        # 更接近 LinearAlphaPool：按 |weight| 最小淘汰
                        try:
                            w = np.asarray(self.weights, dtype=np.float64)
                            worst_idx = self._safe_abs_min_idx(w[: self.size])
                        except Exception:
                            worst_idx = self._safe_abs_min_idx(self.single_ics[: self.size])
                    if worst_idx == self.capacity:
                        self._pop(worst_idx)
                        self._failure_cache.add(str(expr))
                        return self.best_obj
                    self._pop(worst_idx)

                removed_idx = [worst_idx] if worst_idx is not None else []

                # optimize 频率控制（优先保证：pool 第一次填满时至少 optimize 一次）
                do_opt = False
                if self.size > 1:
                    if (not self._did_first_full_optimize) and (self.size >= self.capacity):
                        do_opt = True
                        self._did_first_full_optimize = True
                    else:
                        self._lazy_updates_since_opt += 1
                        every = int(getattr(self, "_optimize_every_updates", 1) or 1)
                        every = max(1, every)
                        do_opt = (every == 1) or (self._lazy_updates_since_opt >= every)

                if do_opt:
                    self.weights = self.optimize()
                    self._lazy_updates_since_opt = 0

                self.update_history.append(
                    AddRemoveAlphas(
                        added_exprs=[expr],
                        removed_idx=removed_idx,
                        old_pool=old_pool,
                        old_pool_ic=self.best_ic_ret,
                        new_pool_ic=ic_ret,
                    )
                )

                self._failure_cache = set()
                new_ic_ret, new_obj = self.calculate_ic_and_objective()
                self._maybe_update_best(new_ic_ret, new_obj)
                return new_obj

            def try_new_expr(self, expr):  # type: ignore[override]
                old_best_obj = float(getattr(self, "best_obj", -1.0))
                # Val Gate：pool 满时先在 val 子集估计 single-IC，不达标直接跳过
                vg_calc = getattr(self, "_val_gate_calc", None)
                if vg_calc is not None:
                    only_full = bool(getattr(self, "_val_gate_only_full", True))
                    if (not only_full) or (getattr(self, "size", 0) >= getattr(self, "capacity", 0)):
                        import time as _time
                        t0g = _time.perf_counter()
                        try:
                            ic_val_fast = float(vg_calc.calc_single_IC_ret(expr))
                        except Exception:
                            ic_val_fast = float("nan")
                        dtg = _time.perf_counter() - t0g
                        if getattr(self, "_perf_enabled", False):
                            self._perf["val_gate_calls"] += 1
                            self._perf["val_gate_time_s"] += float(dtg)
                        self._last_val_gate_ic = float(ic_val_fast)
                        thr = float(getattr(self, "_val_gate_min_abs_ic", 0.0))
                        if not (np.isfinite(ic_val_fast) and (abs(ic_val_fast) >= thr)):
                            if getattr(self, "_perf_enabled", False):
                                self._perf["val_gate_skips"] += 1
                            tl = getattr(self, "_trial_logger", None)
                            if tl is not None:
                                try:
                                    step = int(getattr(self, "_current_step", 0) or 0)
                                except Exception:
                                    step = 0
                                tl.log(
                                    {
                                        "step": step,
                                        "expr": str(expr),
                                        "gate": "val_gate",
                                        "val_ic_fast": float(ic_val_fast) if np.isfinite(ic_val_fast) else None,
                                        "val_gate_thr": float(thr),
                                        "pool_size": int(getattr(self, "size", 0) or 0),
                                    }
                                )
                            return 0.0

                # Fast Gate：pool 满时先用小样本估计 single-IC，不达标直接跳过完整评估
                fg_calc = getattr(self, "_fast_gate_calc", None)
                if fg_calc is not None:
                    only_full = bool(getattr(self, "_fast_gate_only_full", True))
                    if (not only_full) or (getattr(self, "size", 0) >= getattr(self, "capacity", 0)):
                        import time as _time
                        t0g = _time.perf_counter()
                        try:
                            ic_fast = float(fg_calc.calc_single_IC_ret(expr))
                        except Exception:
                            ic_fast = float("nan")
                        dtg = _time.perf_counter() - t0g
                        if getattr(self, "_perf_enabled", False):
                            self._perf["fast_gate_calls"] += 1
                            self._perf["fast_gate_time_s"] += float(dtg)
                        thr = float(getattr(self, "_fast_gate_min_abs_ic", 0.0))
                        if not (np.isfinite(ic_fast) and (abs(ic_fast) >= thr)):
                            if getattr(self, "_perf_enabled", False):
                                self._perf["fast_gate_skips"] += 1
                            tl = getattr(self, "_trial_logger", None)
                            if tl is not None:
                                try:
                                    step = int(getattr(self, "_current_step", 0) or 0)
                                except Exception:
                                    step = 0
                                tl.log(
                                    {
                                        "step": step,
                                        "expr": str(expr),
                                        "gate": "fast_gate",
                                        "single_ic_fast": float(ic_fast) if np.isfinite(ic_fast) else None,
                                        "fast_gate_thr": float(thr),
                                        "val_ic_fast": float(getattr(self, "_last_val_gate_ic", float("nan")))
                                        if np.isfinite(getattr(self, "_last_val_gate_ic", float("nan")))
                                        else None,
                                        "pool_size": int(getattr(self, "size", 0) or 0),
                                    }
                                )
                            return 0.0

                # 默认（every_updates=1）：保持 LinearAlphaPool 原始行为（每次入池都 optimize）
                every = int(getattr(self, "_optimize_every_updates", 1) or 1)
                if every <= 1:
                    if not getattr(self, "_perf_enabled", False):
                        out = super().try_new_expr(expr)
                        return self._shape_reward(old_best_obj, out)
                    import time as _time
                    t0 = _time.perf_counter()
                    out = super().try_new_expr(expr)
                    dt = _time.perf_counter() - t0
                    self._perf["try_new_expr_calls"] += 1
                    self._perf["try_new_expr_time_s"] += float(dt)
                    return self._shape_reward(old_best_obj, out)

                # Lazy（every_updates>1）：减少 optimize 调用，避免后期 fps 崩
                if not getattr(self, "_perf_enabled", False):
                    out = self._try_new_expr_lazy(expr)
                    return self._shape_reward(old_best_obj, out)
                import time as _time
                t0 = _time.perf_counter()
                out = self._try_new_expr_lazy(expr)
                dt = _time.perf_counter() - t0
                self._perf["try_new_expr_calls"] += 1
                self._perf["try_new_expr_time_s"] += float(dt)
                return self._shape_reward(old_best_obj, out)

            def _calc_ics(self, expr, ic_mut_threshold=None):  # type: ignore[override]
                # 使用 env 配置的 mutual 阈值（覆盖 super().try_new_expr 传入的默认值 0.99）
                ic_mut_threshold = float(getattr(self, "_mutual_ic_threshold", MUTUAL_IC_THRESHOLD))
                if not getattr(self, "_perf_enabled", False):
                    try:
                        single_ic = float(self.calculator.calc_single_IC_ret(expr))
                    except Exception:
                        return float("nan"), None

                    # 单因子阈值：可选 abs 版本（适合可做空组合）
                    try:
                        under = bool(getattr(self, "_under_thres_alpha"))
                    except Exception:
                        under = False
                    lb = float(getattr(self, "_ic_lower_bound", -1.0))
                    if (not under) and (self.size > 0) and np.isfinite(lb):
                        v = abs(single_ic) if IC_LOWER_BOUND_ABS else single_ic
                        if v < lb:
                            self._last_single_ic = float(single_ic)
                            self._last_mutual_max = float("nan")
                            return float(single_ic), None

                    mutual_ics = []
                    for i in range(int(getattr(self, "size", 0) or 0)):
                        try:
                            mi = float(self.calculator.calc_mutual_IC(expr, self.exprs[i]))  # type: ignore[index]
                        except Exception:
                            mi = float("nan")
                        if ic_mut_threshold is not None and np.isfinite(mi) and (mi > float(ic_mut_threshold)):
                            self._last_single_ic = float(single_ic)
                            self._last_mutual_max = float(mi)
                            return float(single_ic), None
                        mutual_ics.append(mi)

                    out = (float(single_ic), mutual_ics)
                    try:
                        single_ic, ic_mut = out
                        self._last_single_ic = float(single_ic)
                        if ic_mut is None:
                            self._last_mutual_max = float("nan")
                        else:
                            if not ic_mut:
                                self._last_mutual_max = float("nan")
                            else:
                                arr = np.asarray(ic_mut, dtype=np.float64)
                                m = np.nanmax(arr) if np.isfinite(arr).any() else float("nan")
                                self._last_mutual_max = float(m)
                    except Exception:
                        pass
                    return out
                import time as _time
                t0 = _time.perf_counter()
                try:
                    single_ic = float(self.calculator.calc_single_IC_ret(expr))
                except Exception:
                    out = (float("nan"), None)
                else:
                    try:
                        under = bool(getattr(self, "_under_thres_alpha"))
                    except Exception:
                        under = False
                    lb = float(getattr(self, "_ic_lower_bound", -1.0))
                    if (not under) and (self.size > 0) and np.isfinite(lb):
                        v = abs(single_ic) if IC_LOWER_BOUND_ABS else single_ic
                        if v < lb:
                            out = (float(single_ic), None)
                        else:
                            out = None
                    else:
                        out = None

                if out is None:
                    mutual_ics = []
                    for i in range(int(getattr(self, "size", 0) or 0)):
                        try:
                            mi = float(self.calculator.calc_mutual_IC(expr, self.exprs[i]))  # type: ignore[index]
                        except Exception:
                            mi = float("nan")
                        if ic_mut_threshold is not None and np.isfinite(mi) and (mi > float(ic_mut_threshold)):
                            out = (float(single_ic), None)
                            break
                        mutual_ics.append(mi)
                    if out is None:
                        out = (float(single_ic), mutual_ics)
                dt = _time.perf_counter() - t0
                self._perf["calc_ics_calls"] += 1
                self._perf["calc_ics_time_s"] += float(dt)
                try:
                    single_ic, ic_mut = out
                    self._last_single_ic = float(single_ic)
                    if ic_mut is None:
                        self._last_mutual_max = float("nan")
                    else:
                        if not ic_mut:
                            self._last_mutual_max = float("nan")
                        else:
                            arr = np.asarray(ic_mut, dtype=np.float64)
                            m = np.nanmax(arr) if np.isfinite(arr).any() else float("nan")
                            self._last_mutual_max = float(m)
                except Exception:
                    pass
                return out

            def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:  # type: ignore[override]
                # MeanStdAlphaPool 默认 optimize(max_steps=10000) 在 days*stocks 很大时非常慢。
                # 这里复用 ALPHAGEN_POOL_OPT_* 作为统一的“优化预算阀门”，方便在脚本里提速/稳住。
                if not getattr(self, "_perf_enabled", False):
                    return super().optimize(lr=POOL_OPT_LR, max_steps=POOL_OPT_MAX_STEPS, tolerance=POOL_OPT_TOLERANCE)
                import time as _time
                t0 = _time.perf_counter()
                out = super().optimize(lr=POOL_OPT_LR, max_steps=POOL_OPT_MAX_STEPS, tolerance=POOL_OPT_TOLERANCE)
                dt = _time.perf_counter() - t0
                self._perf["optimize_calls"] += 1
                self._perf["optimize_time_s"] += float(dt)
                return out

        pool = TunableNaNFriendlyMeanStdAlphaPool(
            capacity=POOL_CAPACITY,
            calculator=calculator,  # type: ignore[arg-type]
            ic_lower_bound=init_ic_lb,
            l1_alpha=L1_ALPHA,
            lcb_beta=POOL_LCB_BETA_START if (POOL_LCB_BETA_START is not None) else POOL_LCB_BETA,
            device=device_obj,
        )
    else:
        perf_raw = os.environ.get("ALPHAGEN_PERF_LOG", "0").strip().lower()
        PERF_LOG = perf_raw in {"1", "true", "yes", "y", "on"}

        class TunableMseAlphaPool(MseAlphaPool):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._perf_enabled = bool(PERF_LOG)
                self._perf = {
                    "try_new_expr_calls": 0,
                    "try_new_expr_time_s": 0.0,
                    "calc_ics_calls": 0,
                    "calc_ics_time_s": 0.0,
                    "optimize_calls": 0,
                    "optimize_time_s": 0.0,
                    "fast_gate_calls": 0,
                    "fast_gate_skips": 0,
                    "fast_gate_time_s": 0.0,
                }
                self._optimize_every_updates = int(POOL_OPT_EVERY_UPDATES_START)
                self._lazy_updates_since_opt = 0
                self._did_first_full_optimize = False
                self._fast_gate_calc = fast_gate_calc
                self._fast_gate_only_full = bool(FAST_GATE_ONLY_WHEN_FULL)
                self._fast_gate_min_abs_ic = float(FAST_GATE_MIN_ABS_IC)
                self._reward_mode = str(REWARD_MODE)
                self._reward_scale = float(REWARD_SCALE)
                self._reward_prev_obj: Optional[float] = None
                self._delta_best_min_pool_size = int(DELTA_BEST_MIN_POOL_SIZE)
                self._delta_best_warmup_steps = int(DELTA_BEST_WARMUP_STEPS)
                self._delta_best_min_lcb_beta = DELTA_BEST_MIN_LCB_BETA
                self._lazy_remove_by = str(LAZY_REMOVE_BY)

            def perf_stats(self) -> dict:
                return dict(self._perf) if getattr(self, "_perf_enabled", False) else {}

            def _shape_reward(self, old_best_obj: float, out_obj: float) -> float:
                mode = str(getattr(self, "_reward_mode", "abs"))
                scale = float(getattr(self, "_reward_scale", 1.0) or 1.0)
                base_obj = float(out_obj) if np.isfinite(out_obj) else 0.0
                if mode == "delta_obj":
                    prev = getattr(self, "_reward_prev_obj", None)
                    if prev is None or (not np.isfinite(prev)):
                        setattr(self, "_reward_prev_obj", float(base_obj))
                        return 0.0
                    d = float(base_obj) - float(prev)
                    setattr(self, "_reward_prev_obj", float(base_obj))
                    r = d * scale
                    return float(r) if np.isfinite(r) else 0.0
                if mode == "delta_best":
                    min_pool = int(getattr(self, "_delta_best_min_pool_size", -1))
                    if min_pool < 0:
                        min_pool = int(getattr(self, "capacity", 0) or 0)
                    if int(getattr(self, "size", 0) or 0) < min_pool:
                        r = float(base_obj)
                        r *= scale
                        return float(r) if np.isfinite(r) else 0.0
                    warm = int(getattr(self, "_delta_best_warmup_steps", 0) or 0)
                    if warm > 0:
                        step = int(getattr(self, "_current_step", 0) or 0)
                        if step < warm:
                            r = float(base_obj)
                            r *= scale
                            return float(r) if np.isfinite(r) else 0.0
                    new_best_obj = float(getattr(self, "best_obj", old_best_obj))
                    d = new_best_obj - float(old_best_obj)
                    if not np.isfinite(d) or d <= 0:
                        return 0.0
                    r = d
                else:
                    r = float(base_obj)
                r *= scale
                if not np.isfinite(r):
                    return 0.0
                return float(r)

            @staticmethod
            def _safe_abs_min_idx(x: np.ndarray) -> int:
                if x.size <= 0:
                    return 0
                return int(np.argmin(np.abs(x)))

            def _try_new_expr_lazy(self, expr):
                ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=0.99)
                if ic_ret is None or ic_mut is None or np.isnan(ic_ret) or np.isnan(ic_mut).any():
                    return 0.0
                if str(expr) in self._failure_cache:
                    return self.best_obj

                self.eval_cnt += 1
                old_pool = self.exprs[: self.size]  # type: ignore
                self._add_factor(expr, ic_ret, ic_mut)

                worst_idx = None
                if self.size > self.capacity:
                    remove_by = str(getattr(self, "_lazy_remove_by", "weight"))
                    if remove_by == "single_ic":
                        worst_idx = self._safe_abs_min_idx(self.single_ics[: self.size])
                    else:
                        try:
                            w = np.asarray(self.weights, dtype=np.float64)
                            worst_idx = self._safe_abs_min_idx(w[: self.size])
                        except Exception:
                            worst_idx = self._safe_abs_min_idx(self.single_ics[: self.size])
                    if worst_idx == self.capacity:
                        self._pop(worst_idx)
                        self._failure_cache.add(str(expr))
                        return self.best_obj
                    self._pop(worst_idx)

                removed_idx = [worst_idx] if worst_idx is not None else []

                do_opt = False
                if self.size > 1:
                    if (not self._did_first_full_optimize) and (self.size >= self.capacity):
                        do_opt = True
                        self._did_first_full_optimize = True
                    else:
                        self._lazy_updates_since_opt += 1
                        every = int(getattr(self, "_optimize_every_updates", 1) or 1)
                        every = max(1, every)
                        do_opt = (every == 1) or (self._lazy_updates_since_opt >= every)

                if do_opt:
                    self.weights = self.optimize(lr=POOL_OPT_LR, max_steps=POOL_OPT_MAX_STEPS, tolerance=POOL_OPT_TOLERANCE)
                    self._lazy_updates_since_opt = 0

                self.update_history.append(
                    AddRemoveAlphas(
                        added_exprs=[expr],
                        removed_idx=removed_idx,
                        old_pool=old_pool,
                        old_pool_ic=self.best_ic_ret,
                        new_pool_ic=ic_ret,
                    )
                )

                self._failure_cache = set()
                new_ic_ret, new_obj = self.calculate_ic_and_objective()
                self._maybe_update_best(new_ic_ret, new_obj)
                return new_obj

            def try_new_expr(self, expr):  # type: ignore[override]
                old_best_obj = float(getattr(self, "best_obj", -1.0))
                expr_str = str(expr)
                step = int(getattr(self, "_current_step", 0) or 0)
                # 重置上一次 _calc_ics 记录（避免串台）
                self._last_single_ic = float("nan")
                self._last_mutual_max = float("nan")

                # 代理 Gate：在 pool 满后先用 surrogate 预测 abs(single_ic)，低于阈值则跳过完整评估
                sg = getattr(self, "_surrogate", None)
                if sg is not None:
                    only_full = bool(getattr(self, "_surrogate_only_full", True))
                    if (not only_full) or (getattr(self, "size", 0) >= getattr(self, "capacity", 0)):
                        import time as _time
                        t0s = _time.perf_counter()
                        try:
                            s = float(sg.score(expr_str))
                        except Exception:
                            s = float("nan")
                        dts = _time.perf_counter() - t0s
                        if getattr(self, "_perf_enabled", False):
                            self._perf["surrogate_calls"] += 1
                            self._perf["surrogate_time_s"] += float(dts)
                        thr = float(getattr(self, "_surrogate_thr", 0.0))
                        rand_p = float(getattr(self, "_surrogate_rand", 0.05))
                        # 若模型可用且预测明显低于阈值，并且未被随机放行，则跳过
                        if (np.isfinite(s) and (s < thr)) and (random.random() > rand_p):
                            if getattr(self, "_perf_enabled", False):
                                self._perf["surrogate_skips"] += 1
                            tl = getattr(self, "_trial_logger", None)
                            if tl is not None:
                                tl.log(
                                    {
                                        "step": step,
                                        "expr": expr_str,
                                        "gate": "surrogate",
                                        "surrogate_score": float(s),
                                        "surrogate_thr": float(thr),
                                        "surrogate_rand": float(rand_p),
                                    }
                                )
                            return 0.0

                fg_calc = getattr(self, "_fast_gate_calc", None)
                if fg_calc is not None:
                    only_full = bool(getattr(self, "_fast_gate_only_full", True))
                    if (not only_full) or (getattr(self, "size", 0) >= getattr(self, "capacity", 0)):
                        import time as _time
                        t0g = _time.perf_counter()
                        try:
                            ic_fast = float(fg_calc.calc_single_IC_ret(expr))
                        except Exception:
                            ic_fast = float("nan")
                        dtg = _time.perf_counter() - t0g
                        if getattr(self, "_perf_enabled", False):
                            self._perf["fast_gate_calls"] += 1
                            self._perf["fast_gate_time_s"] += float(dtg)
                        thr = float(getattr(self, "_fast_gate_min_abs_ic", 0.0))
                        if not (np.isfinite(ic_fast) and (abs(ic_fast) >= thr)):
                            if getattr(self, "_perf_enabled", False):
                                self._perf["fast_gate_skips"] += 1
                            tl = getattr(self, "_trial_logger", None)
                            if tl is not None:
                                tl.log(
                                    {
                                        "step": step,
                                        "expr": expr_str,
                                        "gate": "fast_gate",
                                        "ic_fast": float(ic_fast) if np.isfinite(ic_fast) else None,
                                        "fast_gate_thr": float(thr),
                                    }
                                )
                            return 0.0

                pre_hist = len(getattr(self, "update_history", []) or [])
                every = int(getattr(self, "_optimize_every_updates", 1) or 1)
                if every <= 1:
                    if not getattr(self, "_perf_enabled", False):
                        out = super().try_new_expr(expr)
                        post_hist = len(getattr(self, "update_history", []) or [])
                        tl = getattr(self, "_trial_logger", None)
                        if tl is not None:
                            tl.log(
                                {
                                    "step": step,
                                    "expr": expr_str,
                                    "gate": "none",
                                    "single_ic": float(self._last_single_ic) if np.isfinite(self._last_single_ic) else None,
                                    "mutual_max": float(self._last_mutual_max) if np.isfinite(self._last_mutual_max) else None,
                                    "accepted": bool(post_hist > pre_hist),
                                    "pool_size": int(getattr(self, "size", 0) or 0),
                                }
                            )
                        return self._shape_reward(old_best_obj, out)
                    import time as _time
                    t0 = _time.perf_counter()
                    out = super().try_new_expr(expr)
                    dt = _time.perf_counter() - t0
                    self._perf["try_new_expr_calls"] += 1
                    self._perf["try_new_expr_time_s"] += float(dt)
                    post_hist = len(getattr(self, "update_history", []) or [])
                    tl = getattr(self, "_trial_logger", None)
                    if tl is not None:
                        tl.log(
                            {
                                "step": step,
                                "expr": expr_str,
                                "gate": "none",
                                "single_ic": float(self._last_single_ic) if np.isfinite(self._last_single_ic) else None,
                                "mutual_max": float(self._last_mutual_max) if np.isfinite(self._last_mutual_max) else None,
                                "accepted": bool(post_hist > pre_hist),
                                "pool_size": int(getattr(self, "size", 0) or 0),
                            }
                        )
                    return self._shape_reward(old_best_obj, out)

                if not getattr(self, "_perf_enabled", False):
                    out = self._try_new_expr_lazy(expr)
                    post_hist = len(getattr(self, "update_history", []) or [])
                    tl = getattr(self, "_trial_logger", None)
                    if tl is not None:
                        tl.log(
                            {
                                "step": step,
                                "expr": expr_str,
                                "gate": "none",
                                "single_ic": float(self._last_single_ic) if np.isfinite(self._last_single_ic) else None,
                                "mutual_max": float(self._last_mutual_max) if np.isfinite(self._last_mutual_max) else None,
                                "accepted": bool(post_hist > pre_hist),
                                "pool_size": int(getattr(self, "size", 0) or 0),
                            }
                        )
                    return self._shape_reward(old_best_obj, out)
                import time as _time
                t0 = _time.perf_counter()
                out = self._try_new_expr_lazy(expr)
                dt = _time.perf_counter() - t0
                self._perf["try_new_expr_calls"] += 1
                self._perf["try_new_expr_time_s"] += float(dt)
                post_hist = len(getattr(self, "update_history", []) or [])
                tl = getattr(self, "_trial_logger", None)
                if tl is not None:
                    tl.log(
                        {
                            "step": step,
                            "expr": expr_str,
                            "gate": "none",
                            "single_ic": float(self._last_single_ic) if np.isfinite(self._last_single_ic) else None,
                            "mutual_max": float(self._last_mutual_max) if np.isfinite(self._last_mutual_max) else None,
                            "accepted": bool(post_hist > pre_hist),
                            "pool_size": int(getattr(self, "size", 0) or 0),
                        }
                    )
                return self._shape_reward(old_best_obj, out)

            def _calc_ics(self, expr, ic_mut_threshold=None):  # type: ignore[override]
                if not getattr(self, "_perf_enabled", False):
                    return super()._calc_ics(expr, ic_mut_threshold=ic_mut_threshold)
                import time as _time
                t0 = _time.perf_counter()
                out = super()._calc_ics(expr, ic_mut_threshold=ic_mut_threshold)
                dt = _time.perf_counter() - t0
                self._perf["calc_ics_calls"] += 1
                self._perf["calc_ics_time_s"] += float(dt)
                return out

            def optimize(self, lr: float = 5e-4, max_steps: int = 10000, tolerance: int = 500) -> np.ndarray:  # type: ignore[override]
                if not getattr(self, "_perf_enabled", False):
                    return super().optimize(lr=POOL_OPT_LR, max_steps=POOL_OPT_MAX_STEPS, tolerance=POOL_OPT_TOLERANCE)
                import time as _time
                t0 = _time.perf_counter()
                out = super().optimize(lr=POOL_OPT_LR, max_steps=POOL_OPT_MAX_STEPS, tolerance=POOL_OPT_TOLERANCE)
                dt = _time.perf_counter() - t0
                self._perf["optimize_calls"] += 1
                self._perf["optimize_time_s"] += float(dt)
                return out

        pool = TunableMseAlphaPool(
            capacity=POOL_CAPACITY,
            calculator=calculator,
            ic_lower_bound=init_ic_lb,
            l1_alpha=L1_ALPHA,
            device=device_obj,
        )

    # 可选：恢复历史 alpha_pool（重要：ALPHAGEN_RESUME 默认只恢复 PPO 模型，不会恢复 pool）
    # 用法：
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

            # force_load_exprs 会用 _calc_ics 重新构建 mutual IC 等信息；
            # 为了确保能完整复现历史 pool，不让 ic_lower_bound 影响加载，加载时临时关闭阈值。
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
                        # 更激进一点的阈值：
                        # - 当 stack_size >= remaining + 1 时，如果继续 push（特征/常量/dt），
                        #   很容易在“最后一个 token”后仍无法收栈到 1，从而以 -1 结束 episode。
                        # - 这能显著减少冷启动阶段的 -1/15 卡死（reward-sparsity）。
                        if bool(ret["select"][0]) and (stack_size >= (remaining + 1)):
                            ret["select"][1] = False  # Features / Sub-expressions
                            ret["select"][2] = False  # Constants
                            ret["select"][3] = False  # Delta time
                            # 同时尽量禁止 UnaryOperator（它不会减少栈深），强制优先使用“能收栈”的算子。
                            # 否则策略可能连续选 unary，导致剩余 token 不足以把栈收回到 1，最终仍以 -1 结束。
                            try:
                                has_reducing = bool(ret["op"].get(_core_mod.BinaryOperator, False)) or bool(
                                    ret["op"].get(_core_mod.RollingOperator, False)
                                ) or bool(ret["op"].get(_core_mod.PairRollingOperator, False))
                                if has_reducing:
                                    ret["op"][_core_mod.UnaryOperator] = False
                            except Exception:
                                pass
                    except Exception:
                        pass
                return ret

            _core_mod.AlphaEnvCore._valid_action_types = _valid_action_types_with_min_len  # type: ignore[assignment]
        except Exception as e:
            print(f"⚠ 设置 MIN_EXPR_LEN 失败（将忽略）：{e}")

    # ==================== 创建RL环境 ====================
    print("Setting up RL environment...")
    subexprs = None
    if RESUME_FLAG and resume_subexpr_strs:
        # resume 时优先使用历史 subexprs 快照，避免 action space 不一致
        try:
            from alphagen.data.parser import parse_expression

            parsed = []
            for s in resume_subexpr_strs:
                if not isinstance(s, str) or not s.strip():
                    continue
                try:
                    parsed.append(parse_expression(s.strip()))
                except Exception:
                    continue
            if parsed:
                subexprs = parsed
                print(f"✓ RESUME: 使用历史 subexprs 快照（n={len(subexprs)}）以保持空间一致")
        except Exception as e:
            print(f"⚠ RESUME: 解析 subexprs 快照失败，将回退到重建：{e}")

    if subexprs is None:
        subexprs = _build_subexpr_library(subexprs_max)
    if subexprs:
        try:
            out_dir = output_dir
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

    if RESUME_FLAG and resume_path.exists():
        print(f"Resuming PPO model from: {resume_path}")
        model = MaskablePPO.load(
            str(resume_path),
            env=env,
            device=DEVICE,
        )
        # 允许在恢复训练时微调部分超参（不改网络结构）
        model.ent_coef = ENT_COEF
        model.target_kl = TARGET_KL
        model.n_epochs = N_EPOCHS
        # learning_rate 在 SB3 里可能是 schedule，这里不强行覆盖，避免产生误解
    else:
        if RESUME_FLAG and not resume_path.exists():
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
    callbacks.append(Sb3LoggerMaxLengthCallback(max_length=SB3_LOGGER_MAX_LENGTH, verbose=0))
    callbacks.append(AlphaCacheStatsCallback(calculator_obj=calculator, update_every=2048))
    callbacks.append(PoolPerfStatsCallback(pool_obj=pool, update_every=2048))
    if POOL_PRUNE_BY_VAL and (val_gate_calc is not None):
        callbacks.append(
            PeriodicPoolValPruneCallback(
                pool=pool,
                val_calc=val_gate_calc,
                every_steps=POOL_PRUNE_EVERY_STEPS,
                keep_top_k=min(int(POOL_PRUNE_KEEP_TOP_K), int(getattr(pool, "capacity", POOL_PRUNE_KEEP_TOP_K) or POOL_PRUNE_KEEP_TOP_K)),
                min_abs_ic=float(POOL_PRUNE_MIN_ABS_IC),
                only_when_full=bool(POOL_PRUNE_ONLY_WHEN_FULL),
                verbose=0,
            )
        )
    ckpt_cb = None
    if ckpt_every_steps > 0:
        ckpt_cb = PeriodicCheckpointCallback(
            pool=pool,
            ckpt_dir=ckpt_dir,
            every_steps=ckpt_every_steps,
            feature_cols=list(feature_space.feature_cols),
            subexpr_strs=[str(e) for e in (subexprs or [])],
            keep_last=ckpt_keep_last,
            verbose=0,
        )
        callbacks.append(ckpt_cb)
    # pool.optimize 频率 schedule（默认关闭：start=end=1 时等价于原行为）
    if POOL_OPT_EVERY_UPDATES_START != POOL_OPT_EVERY_UPDATES_END:
        callbacks.append(
            PoolOptimizeEveryScheduleCallback(
                pool=pool,
                total_timesteps=TOTAL_TIMESTEPS,
                start_every=POOL_OPT_EVERY_UPDATES_START,
                end_every=POOL_OPT_EVERY_UPDATES_END,
                update_every=POOL_OPT_EVERY_UPDATES_UPDATE_EVERY,
                verbose=0,
            )
        )
    else:
        try:
            setattr(pool, "_optimize_every_updates", int(POOL_OPT_EVERY_UPDATES_START))
        except Exception:
            pass
    # 动态最小长度（可选）：帮助解决“后期越跑越慢（评估过频）”以及“冷启动全 -1（表达式无效）”
    if (MIN_EXPR_LEN_START != MIN_EXPR_LEN_END) or (MIN_EXPR_LEN_START > 1):
        callbacks.append(
            MinExprLenScheduleCallback(
                total_timesteps=TOTAL_TIMESTEPS,
                start_len=MIN_EXPR_LEN_START,
                end_len=MIN_EXPR_LEN_END,
                update_every=MIN_EXPR_LEN_UPDATE_EVERY,
                holder=min_expr_len_holder,
                schedule_steps=MIN_EXPR_LEN_SCHEDULE_STEPS,
                warmup_steps=MIN_EXPR_LEN_WARMUP_STEPS,
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
                print_on_test_error=bool(eval_test_print_err),
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
                    schedule_steps=int(pool_lcb_beta_schedule_steps),
                    warmup_steps=int(pool_lcb_beta_warmup_steps),
                    verbose=0,
                )
            )
    # mutual IC 阈值课程：用来增强后期多样性，缓解平台期
    if float(mutual_thr_start) != float(mutual_thr_end):
        callbacks.append(
            MutualIcThresholdScheduleCallback(
                pool=pool,
                start_thr=float(mutual_thr_start),
                end_thr=float(mutual_thr_end),
                update_every=int(mutual_thr_update_every),
                schedule_steps=int(mutual_thr_schedule_steps),
                warmup_steps=int(mutual_thr_warmup_steps),
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
                schedule_steps=int(ic_lb_schedule_steps),
                warmup_steps=int(ic_lb_warmup_steps),
            )
        )
    callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback
    )

    # ==================== 保存结果 ====================
    print("\n" + "=" * 60)
    print("Training complete! Saving results...")
    print("=" * 60)

    # 采集落盘（不影响主流程）
    try:
        if trial_logger is not None:
            trial_logger.flush()
    except Exception:
        pass

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
