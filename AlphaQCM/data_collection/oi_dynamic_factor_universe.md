# OI 动态因子池（Dynamic Factor Universe）实践说明

你当前遇到的问题本质是：**OI（来自 Binance Vision metrics）在较早历史中不可用**，如果做固定因子组合/固定权重打分，会导致：
- 早期 OI=NaN（或被填 0）引入偏差
- 不同时期“总权重”不一致，最终分数不可比

这里给出“最专业做法”的 OI 适配：**动态因子池 + 权重重归一化**。

---

## 1) 思路（动态因子池）

设你有一组因子（其中 OI 因子可能缺失）：

- A：动量
- B：波动率/流动性
- C：订单流
- D：OI（例如 `oi_delta_usd_over_quote_volume`）

权重为：

- wA, wB, wC, wD

当某个时刻 D 不可用（NaN）时，不做填充，而是让模型/分数只使用可用因子集合 `{A,B,C}`，并做权重归一化：

```
score(t) = (wA*A + wB*B + wC*C) / (|wA| + |wB| + |wC|)
```

当 D 可用时：

```
score(t) = (wA*A + wB*B + wC*C + wD*D) / (|wA| + |wB| + |wC| + |wD|)
```

这样保证：
- 不同时期的 score 仍在同一尺度（分母按可用因子自动变化）
- 不会因为“OI 在早期缺失”而污染其它因子信号

> 说明：分母用 `sum(|w|)` 的好处是允许权重有正有负（长短因子混合），尺度依然稳定。

---

## 2) 工程落地（仓库内脚本）

脚本：
- `AlphaQCM/data_collection/dynamic_factor_universe.py`

它会对每个 `*_final.csv`：
- 读取你指定的因子列
- 对每行自动判断哪些列是 NaN
- 在“可用因子集合”上做权重重归一化并输出 `score_dyn`

### 2.1 权重文件格式（JSON）

示例：`weights_oi.json`

```json
{
  "seg_tail_minus_head": 0.4,
  "rv_std_sqrt60": -0.3,
  "at_cvd_quote": 0.2,
  "oi_delta_usd_over_quote_volume": 0.1
}
```

注意：
- 权重 key 必须是 `final_dataset` 里真实存在的列名
- OI 相关列在早期可能为 NaN，脚本会自动跳过并归一化其它权重

### 2.2 运行命令

```
python3 AlphaQCM/data_collection/dynamic_factor_universe.py \\
  --input-dir AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85 \\
  --output-dir AlphaQCM/AlphaQCM_data/final_dataset_vision_metrics85_dynscore \\
  --weights weights_oi.json \\
  --normalize sum_abs
```

输出文件名默认是：
- `{SYMBOL}_final_dyn.csv`

并新增列：
- `score_dyn`：动态归一化后的分数
- `score_dyn_wsum`：当时刻可用因子的 `sum(|w|)`
- `score_dyn_n`：当时刻参与打分的因子数量

---

## 3) 关于你说的“对 OI 做这个处理”

落地建议（最实用的版本）：

1) 先选一个“真正可比”的 OI 因子作为 D（建议用归一化后的变化类，而不是绝对 OI）：
   - `oi_delta_usd_over_quote_volume`（推荐）
   - 或 `oi_delta_over_volume`

2) 在 OI 不可用的时期，让 score 自动只由其它因子决定（动态因子池）。

3) 如果你要做“训练”，建议训练集起点设到 metrics 起始之后（例如 `2021-12-01`），或者额外加入 `oi_available` 掩码列；
   但如果你只是做多因子打分/组合，动态因子池是最干净的做法。

