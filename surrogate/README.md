# Surrogate（代理模型）引导的表达式搜索

目标：在“表达式 → IC”评估非常昂贵且奖励稀疏的场景下，用一个轻量的代理模型先做**候选重排/筛选**，减少无效评估，并把探索更多集中在“更可能有 IC 的表达式结构”上。

这套方案不会替代现有的 AlphaGen/PPO 训练流程，而是给训练过程增加两类能力：

1) **采集数据**：把每次 `pool.try_new_expr()` 的尝试（表达式、single-IC、是否被 FastGate/代理模型跳过、是否入池等）落盘成 `jsonl`。
2) **训练代理模型**：从 `jsonl` 训练一个轻量线性回归（纯 numpy、无额外依赖），预测 `abs(single_ic)`。
3) **训练时启用代理 Gate（可选）**：当 pool 满后，先用代理模型估计 `abs(single_ic)`，低于阈值则跳过完整评估；同时保留一定随机通过比例，避免过早“锁死”探索空间。

---

## 1. 采集训练数据（JSONL）

训练时打开：

```bash
ALPHAGEN_TRIAL_LOG=1 \
ALPHAGEN_TRIAL_LOG_PATH=alphagen_output/expr_trials.jsonl \
./run_training.sh explore20_ucblcb_cs
```

生成文件：

- `alphagen_output/expr_trials.jsonl`：每行一个 JSON（便于增量写入）

---

## 2. 训练代理模型（预测 abs(single_ic)）

```bash
python3 surrogate/train_surrogate.py \
  --input alphagen_output/expr_trials.jsonl \
  --output alphagen_output/surrogate_model.npz
```

输出：

- `alphagen_output/surrogate_model.npz`

---

## 3. 训练时启用代理 Gate（可选）

只建议在：
- pool 已接近/达到容量；
- 已积累足够多的 trial（至少几千条）；
时再启用。

```bash
ALPHAGEN_SURROGATE_GATE=1 \
ALPHAGEN_SURROGATE_MODEL_PATH=alphagen_output/surrogate_model.npz \
ALPHAGEN_SURROGATE_SCORE_THRESHOLD=0.004 \
ALPHAGEN_SURROGATE_RANDOM_ACCEPT_PROB=0.05 \
./run_training.sh explore20_ucblcb_cs
```

说明：
- `SCORE_THRESHOLD`：代理模型预测的 `abs(single_ic)` 阈值（越大越严格）
- `RANDOM_ACCEPT_PROB`：低分表达式仍随机放行的概率（避免探索崩掉）

---

## 4. 如何判断代理 Gate 是否“在帮忙”

看 TensorBoard 标量：
- `perf/surrogate_calls`、`perf/surrogate_skips`、`perf/surrogate_ms_per_call`
- `pool/eval_cnt`：应该更集中在“更有希望”的表达式上，而不是大量无效尝试
- `eval/val_ic`：更重要的是上限是否上移，而不是短期抖动

