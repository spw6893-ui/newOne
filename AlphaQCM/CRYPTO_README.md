# AlphaQCM for Cryptocurrency

加密货币因子挖掘版本的 AlphaQCM 使用指南

## 快速开始

### 1. 安装依赖

```bash
# 安装 CCXT 用于获取加密货币数据
pip install ccxt

# 其他依赖已在 alphaqcm_env.yml 中
```

### 2. 下载加密货币数据

```bash
python data_collection/fetch_crypto_data.py
```

这将下载 Top 20 加密货币的小时级数据（2020-2024）到 `AlphaQCM_data/crypto_data/`

**自定义数据下载：**

编辑 `fetch_crypto_data.py` 修改：
- `symbols`: 交易对列表
- `timeframe`: 时间周期 ('1m', '5m', '15m', '1h', '4h', '1d')
- `start_date` / `end_date`: 日期范围
- `exchange_name`: 交易所 ('binance', 'okx', 'bybit', 等)

### 3. 训练模型

```bash
# 基础训练 - Top 10 加密货币，1小时周期
python train_qcm_crypto.py --model qrdqn --pool 20 --symbols top10 --timeframe 1h

# 更多币种
python train_qcm_crypto.py --model iqn --pool 50 --symbols top20 --timeframe 1h

# 所有可用币种
python train_qcm_crypto.py --model qrdqn --pool 100 --symbols all --timeframe 4h

# 短周期高频
python train_qcm_crypto.py --model iqn --pool 20 --symbols top10 --timeframe 15m
```

### 4. 参数说明

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `--model` | 模型类型 | qrdqn, iqn, fqf | qrdqn |
| `--pool` | Alpha池容量 | 10, 20, 50, 100 | 20 |
| `--symbols` | 币种组 | top10, top20, all | top10 |
| `--timeframe` | K线周期 | 1m, 5m, 15m, 1h, 4h, 1d | 1h |
| `--target-periods` | 预测周期数 | 任意整数 | 20 |
| `--std-lam` | 标准差权重 | 0.5, 1.0, 2.0 | 1.0 |
| `--seed` | 随机种子 | 任意整数 | 0 |

## 与股票版本的主要区别

### 1. 数据特点
- **7×24 交易**: 无休市，连续时间序列
- **高波动性**: 加密货币波动远大于股票
- **流动性差异**: 不同币种流动性差异巨大

### 2. 时间周期
- 股票: 日线数据，预测 20 天
- 加密货币: 可选 1m-1d，预测 N 个周期

### 3. 数据源
- 股票: Qlib + baostock (中国A股)
- 加密货币: CCXT (全球交易所)

### 4. 特征工程
当前支持基础特征：
- OPEN, CLOSE, HIGH, LOW, VOLUME, VWAP

**可扩展特征** (需修改 `crypto_data.py`):
- 资金费率 (Funding Rate)
- 持仓量 (Open Interest)
- 爆仓数据 (Liquidations)
- 主动买卖比 (Taker Buy Ratio)

## 训练时间估算

基于 1 张 RTX 3070 (8GB) 的粗略估算：

| 配置 | 预计时间 |
|------|----------|
| top10 + 1h + pool=20 | 4-8 小时 |
| top20 + 1h + pool=50 | 8-16 小时 |
| top10 + 15m + pool=20 | 12-24 小时 |
| all + 4h + pool=100 | 1-2 天 |

**影响因素：**
- 时间周期越短，数据点越多，训练越慢
- 币种数量直接影响因子计算量
- Pool size 越大，每次评估越慢

## 输出结果

训练结果保存在 `AlphaQCM_data/crypto_logs/` 目录：

```
crypto_logs/
├── top10_1h/
│   ├── pool_20_QCM_1.0/
│   │   ├── qrdqn-seed0-20260215-1234-N200-lr5e-05-perTrue-gamma1-step3/
│   │   │   ├── alpha_pool.csv          # 发现的 alpha 因子
│   │   │   ├── tensorboard/            # 训练曲线
│   │   │   └── checkpoints/            # 模型检查点
```

**alpha_pool.csv** 包含：
- 因子表达式
- IC (信息系数)
- Rank IC
- 回测收益等指标

## 使用发现的因子

```python
import pandas as pd

# 读取发现的因子
alphas = pd.read_csv('AlphaQCM_data/crypto_logs/.../alpha_pool.csv')

# 查看最佳因子
best_alphas = alphas.nlargest(10, 'ic')
print(best_alphas[['expression', 'ic', 'rank_ic']])

# 因子表达式示例：
# ($close - Ref($close, 5)) / $volume
# Abs($high - $low) / $close
# ...
```

## 常见问题

### Q1: 数据下载失败
- 检查网络连接
- 尝试更换交易所 (binance → okx)
- 减少并发请求（修改 fetch_crypto_data.py 添加延迟）

### Q2: 显存不足
- 减少 batch_size (修改 qcm_config/*.yaml)
- 减少币种数量 (使用 top10 而非 all)
- 使用更短的回溯周期

### Q3: 训练太慢
- 使用更长的时间周期 (4h 或 1d)
- 减少 num_steps (修改 qcm_config/*.yaml)
- 减少 pool size

### Q4: 因子效果不好
- 尝试不同的 target_periods
- 调整 std_lam 参数
- 增加训练数据时间范围
- 尝试不同的币种组合

## 下一步优化

1. **添加更多特征**: 资金费率、持仓量等链上数据
2. **多时间周期**: 同时使用 1h + 4h 数据
3. **动态币种池**: 根据市值/流动性动态调整
4. **风险控制**: 添加最大回撤、夏普比率等约束
5. **实盘对接**: 连接交易所 API 进行实盘交易

## 参考资料

- 原始论文: "AlphaQCM: Alpha Discovery in Finance with Distributional Reinforcement Learning"
- 股票版本: [AlphaGen](https://github.com/RL-MLDM/alphagen)
- CCXT 文档: https://docs.ccxt.com/
