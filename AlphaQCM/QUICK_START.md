# AlphaQCM 加密货币版 - 快速开始指南

## 已修复的关键问题 ✅

1. **VWAP 计算** - 从累积改为滚动窗口
2. **FeatureType 冲突** - 统一使用 stock_data.FeatureType
3. **依赖检查** - 添加 CCXT 安装验证
4. **数据对齐** - 从 intersection 改为 union + coverage filter
5. **内存优化** - float32 + CPU 先加载
6. **时区处理** - 统一使用 UTC
7. **Top100 支持** - 支持 100 个币种，2018-2025 数据

## 立即开始

### 步骤 1: 验证环境

```bash
cd /home/ppw/CryptoQuant/AlphaQCM
python3 validate_fixes.py
```

### 步骤 2: 安装依赖

```bash
pip install ccxt
```

### 步骤 3: 下载 Top100 数据

```bash
# 下载 Top100 加密货币数据 (2018-2025, 1小时周期)
python3 data_collection/fetch_crypto_data.py
```

**预计时间**: 2-4 小时（取决于网络速度）

**数据量**: 约 5-10GB

### 步骤 4: 训练模型

```bash
# Top100 训练
python3 train_qcm_crypto.py \
    --model qrdqn \
    --symbols top100 \
    --timeframe 1h \
    --pool 50 \
    --std-lam 1.0 \
    --seed 0
```

**预计训练时间** (RTX 3070 8GB):
- Top10: 4-8 小时
- Top20: 8-16 小时
- Top100: 1-3 天

## 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--model` | 模型类型 | qrdqn (快速) / iqn (效果好) |
| `--symbols` | 币种组 | top10 / top20 / top100 |
| `--timeframe` | K线周期 | 1h (平衡) / 4h (快) |
| `--pool` | Alpha池大小 | 20-50 |
| `--target-periods` | 预测周期 | 20 (1h) / 5 (15m) |
| `--std-lam` | 风险权重 | 1.0 (平衡) / 0.5 (激进) |

## 数据对齐策略

针对 Top100 从 2018 年开始的数据，处理新旧币种上市时间不同：

```
策略: Union + Coverage Filter + Forward Fill

1. 收集所有日期 (union)
2. 保留至少 50% 币种有数据的日期
3. Forward fill 缺失值

示例:
- BTC/ETH: 2018 年开始
- SOL: 2020 年上市 → 2018-2020 forward fill
- SUI: 2023 年上市 → 2018-2023 forward fill
```

## 内存优化

- **float32**: 节省 50% 内存
- **CPU 先加载**: 避免 CUDA OOM
- **Top100 + 1h + 7年**: 约 2-3GB 内存

## 常见问题

### Q: 数据下载失败
```bash
# 检查网络
ping api.binance.com

# 或编辑 fetch_crypto_data.py 换交易所
exchange_name='okx'  # 或 'bybit'
```

### Q: 显存不足
```bash
# 编辑 qcm_config/qrdqn.yaml
batch_size: 64  # 从 128 改为 64
```

### Q: 训练太慢
```bash
# 使用更长周期
python3 train_qcm_crypto.py --timeframe 4h

# 或减少币种
python3 train_qcm_crypto.py --symbols top20
```

## 输出结果

```
AlphaQCM_data/crypto_logs/
└── top100_1h/
    └── pool_50_QCM_1.0/
        └── qrdqn-seed0-.../
            ├── alpha_pool.csv      # 发现的因子
            ├── tensorboard/        # 训练曲线
            └── checkpoints/        # 模型检查点
```

**alpha_pool.csv 包含**:
- `expression`: 因子表达式
- `ic`: 信息系数 (>0.05 优秀)
- `rank_ic`: 排序信息系数
- `return`: 回测收益

## 文件清单

```
AlphaQCM/
├── data_collection/
│   ├── fetch_crypto_data.py          ✅ 已修复
│   └── top100_symbols.txt            ✅ 新增
├── alphagen_qlib/
│   └── crypto_data.py                ✅ 已修复
├── train_qcm_crypto.py               ✅ 已修复
├── validate_fixes.py                 ✅ 新增
├── BUG_FIXES.md                      ✅ 修复文档
└── QUICK_START.md                    ✅ 本文件
```

## 验证修复

```bash
python3 validate_fixes.py
```

应该看到:
```
✓ PASS: Import compatibility
✓ PASS: VWAP calculation
✓ PASS: CCXT dependency check
✓ PASS: Data alignment strategy
✓ PASS: Memory optimizations
✓ PASS: Top100 support
✓ PASS: Timezone handling
```

## 下一步

1. **小规模测试**: 先用 top10 验证
2. **逐步扩展**: top10 → top20 → top100
3. **监控训练**: 使用 tensorboard 查看曲线
4. **分析因子**: 查看 alpha_pool.csv 中的因子

## 技术支持

- 详细修复说明: `BUG_FIXES.md`
- 完整文档: `CRYPTO_README.md`
- 改造总结: `CRYPTO_MIGRATION_SUMMARY.md`

---

**准备好了吗？立即开始：**

```bash
python3 validate_fixes.py && \
python3 data_collection/fetch_crypto_data.py
```
