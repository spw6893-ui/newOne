# AlphaQCM 加密货币改造完成总结

## 已完成的工作

### 1. 核心文件创建

| 文件 | 说明 | 状态 |
|------|------|------|
| `data_collection/fetch_crypto_data.py` | 使用 CCXT 获取加密货币数据 | ✓ 完成 |
| `alphagen_qlib/crypto_data.py` | CryptoData 类，替代 StockData | ✓ 完成 |
| `train_qcm_crypto.py` | 加密货币训练主脚本 | ✓ 完成 |
| `run_crypto_experiment.sh` | 快速启动脚本 | ✓ 完成 |
| `test_crypto_setup.py` | 环境测试脚本 | ✓ 完成 |
| `CRYPTO_README.md` | 完整使用文档 | ✓ 完成 |

### 2. 主要改动点

**数据层改造：**
- ✓ 从 Qlib (股票) 改为 CCXT (加密货币)
- ✓ 支持 7×24 连续交易数据
- ✓ 支持多时间周期 (1m, 5m, 15m, 1h, 4h, 1d)
- ✓ 支持自定义币种组合 (top10, top20, all)

**兼容性保持：**
- ✓ 保持与原 AlphaQCM 相同的接口
- ✓ 无需修改 RL 算法代码
- ✓ 无需修改因子表达式系统
- ✓ 无需修改配置文件

## 使用流程

### 第一步：安装依赖

```bash
# 安装 CCXT
pip install ccxt

# 验证环境
python test_crypto_setup.py
```

### 第二步：下载数据

```bash
# 下载 Top 20 加密货币数据 (2020-2024)
python data_collection/fetch_crypto_data.py
```

**自定义数据下载：**
编辑 `fetch_crypto_data.py` 修改：
- `symbols`: 交易对列表
- `timeframe`: '1h', '4h', '1d' 等
- `start_date` / `end_date`: 日期范围
- `exchange_name`: 'binance', 'okx', 'bybit' 等

### 第三步：训练模型

```bash
# 方式 1: 使用快速启动脚本
./run_crypto_experiment.sh qrdqn top10 1h 20

# 方式 2: 直接运行 Python
python train_qcm_crypto.py \
    --model qrdqn \
    --symbols top10 \
    --timeframe 1h \
    --pool 20 \
    --std-lam 1.0
```

### 第四步：查看结果

```bash
# 结果保存在
ls AlphaQCM_data/crypto_logs/top10_1h/pool_20_QCM_1.0/

# 查看发现的因子
cat AlphaQCM_data/crypto_logs/.../alpha_pool.csv
```

## 参数配置

### 训练参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--model` | 模型类型 | qrdqn (快) / iqn (慢但好) |
| `--symbols` | 币种组 | top10 (快) / top20 / all (慢) |
| `--timeframe` | K线周期 | 1h (平衡) / 4h (快) / 15m (慢) |
| `--pool` | Alpha池容量 | 20 (快) / 50 / 100 (慢) |
| `--target-periods` | 预测周期数 | 20 (1h周期) / 5 (15m周期) |
| `--std-lam` | 风险权重 | 1.0 (平衡) / 0.5 (激进) / 2.0 (保守) |

### 训练时间估算 (RTX 3070 8GB)

| 配置 | 预计时间 |
|------|----------|
| top10 + 1h + pool=20 | 4-8 小时 |
| top20 + 1h + pool=50 | 8-16 小时 |
| top10 + 15m + pool=20 | 12-24 小时 |
| all + 4h + pool=100 | 1-2 天 |

## 与股票版本的对比

| 特性 | 股票版本 | 加密货币版本 |
|------|----------|--------------|
| 数据源 | Qlib + baostock | CCXT |
| 市场 | 中国A股 | 全球加密货币 |
| 交易时间 | 工作日 9:30-15:00 | 7×24 连续 |
| 时间周期 | 日线 | 1m - 1d 可选 |
| 波动性 | 低 (±10%) | 高 (±50%) |
| 数据量 | CSI300: 300只 | Top20: 20个币 |
| 训练时间 | 6-12 小时 | 4-8 小时 (top10) |

## 输出结果说明

训练完成后，在 `AlphaQCM_data/crypto_logs/` 目录下会生成：

```
crypto_logs/
└── top10_1h/
    └── pool_20_QCM_1.0/
        └── qrdqn-seed0-20260215-1234-.../
            ├── alpha_pool.csv          # 发现的因子池
            ├── tensorboard/            # 训练曲线
            │   └── events.out.tfevents...
            └── checkpoints/            # 模型检查点
                ├── model_100000.pth
                └── model_200000.pth
```

**alpha_pool.csv 包含：**
- `expression`: 因子表达式 (如 `($close - Ref($close, 5)) / $volume`)
- `ic`: 信息系数 (越高越好，>0.05 为优秀)
- `rank_ic`: 排序信息系数
- `return`: 回测收益率
- `sharpe`: 夏普比率

## 常见问题

### Q1: 数据下载失败
```bash
# 检查网络
ping api.binance.com

# 尝试其他交易所
# 编辑 fetch_crypto_data.py，修改 exchange_name='okx'
```

### Q2: 显存不足 (OOM)
```bash
# 方案 1: 减少 batch_size
# 编辑 qcm_config/qrdqn.yaml
batch_size: 64  # 从 128 改为 64

# 方案 2: 减少币种
python train_qcm_crypto.py --symbols top10  # 而非 top20

# 方案 3: 使用 CPU (慢)
# 编辑 train_qcm_crypto.py
device = torch.device('cpu')
```

### Q3: 训练太慢
```bash
# 方案 1: 使用更长周期
python train_qcm_crypto.py --timeframe 4h  # 而非 1h

# 方案 2: 减少训练步数
# 编辑 qcm_config/qrdqn.yaml
num_steps: 1_000_000  # 从 2_000_000 改为 1_000_000

# 方案 3: 减少 pool size
python train_qcm_crypto.py --pool 10  # 而非 20
```

### Q4: 因子效果不好
```bash
# 尝试不同的预测周期
python train_qcm_crypto.py --target-periods 10  # 短期
python train_qcm_crypto.py --target-periods 50  # 长期

# 尝试不同的风险偏好
python train_qcm_crypto.py --std-lam 0.5  # 激进
python train_qcm_crypto.py --std-lam 2.0  # 保守

# 增加训练数据
# 编辑 fetch_crypto_data.py，修改 start_date='2018-01-01'
```

## 下一步优化方向

### 1. 添加加密货币特有特征
```python
# 在 crypto_data.py 中添加
class FeatureType(IntEnum):
    # ... 现有特征
    FUNDING_RATE = 6      # 资金费率
    OPEN_INTEREST = 7     # 持仓量
    LIQUIDATIONS = 8      # 爆仓量
    TAKER_BUY_RATIO = 9   # 主动买入比例
```

### 2. 多时间周期融合
同时使用 1h + 4h 数据，捕捉不同时间尺度的模式

### 3. 动态币种池
根据市值、流动性、波动率动态调整交易币种

### 4. 风险管理增强
- 添加最大回撤约束
- 添加夏普比率优化
- 添加仓位管理

### 5. 实盘对接
连接交易所 API，实现自动化交易

## 技术支持

如遇问题，请检查：
1. 运行 `python test_crypto_setup.py` 验证环境
2. 查看 `CRYPTO_README.md` 详细文档
3. 检查日志文件 `AlphaQCM_data/crypto_logs/.../log.txt`

## 文件清单

```
AlphaQCM/
├── data_collection/
│   ├── fetch_crypto_data.py          # 新增：加密货币数据下载
│   └── fetch_baostock_data.py        # 原有：股票数据下载
├── alphagen_qlib/
│   ├── crypto_data.py                # 新增：CryptoData 类
│   ├── stock_data.py                 # 原有：StockData 类
│   └── calculator.py                 # 共用：因子计算器
├── train_qcm_crypto.py               # 新增：加密货币训练脚本
├── train_qcm_csi300.py               # 原有：股票训练脚本
├── run_crypto_experiment.sh          # 新增：快速启动脚本
├── test_crypto_setup.py              # 新增：环境测试脚本
├── CRYPTO_README.md                  # 新增：使用文档
└── README.md                         # 原有：原始文档
```

## 总结

✓ 已完成从股票到加密货币的完整改造
✓ 保持了原有 AlphaQCM 的核心算法
✓ 支持灵活的币种、周期、参数配置
✓ 提供完整的测试和文档

**立即开始：**
```bash
python test_crypto_setup.py
./run_crypto_experiment.sh qrdqn top10 1h 20
```
