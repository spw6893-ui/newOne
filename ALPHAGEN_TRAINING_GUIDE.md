# AlphaGen加密货币因子训练方案

## 概述

AlphaQCM已经包含完整的AlphaGen适配代码,可以直接用于加密货币因子挖掘。

## 架构说明

### 核心组件

1. **数据加载器**: `AlphaQCM/alphagen_qlib/crypto_data.py`
   - `CryptoData`: 替代StockData,加载加密货币数据
   - 支持多币种、多特征的3D tensor格式
   - 自动处理缺失数据和新币上市

2. **因子计算器**: `AlphaQCM/alphagen_qlib/calculator.py`
   - `QLibStockDataCalculator`: 计算IC/Rank IC
   - 支持单因子评估和因子池组合评估

3. **数据准备脚本**: `AlphaQCM/data_collection/prepare_alphagen_training_data.py`
   - 添加前向收益率标签(无前视偏差)
   - 质量过滤和缺失值处理
   - 时间分割(train/val/test)

## 训练流程

### Step 1: 准备训练数据

```bash
# 从过滤后的数据集生成AlphaGen训练数据
python AlphaQCM/data_collection/prepare_alphagen_training_data.py \
  --input-dir AlphaQCM_data/final_dataset \
  --output-dir AlphaQCM_data/alphagen_ready \
  --horizon-hours 1 \
  --filter-quality \
  --impute ffill \
  --ffill-limit 24 \
  --train-end "2024-01-01 00:00:00+00:00" \
  --val-end "2024-07-01 00:00:00+00:00"
```

**参数说明**:
- `--horizon-hours 1`: 预测1小时后的收益率
- `--filter-quality`: 过滤维护期/异常/新币等低质量样本
- `--impute ffill`: 前向填充缺失值(最多24小时)
- `--train-end/val-end`: 时间分割点

**输出**: `AlphaQCM_data/alphagen_ready/{SYMBOL}_train.csv`
- 包含特征 + `y_logret_fwd_1h` (目标)
- 包含 `split` 列(train/val/test)

### Step 2: 创建训练脚本

```python
# train_alphagen_crypto.py
import sys
sys.path.insert(0, 'alphagen')
sys.path.insert(0, 'AlphaQCM')

from alphagen_qlib.crypto_data import CryptoData, FeatureType
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.rl.env.core import AlphaEnvCore
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from sb3_contrib.ppo_mask import MaskablePPO
import torch

# 配置
SYMBOLS = 'top20'  # 或指定列表
START_TIME = '2020-01-01'
TRAIN_END = '2024-01-01'
VAL_END = '2024-07-01'
END_TIME = '2025-02-15'

# 定义特征
FEATURES = [
    FeatureType.OPEN, FeatureType.HIGH, FeatureType.LOW,
    FeatureType.CLOSE, FeatureType.VOLUME,
    # 添加你的自定义特征...
]

# 定义目标(预测1小时收益率)
target = Ref(Close(), -1) / Close() - 1  # 1小时前向收益率

# 加载训练数据
train_data = CryptoData(
    symbols=SYMBOLS,
    start_time=START_TIME,
    end_time=TRAIN_END,
    timeframe='1h',
    data_dir='AlphaQCM_data/alphagen_ready',
    features=FEATURES,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# 创建Calculator
calculator = QLibStockDataCalculator(train_data, target)

# 创建Alpha Pool
pool = AlphaPool(
    capacity=10,  # 保留top 10因子
    calculator=calculator,
    ic_lower_bound=0.01  # IC阈值
)

# 创建RL环境
env = AlphaEnv(AlphaEnvCore(calculator, pool))

# 创建PPO模型
model = MaskablePPO(
    'MlpPolicy',
    env,
    policy_kwargs=dict(
        features_extractor_class=LSTMSharedNet,
        features_extractor_kwargs=dict(n_layers=2, d_model=128)
    ),
    batch_size=128,
    device='cuda',
    verbose=1,
    tensorboard_log='./logs'
)

# 训练
model.learn(total_timesteps=100000)

# 保存
model.save('./output/model_final')
pool.save('./output/alpha_pool.json')
```

### Step 3: 运行训练

```bash
# 安装依赖
pip install -r alphagen/requirements.txt
pip install -r AlphaQCM/requirements_crypto.txt

# 训练
python train_alphagen_crypto.py
```

## 关键配置

### 1. 特征选择

从73个过滤后的特征中选择:
- **价格特征**: close, volume_clean
- **动量特征**: seg_head_logret, seg_tail_logret, returns_1h/24h/168h
- **波动率特征**: rv_l2, shape_skew, vol_regime
- **流动性特征**: liq_amihud, liq_spread_std
- **订单流特征**: at_cvd_qty, at_imbalance_ratio_quote

### 2. 目标定义

```python
# 1小时收益率
target = Ref(Close(), -1) / Close() - 1

# 24小时收益率
target = Ref(Close(), -24) / Close() - 1

# 排名收益(做多top 20%,做空bottom 20%)
target = Rank(Ref(Close(), -1) / Close() - 1)
```

### 3. AlphaGen表达式算子

可用算子(来自`alphagen/alphagen/data/expression.py`):
- **算术**: Add, Sub, Mul, Div
- **时序**: Ref(回看), Delta(差分), Ts_Mean(移动平均)
- **横截面**: Rank(排名), Normalize(标准化)
- **逻辑**: Greater, Less, If
- **数学**: Abs, Log, Sign, Power

### 4. 训练参数

```python
# Alpha Pool配置
pool = AlphaPool(
    capacity=10,           # 保留因子数量
    ic_lower_bound=0.01,   # IC阈值(低于此值的因子被淘汰)
    l1_alpha=5e-3          # L1正则化(鼓励稀疏组合)
)

# PPO配置
model = MaskablePPO(
    batch_size=128,        # 批大小
    learning_rate=3e-4,    # 学习率
    n_steps=2048,          # 每次更新的步数
    gamma=0.99,            # 折扣因子
    device='cuda'          # GPU加速
)
```

## 输出结果

### 1. 训练日志
- TensorBoard: `./logs/`
- 查看: `tensorboard --logdir=./logs`

### 2. 因子池
- 文件: `./output/alpha_pool.json`
- 格式:
```json
{
  "exprs": [
    "Rank(Ts_Mean(Close() / Ref(Close(), 1) - 1, 24))",
    "Normalize(Delta(Volume(), 1) / Volume())",
    ...
  ],
  "weights": [0.25, 0.18, ...],
  "ics": [0.045, 0.038, ...]
}
```

### 3. 模型检查点
- 文件: `./output/model_final.zip`
- 可用于继续训练或推理

## 评估和回测

### 验证集评估

```python
# 加载验证数据
val_data = CryptoData(
    symbols=SYMBOLS,
    start_time=TRAIN_END,
    end_time=VAL_END,
    ...
)

val_calc = QLibStockDataCalculator(val_data, target)

# 评估因子池
ic, ric = val_calc.calc_pool_all_ret(pool.exprs, pool.weights)
print(f"Validation IC: {ic:.4f}, Rank IC: {ric:.4f}")
```

### 回测

使用`alphagen/backtest.py`或`alphagen/trade_decision.py`进行回测。

## 硬件需求

根据之前的估算:
- **GPU**: RTX 3070 (8GB) 或更好
- **RAM**: 16GB+
- **存储**: 20GB+

## 预期结果

- **训练时间**: 2-6小时(取决于GPU)
- **因子数量**: 10个高质量因子
- **IC范围**: 0.02-0.05 (加密货币市场)
- **Rank IC**: 通常高于IC

## 下一步

1. 运行数据准备脚本
2. 创建训练脚本
3. 开始训练
4. 监控TensorBoard
5. 评估验证集
6. 回测和优化

需要我帮你执行哪一步?
