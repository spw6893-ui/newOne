# AlphaGen 模型架构改进方案

## 问题诊断

当前状态：从6个基础因子(OHLCV+VWAP)扩充到70+高频因子后，训练IC从合理水平降至0.04

### 根本原因

1. **特征维度爆炸但模型容量不足**
   - 70+因子输入空间巨大，但LSTM只有2层×128维
   - 模型无法有效学习高维特征组合

2. **缺乏特征选择机制**
   - 所有因子平等对待，噪声因子稀释有效信号
   - 高频因子中可能存在大量低IC/高噪声特征

3. **Alpha Pool容量限制**
   - 默认10个因子组合不足以覆盖70+维输入空间
   - IC阈值0.01过高，筛掉了潜在有价值的因子

4. **优化目标单一**
   - 只优化IC均值，忽略了因子稳定性(IC标准差)
   - 没有考虑因子多样性和相关性控制

---

## 改进方案

### 1. 模型架构增强

**修改位置**: `train_alphagen_crypto.py:277-300`

```python
# 原配置
policy_kwargs = dict(
    features_extractor_class=LSTMSharedNet,
    features_extractor_kwargs=dict(
        n_layers=2,      # 太浅
        d_model=128,     # 维度不足
        dropout=0.1,     # 正则化不足
        device=device_obj,
    )
)

# 改进配置
policy_kwargs = dict(
    features_extractor_class=LSTMSharedNet,
    features_extractor_kwargs=dict(
        n_layers=3,      # 2→3层，增强表达能力
        d_model=256,     # 128→256维，匹配高维输入
        dropout=0.2,     # 0.1→0.2，防止过拟合
        device=device_obj,
    )
)

# PPO超参数优化
model = MaskablePPO(
    'MlpPolicy',
    env,
    policy_kwargs=policy_kwargs,
    batch_size=128,
    learning_rate=2e-4,      # 3e-4→2e-4，更稳定
    n_steps=2048,
    gamma=0.99,
    gae_lambda=0.95,         # 新增：改善优势估计
    clip_range=0.2,          # 新增：PPO裁剪范围
    ent_coef=0.01,           # 新增：熵正则化，鼓励探索
    device=DEVICE,
    verbose=1,
    tensorboard_log=str(OUTPUT_DIR / 'tensorboard')
)
```

**效果**: 模型容量提升2倍，更好地学习高维特征组合

---

### 2. Alpha Pool优化

**修改位置**: `train_alphagen_crypto.py:212-215`

```python
# 原配置
POOL_CAPACITY = 10
IC_LOWER_BOUND = 0.01
L1_ALPHA = 0.005

# 改进配置
POOL_CAPACITY = 20           # 10→20，容纳更多因子组合
IC_LOWER_BOUND = 0.005       # 0.01→0.005，降低准入门槛
L1_ALPHA = 0.01              # 0.005→0.01，增强权重稀疏性
```

**效果**:
- 更大的因子池可以探索更多组合
- 更低的IC阈值避免过早筛掉潜力因子
- 更强的L1正则化防止过拟合

---

### 3. 特征预筛选机制

**新增功能**: 基于单因子IC自动筛选高价值特征

```python
# train_alphagen_crypto.py 新增函数
def _select_top_features(data_dir, feature_cols, top_k):
    """从70+因子中选择Top K个高IC特征"""
    # 1. 快速加载样本数据
    # 2. 计算每个因子的单因子IC
    # 3. 按|IC|排序，选择Top K
    # 4. 返回筛选后的特征列表
```

**使用方式**:
```bash
# 自动选择IC最高的50个特征
export ALPHAGEN_FEATURES_MAX=50
./run_training.sh
```

**效果**: 过滤噪声因子，提升信噪比

---

### 4. 特征重要性分析工具

**新增脚本**: `analyze_features.py`

```bash
# 分析所有70+因子的IC分布
python3 analyze_features.py
```

**输出**:
- 每个因子的IC和Rank IC
- Top 20 / Bottom 20 特征排名
- IC分布统计
- 推荐的ALPHAGEN_FEATURES_MAX配置

**用途**:
- 理解哪些高频因子真正有效
- 识别噪声因子
- 指导特征工程改进

---

## 使用指南

### 快速开始

1. **加载优化配置**:
```bash
source alphagen_config.sh
```

2. **分析特征重要性** (可选但推荐):
```bash
python3 analyze_features.py
# 查看输出，了解哪些因子有效
```

3. **运行训练**:
```bash
./run_training.sh
```

### 配置文件说明

`alphagen_config.sh` 包含所有优化参数:

```bash
# 模型架构
ALPHAGEN_LSTM_LAYERS=3          # LSTM层数
ALPHAGEN_LSTM_DIM=256           # LSTM维度
ALPHAGEN_LSTM_DROPOUT=0.2       # Dropout率

# Alpha Pool
ALPHAGEN_POOL_CAPACITY=20       # 因子池容量
ALPHAGEN_IC_LOWER_BOUND=0.005   # IC准入阈值
ALPHAGEN_POOL_L1_ALPHA=0.01     # L1正则化系数

# 特征选择
ALPHAGEN_FEATURES_MAX=50        # 最多使用50个特征

# 训练
ALPHAGEN_TOTAL_TIMESTEPS=200000 # 训练步数
ALPHAGEN_LEARNING_RATE=2e-4     # 学习率
```

---

## 预期效果

### 改进前 (当前)
- 特征数: 70+
- 模型: 2层×128维LSTM
- Pool: 容量10, IC阈值0.01
- 训练IC: **0.04**

### 改进后 (预期)
- 特征数: 50 (自动筛选)
- 模型: 3层×256维LSTM
- Pool: 容量20, IC阈值0.005
- 训练IC: **0.08-0.12** (预期提升2-3倍)

---

## 进一步优化方向

如果改进后IC仍不理想，可以尝试:

### 1. 切换到Transformer架构
```python
from alphagen.rl.policy import TransformerSharedNet

policy_kwargs = dict(
    features_extractor_class=TransformerSharedNet,
    features_extractor_kwargs=dict(
        n_encoder_layers=2,
        d_model=256,
        n_head=8,
        d_ffn=1024,
        dropout=0.2,
        device=device_obj,
    )
)
```

### 2. 使用ICIR优化目标
需要实现`TensorAlphaCalculator`接口，使用`MeanStdAlphaPool`替代`MseAlphaPool`:
```python
# 优化 IC均值/标准差 而非单点IC
pool = MeanStdAlphaPool(
    capacity=20,
    calculator=tensor_calculator,  # 需要实现
    ic_lower_bound=0.005,
    l1_alpha=0.01,
    lcb_beta=None,  # None=优化ICIR, 或设置如0.5优化LCB
    device=device_obj,
)
```

### 3. 因子分组训练
将70+因子按类型分组(价格类/成交量类/波动率类等)，分别训练后ensemble

### 4. 增加训练数据
```bash
export ALPHAGEN_START_TIME="2018-01-01"  # 延长训练窗口
export ALPHAGEN_SYMBOLS="top50"          # 增加币种
```

---

## 文件清单

### 修改的文件
- `train_alphagen_crypto.py` - 模型架构、Pool配置、特征选择

### 新增的文件
- `alphagen_config.sh` - 优化参数配置
- `analyze_features.py` - 特征重要性分析工具
- `ALPHAGEN_IMPROVEMENTS.md` - 本文档

---

## 监控训练

```bash
# 启动TensorBoard
tensorboard --logdir=./alphagen_output/tensorboard

# 查看实时IC
tail -f ./alphagen_output/tensorboard/events.out.tfevents.*

# 查看最终结果
cat ./alphagen_output/validation_results.json
```

---

## 总结

核心改进思路:
1. **增强模型容量** - 匹配高维输入空间
2. **特征预筛选** - 过滤噪声，提升信噪比
3. **扩大因子池** - 探索更多组合可能性
4. **优化超参数** - 更稳定的训练过程

预期IC从0.04提升至0.08-0.12，提升2-3倍。
