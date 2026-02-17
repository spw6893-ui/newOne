# AlphaGen加密货币因子训练 - 快速开始

## 一键训练

```bash
./run_training.sh
```

## 前置条件

### 1. 重组训练数据（首次运行）

```bash
cd AlphaQCM_data
cat final_dataset_filtered.parquet.gz.part_* > final_dataset_filtered.parquet.gz
gunzip final_dataset_filtered.parquet.gz
cd ..
```

### 2. 确认数据文件

```bash
ls -lh AlphaQCM_data/final_dataset_filtered.parquet
# 应该显示 311MB
```

## 训练流程

脚本会自动执行以下步骤：

1. **数据准备** - 从过滤后的数据集生成AlphaGen训练格式
2. **依赖检查** - 自动安装缺失的依赖包
3. **模型训练** - 使用PPO强化学习挖掘alpha因子
4. **验证评估** - 在验证集上评估因子性能

## 监控训练

在另一个终端运行：

```bash
tensorboard --logdir=./alphagen_output/tensorboard
```

然后访问 http://localhost:6006

## 输出结果

训练完成后会生成：

- `alphagen_output/model_final.zip` - PPO模型检查点
- `alphagen_output/alpha_pool.json` - 10个最佳因子及权重
- `alphagen_output/validation_results.json` - 验证集IC/Rank IC

## 查看结果

```bash
# 查看因子池
cat alphagen_output/alpha_pool.json | jq

# 查看验证结果
cat alphagen_output/validation_results.json | jq
```

## 自定义配置

编辑 `train_alphagen_crypto.py` 修改：

- `SYMBOLS`: 币种选择（'top20', 'top100', 或自定义列表）
- `FEATURES`: 特征选择
- `TOTAL_TIMESTEPS`: 训练步数（默认100000）
- `POOL_CAPACITY`: 因子池大小（默认10）
- `IC_LOWER_BOUND`: IC阈值（默认0.01）

## 硬件需求

- GPU: RTX 3070 (8GB) 或更好
- RAM: 16GB+
- 训练时间: 2-6小时

## 故障排除

### CUDA内存不足

编辑 `train_alphagen_crypto.py`，修改：
```python
DEVICE = 'cpu'  # 使用CPU训练
```

### 数据加载失败

检查数据文件是否完整：
```bash
python -c "import pandas as pd; df = pd.read_parquet('AlphaQCM_data/final_dataset_filtered.parquet'); print(df.shape)"
```

### 依赖安装失败

手动安装：
```bash
pip install torch sb3-contrib stable-baselines3 qlib
```
