# 发现的问题和修复状态

## 已修复 ✅

### 1. CUDA 硬编码问题
- **问题**: `train_qcm_crypto.py:23` 硬编码 `device = torch.device('cuda')`
- **影响**: 没有 GPU 的环境会崩溃
- **修复**: 改为 `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- **文件**: `train_qcm_crypto.py`, `crypto_data.py`

### 2. Calculator 类型检查过严
- **问题**: `QLibStockDataCalculator.__init__` 要求 `StockData` 类型
- **影响**: 无法传入 `CryptoData`
- **修复**: 移除类型注解，改为鸭子类型
- **文件**: `alphagen_qlib/calculator.py`

### 3. 缺少 __init__.py
- **问题**: `alphagen_qlib/` 目录缺少 `__init__.py`
- **影响**: 导入失败
- **修复**: 创建空的 `__init__.py`
- **文件**: `alphagen_qlib/__init__.py`

## 待修复 ⚠️

### 4. 缺少 tensorboard 依赖
- **问题**: `fqf_iqn_qrdqn/agent/base_agent.py` 导入 tensorboard 失败
- **影响**: 无法运行训练
- **修复方案**: 安装 `pip install tensorboard`
- **状态**: 需要用户手动安装

### 5. 数据文件不存在
- **问题**: `AlphaQCM_data/crypto_data/` 目录为空
- **影响**: 训练时会报 FileNotFoundError
- **修复方案**: 运行 `python3 data_collection/fetch_crypto_data.py`
- **状态**: 需要用户下载数据

## 潜在问题 ⚡

### 6. CPU 训练速度慢
- **问题**: 当前环境没有 CUDA，CPU 训练会很慢
- **影响**: Top10 + 1h 可能需要 24-48 小时（vs GPU 4-8 小时）
- **建议**:
  - 使用更小的数据集（top10 而非 top100）
  - 使用更大的时间周期（4h 或 1d 而非 1h）
  - 减少 pool size（--pool 10 而非 20）

### 7. 内存使用
- **问题**: Top100 + 1h + 7年数据约需 2-3GB 内存
- **影响**: 低内存环境可能 OOM
- **建议**: 监控内存使用，必要时减少币种数量

## 快速修复命令

```bash
# 安装所有依赖
./setup_crypto.sh

# 或手动安装
pip install ccxt tensorboard torch pandas numpy pyyaml

# 下载数据（Top10，快速测试）
python3 data_collection/fetch_crypto_data.py

# 开始训练（CPU 友好配置）
python3 train_qcm_crypto.py \
    --model qrdqn \
    --symbols top10 \
    --timeframe 4h \
    --pool 10 \
    --target-periods 10
```

## 验证修复

```bash
# 运行验证脚本
python3 validate_fixes.py

# 测试导入
python3 -c "from train_qcm_crypto import *; print('OK')"

# 检查 CUDA
python3 -c "import torch; print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"
```
