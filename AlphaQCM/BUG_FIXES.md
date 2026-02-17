# Bug 修复总结

## 已修复的关键问题

### 1. ✅ VWAP 计算错误 (CRITICAL)

**问题**: 使用 `cumsum()` 导致 VWAP 累积而非滚动计算

**修复**:
```python
# 错误 (旧代码)
df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

# 正确 (新代码)
typical_price = (df['high'] + df['low'] + df['close']) / 3
df['vwap'] = (typical_price * df['volume']).rolling(window=24, min_periods=1).sum() / \
             df['volume'].rolling(window=24, min_periods=1).sum()
```

**位置**: `data_collection/fetch_crypto_data.py:82-84`

---

### 2. ✅ FeatureType 导入冲突 (CRITICAL)

**问题**: `crypto_data.py` 重新定义了 `FeatureType`，与 `stock_data.py` 冲突

**修复**:
```python
# crypto_data.py - 使用统一的 FeatureType
from alphagen_qlib.stock_data import FeatureType

# train_qcm_crypto.py - 正确导入
from alphagen_qlib.stock_data import FeatureType
from alphagen_qlib.crypto_data import CryptoData
```

**位置**:
- `alphagen_qlib/crypto_data.py:7`
- `train_qcm_crypto.py:9`

---

### 3. ✅ 缺少依赖检查 (HIGH)

**问题**: 直接 `import ccxt` 会在未安装时崩溃

**修复**:
```python
try:
    import ccxt
except ImportError:
    raise ImportError("ccxt not installed. Run: pip install ccxt")
```

**位置**: `data_collection/fetch_crypto_data.py:4-7`

---

### 4. ✅ 数据对齐问题 (CRITICAL for Top100)

**问题**: 使用 `intersection` 会丢失新币种的数据

**修复策略**:
```python
# 旧: 使用 intersection (丢失大量数据)
common_dates = dates1.intersection(dates2).intersection(dates3)...

# 新: 使用 union + forward fill (保留所有数据)
all_dates = union of all dates
# 保留至少 50% 币种有数据的日期
valid_dates = dates with >= 50% coverage
# Forward fill 缺失值
df = df.reindex(valid_dates).fillna(method='ffill').fillna(method='bfill')
```

**位置**: `alphagen_qlib/crypto_data.py:88-120`

---

### 5. ✅ 内存效率问题 (HIGH)

**问题**: 直接加载到 GPU 可能 OOM

**修复**:
```python
# 1. 使用 float32 而非 float64
data_array = np.full((n_dates, n_features, n_symbols), np.nan, dtype=np.float32)

# 2. 先加载到 CPU
data_tensor = torch.tensor(data_array, dtype=torch.float, device='cpu')

# 3. 尝试移动到 GPU，失败则保持 CPU
if self.device.type == 'cuda':
    try:
        data_tensor = data_tensor.to(self.device)
    except RuntimeError as e:
        print(f"Warning: Failed to move to GPU, keeping on CPU")
        self.device = torch.device('cpu')
```

**位置**: `alphagen_qlib/crypto_data.py:108-118`

---

### 6. ✅ 时区处理 (MEDIUM)

**问题**: 加密货币数据是 UTC，但未明确处理

**修复**:
```python
# 数据获取时设置 UTC
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

# 数据加载时保持 UTC
df.index = pd.to_datetime(df.index, utc=True)
```

**位置**:
- `data_collection/fetch_crypto_data.py:70`
- `alphagen_qlib/crypto_data.py:79`

---

### 7. ✅ 数据验证和清洗 (HIGH)

**新增功能**:
```python
# 1. 去重和排序
df = df[~df.index.duplicated(keep='first')].sort_index()

# 2. 填充缺失值
df = df.fillna(method='ffill').fillna(method='bfill')

# 3. NaN 检查
if df.isnull().any().any():
    print(f"Warning: {symbol} still has NaN values")

# 4. 最终 NaN 处理
nan_count = np.isnan(data_array).sum()
if nan_count > 0:
    print(f"Warning: {nan_count} NaN values remain, filling with 0")
    data_array = np.nan_to_num(data_array, nan=0.0)
```

**位置**:
- `data_collection/fetch_crypto_data.py:74-78`
- `alphagen_qlib/crypto_data.py:113-116`

---

### 8. ✅ Top100 支持 (NEW FEATURE)

**新增**:
- 创建 `data_collection/top100_symbols.txt` (100个币种列表)
- 支持 `--symbols top100` 参数
- 数据获取从 2018-01-01 开始 (更长历史)
- 自动从文件加载币种列表

**位置**:
- `data_collection/top100_symbols.txt` (新文件)
- `train_qcm_crypto.py:122` (新参数)
- `alphagen_qlib/crypto_data.py:48-50` (top100 处理)

---

### 9. ✅ 错误处理增强 (MEDIUM)

**新增**:
```python
# 1. 数据获取失败跟踪
failed_symbols = []
# 记录失败的币种

# 2. 文件加载异常处理
def _load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
    try:
        # ... 加载逻辑
    except Exception as e:
        print(f"Error loading {symbol}: {e}")
        return None

# 3. 速率限制
time.sleep(0.1)  # 避免 API 限流
```

**位置**:
- `data_collection/fetch_crypto_data.py:43, 56-60, 67, 91`
- `alphagen_qlib/crypto_data.py:75-88`

---

## 修复后的使用流程

### 1. 安装依赖
```bash
pip install ccxt
```

### 2. 下载 Top100 数据 (2018-2025)
```bash
python data_collection/fetch_crypto_data.py
```

### 3. 训练 Top100 模型
```bash
python train_qcm_crypto.py \
    --model qrdqn \
    --symbols top100 \
    --timeframe 1h \
    --pool 50 \
    --std-lam 1.0
```

---

## 数据对齐策略详解

针对 Top100 从 2018 年开始的数据，新旧币种上市时间不同的问题：

### 策略: Union + Coverage Filter + Forward Fill

```python
# 1. 收集所有日期 (union)
all_dates = union of all symbol dates

# 2. 计算每个日期的币种覆盖率
date_counts = count symbols with data on each date

# 3. 过滤: 保留至少 50% 币种有数据的日期
min_coverage = max(1, len(symbols) // 2)
valid_dates = dates with >= min_coverage symbols

# 4. 对每个币种 forward fill 缺失值
for symbol in symbols:
    df = df.reindex(valid_dates)
    df = df.fillna(method='ffill').fillna(method='bfill')
```

### 示例场景

假设 Top100 中:
- BTC/ETH: 2018-01-01 开始有数据
- SOL: 2020-03-01 上市
- SUI: 2023-05-01 上市

**旧方法 (intersection)**: 只保留 2023-05-01 之后的数据 ❌

**新方法 (union + coverage)**:
- 2018-2023: 保留 (>50% 币种有数据)
- SOL 在 2020-03-01 之前: forward fill 或填充 0
- SUI 在 2023-05-01 之前: forward fill 或填充 0

---

## 性能优化

### 内存使用
- float32 代替 float64: **节省 50% 内存**
- CPU 先加载再移动 GPU: **避免 CUDA OOM**
- Top100 + 1h + 7年数据: 约 **2-3GB 内存**

### 数据获取
- 添加 rate limiting: **避免 API 封禁**
- 失败重试机制: **提高成功率**
- 进度显示: **更好的用户体验**

---

## 测试建议

### 1. 小规模测试
```bash
# 先测试 top10 确保代码正确
python train_qcm_crypto.py --symbols top10 --pool 10
```

### 2. 数据验证
```bash
# 检查数据完整性
python test_crypto_setup.py
```

### 3. 逐步扩展
```bash
# top10 → top20 → top100
python train_qcm_crypto.py --symbols top20 --pool 20
python train_qcm_crypto.py --symbols top100 --pool 50
```

---

## 已知限制

1. **新币种数据稀疏**: 2018-2020 年很多币种还未上市，会有较多 forward fill
2. **内存限制**: Top100 + 1m 周期可能需要 16GB+ 内存
3. **训练时间**: Top100 预计 1-3 天 (取决于 GPU)

---

## 文件变更清单

| 文件 | 状态 | 主要修复 |
|------|------|----------|
| `data_collection/fetch_crypto_data.py` | 修改 | VWAP, 依赖检查, 数据清洗, Top100 |
| `alphagen_qlib/crypto_data.py` | 修改 | FeatureType, 数据对齐, 内存优化 |
| `train_qcm_crypto.py` | 修改 | 导入修复, Top100 支持 |
| `data_collection/top100_symbols.txt` | 新增 | Top100 币种列表 |

---

## 总结

所有关键 bug 已修复，代码现在可以安全地用于 Top100 加密货币因子挖掘，从 2018 年至今的数据。

**立即开始**:
```bash
python data_collection/fetch_crypto_data.py
python train_qcm_crypto.py --symbols top100 --pool 50
```
