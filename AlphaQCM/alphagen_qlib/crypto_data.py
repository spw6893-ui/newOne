from typing import List, Union, Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import torch
import os
import glob
import re

# Import FeatureType from stock_data to maintain compatibility
import alphagen_qlib.stock_data as sd


class CryptoData:
    """
    Cryptocurrency data loader for AlphaQCM
    Replaces StockData for crypto markets
    """

    def __init__(self,
                 symbols: Union[str, List[str]],
                 start_time: str,
                 end_time: str,
                 timeframe: str = '1h',
                 data_dir: str = 'AlphaQCM_data/crypto_data',
                 max_backtrack_periods: int = 100,
                 max_future_periods: int = 30,
                 features: Optional[List["sd.FeatureType"]] = None,
                 feature_columns: Optional[Sequence[str]] = None,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> None:

        self.max_backtrack_days = max_backtrack_periods
        self.max_future_days = max_future_periods
        self._start_time = start_time
        self._end_time = end_time
        self._timeframe = timeframe
        self._data_dir = data_dir
        # `_get_symbols('all'/'top100')` 需要依赖 `_data_dir/_timeframe` 扫描文件，
        # 因此必须在这里先完成赋值，再解析 symbols。
        self._symbols = symbols if isinstance(symbols, list) else self._get_symbols(symbols)
        self._features = features if features is not None else list(sd.FeatureType)
        # 当 FeatureType 是动态构造时，需要一个“特征列顺序”来做 index->列名映射
        self._feature_columns = list(feature_columns) if feature_columns is not None else list(
            getattr(sd, "FEATURE_COLUMNS", [])
        )
        # 训练阶段会把宽表“所有因子列”都塞进同一个 CSV，这会让 read_csv 默认读全列非常吃内存。
        # 这里提前计算本次加载真正需要的列名集合：仅包含 FeatureType 对应的列（不含 y_*、质量标记等）。
        self._required_columns = self._compute_required_columns()
        self.device = device
        self.data, self._dates, self._symbol_ids = self._get_data()

    def _compute_required_columns(self) -> List[str]:
        """
        仅保留 AlphaGen 环境会用到的特征列，避免把 label/flag 等无关列读入内存导致 OOM。
        """
        required: set[str] = set()
        if self._feature_columns:
            for ft in self._features:
                j = int(ft)
                if 0 <= j < len(self._feature_columns):
                    required.add(str(self._feature_columns[j]))
        else:
            for c in ("open", "high", "low", "close", "vwap", "volume", "volume_clean"):
                required.add(c)
        # 兜底：如果文件里有 symbol 列，读不读都无所谓；但为了兼容一些下游处理，这里不强依赖。
        return sorted([c for c in required if c])

    @staticmethod
    def _symbol_variants(symbol: str) -> List[str]:
        """
        生成 symbol 的常见变体，提升对不同命名风格（是否含 :USDT 等）的兼容性。
        """
        s = str(symbol).strip()
        out: List[str] = []
        if s:
            out.append(s)

        # 处理类似 BTC_USDT:USDT / BTC_USDT
        if ":USDT" in s:
            out.append(s.replace(":USDT", ""))
        else:
            out.append(s + ":USDT")

        # 去重保持顺序
        seen = set()
        uniq: List[str] = []
        for x in out:
            if x and x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _get_symbols(self, symbol_group: str) -> List[str]:
        """Get symbol list based on group name"""
        if symbol_group == 'top10':
            return ['BTC_USDT', 'ETH_USDT', 'BNB_USDT', 'SOL_USDT', 'XRP_USDT',
                    'ADA_USDT', 'AVAX_USDT', 'DOGE_USDT', 'DOT_USDT', 'MATIC_USDT']
        elif symbol_group == 'top20':
            return ['BTC_USDT', 'ETH_USDT', 'BNB_USDT', 'SOL_USDT', 'XRP_USDT',
                    'ADA_USDT', 'AVAX_USDT', 'DOGE_USDT', 'DOT_USDT', 'MATIC_USDT',
                    'LINK_USDT', 'UNI_USDT', 'ATOM_USDT', 'LTC_USDT', 'ETC_USDT',
                    'APT_USDT', 'ARB_USDT', 'OP_USDT', 'INJ_USDT', 'SUI_USDT']
        elif symbol_group == 'top100':
            # Load from data directory, sorted by availability
            # 兼容两种布局：
            # 1) {symbol}_{timeframe}.csv（原始 crypto_data 口径）
            # 2) {symbol}_train.csv（alphagen_ready 口径）
            csv_files = glob.glob(os.path.join(self._data_dir, f'*_{self._timeframe}.csv'))
            if not csv_files:
                csv_files = glob.glob(os.path.join(self._data_dir, '*_train.csv'))
                return sorted([os.path.basename(f).replace('_train.csv', '') for f in csv_files])
            return sorted([os.path.basename(f).replace(f'_{self._timeframe}.csv', '') for f in csv_files])
        elif symbol_group == 'all':
            csv_files = glob.glob(os.path.join(self._data_dir, f'*_{self._timeframe}.csv'))
            if not csv_files:
                csv_files = glob.glob(os.path.join(self._data_dir, '*_train.csv'))
                return sorted([os.path.basename(f).replace('_train.csv', '') for f in csv_files])
            return sorted([os.path.basename(f).replace(f'_{self._timeframe}.csv', '') for f in csv_files])
        else:
            raise ValueError(f"Unknown symbol group: {symbol_group}. Use 'top10', 'top20', 'top100', or 'all'")

    def _load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data for a single symbol"""
        candidates: List[str] = []
        for s in self._symbol_variants(symbol):
            candidates.append(os.path.join(self._data_dir, f'{s}_{self._timeframe}.csv'))
            candidates.append(os.path.join(self._data_dir, f'{s}_train.csv'))

        file_path = next((p for p in candidates if os.path.exists(p)), "")
        if not file_path:
            return None

        try:
            # 优先按需读列（大幅降低内存峰值）；若列不存在则自动回退全量读取。
            usecols = None
            if self._required_columns:
                # 输入 CSV 约定包含 datetime 列（index 列名），prepare 脚本会写出该字段
                usecols = ["datetime"] + self._required_columns
            try:
                if usecols is None:
                    raise ValueError("skip usecols")
                df = pd.read_csv(file_path, usecols=usecols, parse_dates=["datetime"])
                if "datetime" not in df.columns:
                    raise ValueError("missing datetime column")
                df = df.set_index("datetime")
            except Exception:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            # 兼容：如果 index 不是 datetime，尝试从 datetime 列提取
            idx = pd.to_datetime(df.index, utc=True, errors="coerce")
            if idx.isna().all() and "datetime" in df.columns:
                idx = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
                df = df.drop(columns=["datetime"])
            df.index = idx
            df = df.loc[df.index.notna()].copy()

            # Filter by date range
            df = df[(df.index >= self._start_time) & (df.index <= self._end_time)]

            # Validate data
            if len(df) == 0:
                return None

            return df
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
            return None

    def _get_data(self) -> Tuple[torch.Tensor, pd.DatetimeIndex, List[str]]:
        """Load and process data for all symbols with smart alignment"""
        all_dfs = {}

        print(f"Loading {len(self._symbols)} symbols...")
        for symbol in self._symbols:
            df = self._load_symbol_data(symbol)
            if df is not None:
                all_dfs[symbol] = df
            else:
                print(f"  Skipped {symbol} (no data)")

        if not all_dfs:
            raise ValueError("No data loaded for any symbols")

        print(f"Loaded {len(all_dfs)} symbols successfully")

        # Strategy: Use union of dates, forward-fill missing data
        # This handles coins with different listing dates (e.g., newer coins)
        all_dates = pd.DatetimeIndex([])
        for df in all_dfs.values():
            all_dates = all_dates.union(df.index)
        all_dates = all_dates.sort_values()

        # Calculate data coverage
        date_counts = pd.Series(0, index=all_dates)
        for df in all_dfs.values():
            date_counts[df.index] += 1

        # Filter: keep dates with at least 50% symbol coverage
        min_coverage = max(1, len(all_dfs) // 2)
        valid_dates = date_counts[date_counts >= min_coverage].index

        print(f"Date range: {valid_dates[0]} to {valid_dates[-1]}")
        print(f"Total periods: {len(valid_dates)}")

        # Build 3D tensor: (time, features, symbols)
        # FeatureType -> 实际列名映射
        # 1) 动态 FeatureType：优先用 `_feature_columns`（与 FeatureType index 一一对应）
        # 2) 否则回退到基础 OHLCV/VWAP 映射
        feat_col_map: dict[sd.FeatureType, List[str]] = {}
        if self._feature_columns:
            for ft in self._features:
                j = int(ft)
                if 0 <= j < len(self._feature_columns):
                    feat_col_map[ft] = [self._feature_columns[j]]
        else:
            # fallback for legacy fixed FeatureType
            feat_col_map = {
                sd.FeatureType.OPEN: ["open"],
                sd.FeatureType.HIGH: ["high"],
                sd.FeatureType.LOW: ["low"],
                sd.FeatureType.CLOSE: ["close"],
                sd.FeatureType.VWAP: ["vwap"],
                sd.FeatureType.VOLUME: ["volume", "volume_clean"],
            }
        n_dates = len(valid_dates)
        n_features = len(self._features)
        n_symbols = len(all_dfs)

        # Use float32 to save memory
        data_array = np.full((n_dates, n_features, n_symbols), np.nan, dtype=np.float32)

        for i, symbol in enumerate(all_dfs.keys()):
            df = all_dfs[symbol].reindex(valid_dates)
            # 仅允许 forward-fill（bfill 会引入前视偏差）
            df = df.ffill()

            for j, ft in enumerate(self._features):
                col_candidates = feat_col_map.get(ft, [str(getattr(ft, "name", "")).lower()])
                col = next((c for c in col_candidates if c in df.columns), "")
                if col:
                    data_array[:, j, i] = pd.to_numeric(df[col], errors="coerce").astype("float32").values

        # 可选：按“因子可用性覆盖率”做时段门控（Dynamic Factor Universe 的工程近似）
        # - 目的：某些因子在很长一段历史里根本不存在（或覆盖极低），不应被当作有效 0 值参与训练。
        # - 做法：当某因子在某个时点的跨币种覆盖率 < 阈值时，把该时点的该因子全部置为 NaN，
        #         让 IC/相关系数计算自动忽略这些样本。
        #
        # 注意：这不会动态改变 action space（AlphaGen 的 feature token 集合仍是固定的），
        #       但能避免“因子上线前被硬塞 0”的不合理训练信号。
        min_cov = float(os.environ.get("ALPHAGEN_FACTOR_MIN_COVERAGE", "0").strip() or 0.0)
        if min_cov > 0:
            min_cov = max(0.0, min(1.0, min_cov))
            for j in range(n_features):
                cov = np.mean(~np.isnan(data_array[:, j, :]), axis=1)  # (time,)
                low = cov < min_cov
                if np.any(low):
                    data_array[low, j, :] = np.nan
            print(f"Applied factor availability gating: min_coverage={min_cov}")

        # 统一处理 inf -> NaN（避免 log/除法时污染）
        inf_mask = ~np.isfinite(data_array)
        if np.any(inf_mask):
            data_array[inf_mask] = np.nan

        # Load to CPU first to avoid CUDA OOM
        # 使用 from_numpy 共享内存，避免 torch.tensor 产生第二份拷贝
        data_tensor = torch.from_numpy(data_array)

        # Move to target device only if explicitly requested
        if self.device.type == 'cuda':
            try:
                data_tensor = data_tensor.to(self.device)
            except RuntimeError as e:
                print(f"Warning: Failed to move data to GPU ({e}), keeping on CPU")
                self.device = torch.device('cpu')

        return data_tensor, valid_dates, list(all_dfs.keys())

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        """Keep same name as StockData for compatibility"""
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        """Keep same name as StockData for compatibility"""
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert tensor data to DataFrame

        Parameters:
        - `data`: a tensor of size `(n_days, n_symbols[, n_columns])`, or
        a list of tensors of size `(n_days, n_symbols)`
        - `columns`: an optional list of column names
        """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_symbols, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current CryptoData ({self.n_days})")
        if self.n_stocks != n_symbols:
            raise ValueError(f"number of symbols in the provided tensor ({n_symbols}) doesn't "
                             f"match that of the current CryptoData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._symbol_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
