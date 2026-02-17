from typing import List, Union, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import os
import glob

# Import FeatureType from stock_data to maintain compatibility
from alphagen_qlib.stock_data import FeatureType


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
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> None:

        self._symbols = symbols if isinstance(symbols, list) else self._get_symbols(symbols)
        self.max_backtrack_days = max_backtrack_periods
        self.max_future_days = max_future_periods
        self._start_time = start_time
        self._end_time = end_time
        self._timeframe = timeframe
        self._data_dir = data_dir
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.data, self._dates, self._symbol_ids = self._get_data()

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
            csv_files = glob.glob(os.path.join(self._data_dir, f'*_{self._timeframe}.csv'))
            return sorted([os.path.basename(f).replace(f'_{self._timeframe}.csv', '') for f in csv_files])
        elif symbol_group == 'all':
            csv_files = glob.glob(os.path.join(self._data_dir, f'*_{self._timeframe}.csv'))
            return sorted([os.path.basename(f).replace(f'_{self._timeframe}.csv', '') for f in csv_files])
        else:
            raise ValueError(f"Unknown symbol group: {symbol_group}. Use 'top10', 'top20', 'top100', or 'all'")

    def _load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data for a single symbol"""
        file_path = os.path.join(self._data_dir, f'{symbol}_{self._timeframe}.csv')
        if not os.path.exists(file_path):
            return None

        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)

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
        feature_names = [f.name.lower() for f in self._features]
        n_dates = len(valid_dates)
        n_features = len(self._features)
        n_symbols = len(all_dfs)

        # Use float32 to save memory
        data_array = np.full((n_dates, n_features, n_symbols), np.nan, dtype=np.float32)

        for i, symbol in enumerate(all_dfs.keys()):
            df = all_dfs[symbol].reindex(valid_dates)
            # Forward fill missing values (for newly listed coins)
            df = df.fillna(method='ffill').fillna(method='bfill')

            for j, feat in enumerate(feature_names):
                if feat in df.columns:
                    data_array[:, j, i] = df[feat].values

        # Check for remaining NaN values
        nan_count = np.isnan(data_array).sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values remain, filling with 0")
            data_array = np.nan_to_num(data_array, nan=0.0)

        # Load to CPU first to avoid CUDA OOM
        data_tensor = torch.tensor(data_array, dtype=torch.float, device='cpu')

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
