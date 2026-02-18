from typing import List, Optional
from torch import Tensor
import torch
from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import masked_mean_std, normalize_by_day
from alphagen_qlib.stock_data import StockData


class QLibStockDataCalculator(AlphaCalculator):
    def __init__(self, data, target: Optional[Expression]):
        self.data = data
        if target is None: # Combination-only mode
            self.target_value = None
        else:
            self.target_value = self._normalize_keep_nan(target.evaluate(self.data))

    @staticmethod
    def _normalize_keep_nan(value: Tensor) -> Tensor:
        """
        归一化但保留 NaN（非常关键）：
        - AlphaGen/IC 计算阶段用 NaN 表达“该样本该因子不可用”（例如：因子上线前、币种上市前）。
        - 如果把 NaN 填成 0，会把“不可用”误当作“有效的 0 值”，导致时段/可用性逻辑被破坏。
        """
        nan_mask = torch.isnan(value)
        mean, std = masked_mean_std(value, mask=nan_mask)
        out = (value - mean[:, None]) / std[:, None]
        out[nan_mask] = torch.nan
        return out

    def _calc_alpha(self, expr: Expression) -> Tensor:
        return self._normalize_keep_nan(expr.evaluate(self.data))

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def make_ensemble_alpha(self, exprs: List[Expression], weights: List[float]) -> Tensor:
        """
        线性组合时做“NaN 友好”的合成：
        - 单个因子缺失（NaN）时，把该因子的贡献当作 0；
        - 只有当该时点所有因子都缺失时，组合结果才标记为 NaN。
        这样既能保留“不可用”的时段语义，又不会让少量缺失导致整列变 NaN。
        """
        n = len(exprs)
        if n == 0:
            raise ValueError("exprs 不能为空")
        stacked = torch.stack([self._calc_alpha(exprs[i]) * float(weights[i]) for i in range(n)], dim=0)
        all_nan = torch.isnan(stacked).all(dim=0)
        out = torch.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0).sum(dim=0)
        out[all_nan] = torch.nan
        return out

    def calc_single_IC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_rIC(value, self.target_value)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            ic = batch_pearsonr(ensemble_value, self.target_value).mean().item()
            return ic

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            rank_ic = batch_spearmanr(ensemble_value, self.target_value).mean().item()
            return rank_ic

    def calc_pool_all_ret(self, exprs: List[Expression], weights: List[float]):
        return self.calc_pool_IC_ret(exprs, weights), self.calc_pool_rIC_ret(exprs, weights)

class TestStockDataCalculator(AlphaCalculator):
    def __init__(self, data: StockData, target: Optional[Expression]):
        self.data = data
        
        if target is None: # Combination-only mode
            self.target_value = None
        else:
            self.target_value = normalize_by_day(target.evaluate(self.data)).cpu().half()

    def _calc_alpha(self, expr: Expression) -> Tensor:
        return normalize_by_day(expr.evaluate(self.data)).cpu().half()

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def make_ensemble_alpha(self, exprs: List[Expression], weights: List[float]) -> Tensor:
        n = len(exprs)
        factors: List[Tensor] = [self._calc_alpha(exprs[i]) * weights[i] for i in range(n)]
        return sum(factors)  # type: ignore

    def calc_single_IC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_rIC(value, self.target_value)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            ic = batch_pearsonr(ensemble_value, self.target_value).mean().item()
            return ic

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            rank_ic = batch_spearmanr(ensemble_value, self.target_value).mean().item()
            return rank_ic

    def calc_pool_all_ret(self, exprs: List[Expression], weights: List[float]):
        return self.calc_pool_IC_ret(exprs, weights), self.calc_pool_rIC_ret(exprs, weights)
