from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Optional, Union
    from typing import Dict
    from typing import Callable
    import numpy.typing as npt

    BaseTypes = Union[None, bool, str, int, float]
    StatsOutType = Dict[str, Optional[float]]


class OptimiseStats(object):

    target_metric: str

    def __call__(
        self,
        y: "npt.ArrayLike",
        preds: "npt.ArrayLike",
    ) -> "StatsOutType":
        raise NotImplementedError

    @staticmethod
    def try_else(
        fn: "Callable[[np.ndarray, np.ndarray], float]",
        y: "np.ndarray",
        preds: "np.ndarray"
    ) -> "Optional[float]":
        try:
            result = fn(y, preds)
            if result == np.nan:
                return None
            elif result == np.inf:
                return None
            elif result == -np.inf:
                return None
            else:
                return result
        except Exception:
            return None

    @staticmethod
    def as_2d(x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        elif len(x.shape) > 2:
            raise ValueError("NDCG does not support tensors")

        return x

    @staticmethod
    def as_1d(x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 2:
            x = x.reshape(-1)
        elif len(x.shape) > 2:
            raise ValueError("NDCG does not support tensors")

        return x


def clip(x: np.ndarray) -> np.ndarray:
    try:
        info: "Union[np.finfo, np.iinfo]" = np.finfo(x.dtype)
    except ValueError as e:
        try:
            info = np.iinfo(x.dtype)
        except Exception:
            raise e

    x = np.clip(x, info.min, info.max)
    return x


def exclude_nans(
    fn: "Callable[[np.ndarray, np.ndarray], float]",
    true: "np.ndarray",
    preds: "np.ndarray",
    maximise: bool = False,
) -> float:
    if np.isnan(true).any() or np.isnan(preds).any():
        if maximise:
            return -np.inf
        else:
            return np.inf

    val = fn(true, preds)
    return val


class RegressionStats(OptimiseStats):

    target_metric = "mse"

    def __call__(
        self,
        y: "npt.ArrayLike",
        preds: "npt.ArrayLike",
    ) -> "StatsOutType":
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            median_absolute_error,
            explained_variance_score,
            r2_score,
        )

        from ..sk.metrics import (
            spearmans_correlation,
            pearsons_correlation,
            tau_correlation,
        )

        y_ = clip(np.asarray(y))
        preds_ = clip(np.asarray(preds))

        results = {
            "mae": mean_absolute_error(y_, preds_),
            "median_ae": median_absolute_error(y_, preds_),
            "mse": mean_squared_error(y_, preds_),
            "explained_variance": explained_variance_score(
                y_,
                preds_
            ),
            "r2": r2_score(y_, preds_),
            "pearsons": pearsons_correlation(
                self.as_1d(y_),
                self.as_1d(preds_)
            ),
            "spearmans": spearmans_correlation(
                self.as_1d(y_),
                self.as_1d(preds_)
            ),
            "tau": tau_correlation(
                self.as_1d(y_),
                self.as_1d(preds_)
            ),
        }

        return results


class RankingStats(OptimiseStats):

    target_metric = "pearsons"

    def __call__(
        self,
        y: "npt.ArrayLike",
        preds: "npt.ArrayLike",
    ) -> "StatsOutType":
        from ..sk.metrics import (
            spearmans_correlation,
            pearsons_correlation,
            tau_correlation,
        )
        from sklearn.metrics import ndcg_score

        y_ = clip(np.asarray(y))
        preds_ = clip(np.asarray(preds))

        results: "StatsOutType" = {
            "pearsons": pearsons_correlation(
                self.as_1d(y_),
                self.as_1d(preds_)
            ),
            "spearmans": spearmans_correlation(
                self.as_1d(y_),
                self.as_1d(preds_)
            ),
            "tau": tau_correlation(
                self.as_1d(y_),
                self.as_1d(preds_)
            ),
            "ndcg": ndcg_score(self.as_2d(y_).T, self.as_2d(preds_).T),
        }

        return results


class ClassificationStats(OptimiseStats):

    target_metric = "matthews_correlation"

    def __call__(
        self,
        y: "npt.ArrayLike",
        preds: "npt.ArrayLike",
    ) -> "StatsOutType":

        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            balanced_accuracy_score,
            f1_score,
            log_loss,
            matthews_corrcoef,
            recall_score,
            precision_score,
        )

        y_ = clip(np.asarray(y))
        preds_ = clip(np.asarray(preds))

        results = {
            "recall": recall_score(y_, preds_),
            "precision": precision_score(y_, preds_),
            "accuracy": accuracy_score(y_, preds_),
            "balanced_accuracy": balanced_accuracy_score(y_, preds_),
            "average_precision": average_precision_score(y_, preds_),
            "f1": f1_score(y_, preds_),
            "matthews_correlation": matthews_corrcoef(y_, preds_),
            "log_loss": log_loss(y_, preds_),
        }

        return results
