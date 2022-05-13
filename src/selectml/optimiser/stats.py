from typing import TYPE_CHECKING

from collections import defaultdict

if TYPE_CHECKING:
    from typing import Optional, Union
    from typing import Dict, List
    import numpy.typing as npt
    import pandas as pd

    BaseTypes = Union[None, bool, str, int, float]
    StatsOutType = Dict[str, Optional[float]]


class OptimiseStats(object):

    def __init__(self):
        self.records: "defaultdict[str, List[BaseTypes]]" = defaultdict(list)
        return

    def __call__(
        self,
        y: "npt.ArrayLike",
        preds: "npt.ArrayLike",
    ) -> "StatsOutType":
        raise NotImplementedError

    def as_df(self) -> "pd.DataFrame":
        return pd.DataFrame(self.records)


class RegressionStats(OptimiseStats):

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

        results = {
            "mae": mean_absolute_error(y, preds),
            "median_ae": median_absolute_error(y, preds),
            "mse": mean_squared_error(y, preds),
            "explained_variance": explained_variance_score(
                y,
                preds
            ),
            "r2": r2_score(y, preds),
            "pearsons": pearsons_correlation(y, preds),
            "spearmans": spearmans_correlation(y, preds),
            "tau": tau_correlation(y, preds),
        }

        for k, v in results.items():
            self.records[k].append(v)

        return results


class RankingStats(OptimiseStats):

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

        results: "StatsOutType" = {
            "pearsons": pearsons_correlation(y, preds),
            "spearmans": spearmans_correlation(y, preds),
            "tau": tau_correlation(y, preds),
        }

        for k, v in results.items():
            self.records[k].append(v)

        return results


class ClassificationStats(OptimiseStats):

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

        results = {
            "recall": recall_score(y, preds),
            "precision": precision_score(y, preds),
            "accuracy": accuracy_score(y, preds),
            "balanced_accuracy": balanced_accuracy_score(y, preds),
            "average_precision": average_precision_score(y, preds),
            "f1": f1_score(y, preds),
            "matthews_correlation": matthews_corrcoef(y, preds),
            "log_loss": log_loss(y, preds),
        }

        for k, v in results.items():
            self.records[k].append(v)

        return results
