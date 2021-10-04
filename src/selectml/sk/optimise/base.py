from math import floor, log10, sqrt

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy.typing as npt
    from ..weighting import WeightingInterface
    import optuna

from typing import Sequence, Union, Optional, Any
from typing import List, Tuple, Mapping, Dict
from typing import Iterator


from ..weighting import variance_weights, distance_weights, cluster_weights

WEIGHT_FNS: Dict[str, Optional["WeightingInterface"]] = {
    "none": None,
    "variance": variance_weights,
    "distance": distance_weights,
    "cluster": cluster_weights,
}


BaseTypes = Union[None, bool, str, int, float]


class OptimiseModel(object):

    use_weights: bool = False

    def __init__(
        self,
        experiment: pd.DataFrame,
        markers: pd.DataFrame,
        individual_columns: Union[str, Sequence[str]],
        response_columns: Union[str, Sequence[str]],
        grouping_columns: Union[str, Sequence[str], None] = None,
        stat: str = "mae",
        seed: Optional[int] = None,
        model_name: str = "test",
        ploidy: int = 2
    ):
        self.experiment: pd.DataFrame = experiment
        self.markers: pd.DataFrame = markers

        if isinstance(individual_columns, str):
            self.individual_columns: List[str] = [individual_columns]
        else:
            self.individual_columns = list(individual_columns)

        if isinstance(response_columns, str):
            self.response_columns: List[str] = [response_columns]
        else:
            self.response_columns = list(response_columns)

        if grouping_columns is None:
            self.grouping_columns: List[str] = []
        elif isinstance(grouping_columns, (str, int)):
            self.grouping_columns = [grouping_columns]
        else:
            self.grouping_columns = list(grouping_columns)

        self.marker_columns: List[str] = [
            str(c) for c in self.markers.columns
            if c not in self.individual_columns
        ]

        assert stat in ("mae", "median_ae", "pearsons", "spearmans",
                        "tau", "explained_variance", "r2")
        self.stat: str = stat
        self.seed = seed
        self.model_name: str = model_name
        self.ploidy: int = ploidy
        return

    @classmethod
    def regression_stats(
        cls,
        y: "npt.ArrayLike",
        preds: "npt.ArrayLike",
        y_means: "npt.ArrayLike",
        means_preds: "npt.ArrayLike",
    ) -> Dict[str, float]:
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            median_absolute_error,
            explained_variance_score,
            r2_score,
        )

        from ..metrics import (
            spearmans_correlation,
            pearsons_correlation,
            tau_correlation,
        )

        results = {
            "mae": mean_absolute_error(y, preds),
            "mae_means": mean_absolute_error(
                y_means,
                means_preds
            ),
            "median_ae": median_absolute_error(y, preds),
            "median_ae_means": median_absolute_error(
                y_means,
                means_preds
            ),
            "mse": mean_squared_error(y, preds),
            "mse_means": mean_squared_error(
                y_means,
                means_preds
            ),
            "pearsons": pearsons_correlation(y, preds),
            "pearsons_means": pearsons_correlation(
                y_means,
                means_preds
            ),
            "spearmans": spearmans_correlation(y, preds),
            "spearmans_means": spearmans_correlation(
                y_means,
                means_preds
            ),
            "tau": tau_correlation(y, preds),
            "tau_means": tau_correlation(
                y_means,
                means_preds
            ),
            "explained_variance": explained_variance_score(
                y,
                preds
            ),
            "explained_variance_means": explained_variance_score(
                y_means,
                means_preds
            ),
            "r2": r2_score(y, preds),
            "r2_means": r2_score(y_means, means_preds),
        }

        return results

    def cv(self, k: int = 5) -> Iterator[Tuple[
        int,
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
        "npt.ArrayLike",
    ]]:
        from sklearn.model_selection import GroupKFold

        if self.seed is not None:
            np.random.seed(self.seed)

        data = pd.merge(
            self.experiment,
            self.markers,
            on=self.individual_columns,
            how="inner"
        )

        data["_indiv_"] = (
            data.loc[:, self.individual_columns]
            .apply(lambda g: "_".join(map(str, g)), axis=1)
            .values
        )
        data["_indiv_grouping_"] = (
            data.loc[:, self.individual_columns + self.grouping_columns]
            .apply(lambda g: "_".join(map(str, g)), axis=1)
            .values
        )

        for i, (train_idx, test_idx) in enumerate(
            GroupKFold(k)
            .split(
                data,
                groups=data["_indiv_"].values,
            )
        ):
            train = data.iloc[train_idx]
            X_train = train.loc[:, self.grouping_columns + self.marker_columns]
            y_train = train.loc[:, self.response_columns]
            indiv_train = train.loc[:, "_indiv_"]
            grouping_train = train.loc[:, "_indiv_grouping_"]

            X_train_means = (
                train
                .groupby("_indiv_grouping_")
                [self.grouping_columns + self.marker_columns]
                .mean()
            )
            y_train_means = (
                train
                .groupby("_indiv_grouping_")
                [self.response_columns]
                .mean()
                .loc[X_train_means.index.values, ]
            )
            indiv_train_means = (
                train
                .groupby("_indiv_grouping_")
                ["_indiv_"]
                .first()
                .loc[X_train_means.index.values, ]
            )
            grouping_train_means = X_train_means.index.values

            test = data.iloc[test_idx]
            X_test = test.loc[:, self.grouping_columns + self.marker_columns]
            y_test = test.loc[:, self.response_columns]
            indiv_test = test.loc[:, "_indiv_"]
            grouping_test = test.loc[:, "_indiv_grouping_"]

            X_test_means = (
                test
                .groupby("_indiv_grouping_")
                [self.grouping_columns + self.marker_columns]
                .mean()
            )
            y_test_means = (
                test
                .groupby("_indiv_grouping_")
                [self.response_columns]
                .mean()
                .loc[X_test_means.index.values, ]
            )
            indiv_test_means = (
                test
                .groupby("_indiv_grouping_")
                ["_indiv_"]
                .first()
                .loc[X_test_means.index.values, ]
            )
            grouping_test_means = X_test_means.index.values

            yield (
                i,
                X_train.values,
                y_train.values,
                indiv_train.values,
                grouping_train.values,
                X_train_means.values,
                y_train_means.values,
                indiv_train_means.values,
                grouping_train_means,
                X_test.values,
                y_test.values,
                indiv_test.values,
                grouping_test.values,
                X_test_means.values,
                y_test_means.values,
                indiv_test_means.values,
                grouping_test_means
            )
        return

    def cv_eval(
        self,
        params: Dict[str, Any],
        k: int = 5,
        **kwargs
    ) -> pd.DataFrame:
        from copy import deepcopy
        params = deepcopy(params)  # Avoid mutating

        out = []
        weight_fn = WEIGHT_FNS[params.get("weight", "none")]

        for (
            i,
            X_train,
            y_train,
            indiv_train,
            grouping_train,
            X_train_means,
            y_train_means,
            indiv_train_means,
            grouping_train_means,
            X_test,
            y_test,
            indiv_test,
            grouping_test,
            X_test_means,
            y_test_means,
            indiv_test_means,
            grouping_test_means
        ) in self.cv(k):

            if weight_fn is None:
                weights = None
            else:
                weights = weight_fn(
                    X_train,
                    y_train,
                    grouping_train,
                    params["train_means"]
                )

            if params["train_means"]:
                X = X_train_means
                y = y_train_means
                indiv_fit = indiv_train_means
                grouping_fit = grouping_train_means
            else:
                X = X_train
                y = y_train
                indiv_fit = indiv_train
                grouping_fit = grouping_train

            params["nsamples"] = len(np.unique(indiv_train))

            model = self.fit(
                params,
                X,
                y,
                indiv_fit,
                grouping_fit,
                weights,
                **kwargs
            )

            train_preds = self.predict(model, X_train)
            train_means_preds = self.predict(model, X_train_means)
            test_preds = self.predict(model, X_test)
            test_means_preds = self.predict(model, X_test_means)

            train_stats = self.regression_stats(
                y_train,
                train_preds,
                y_train_means,
                train_means_preds,
            )

            these_stats: Dict[str, Union[float, str]] = {
                f"train_{k}": v
                for k, v
                in train_stats.items()
            }

            test_stats = self.regression_stats(
                y_test,
                test_preds,
                y_test_means,
                test_means_preds,
            )
            these_stats.update({f"test_{k}": v for k, v in test_stats.items()})
            these_stats.update({"cv": i, "name": self.model_name})
            out.append(these_stats)

        return pd.DataFrame(out)

    def __call__(self, trial: "optuna.Trial") -> float:
        params = self.sample_common_params(trial)
        params.update(self.sample_params(trial))

        try:
            stats = self.cv_eval(params)
        except ValueError as e:
            raise e
            # Sometimes calculating mae etc doesn't work.
            return np.nan

        stats_summary = {
            "mae": stats["test_mae_means"].mean(),
            "median_ae": stats["test_median_ae_means"].mean(),
            "pearsons": stats["test_pearsons_means"].mean(),
            "spearmans": stats["test_spearmans_means"].mean(),
            "tau": stats["test_tau_means"].mean(),
            "explained_variance": (
                stats["test_explained_variance_means"].mean()
            ),
            "r2": stats["test_r2_means"].mean(),
        }

        for k, v in stats_summary.items():
            trial.set_user_attr(k, v)

        return stats_summary[self.stat]

    @classmethod
    def sample_common_params(
        cls,
        trial: "optuna.Trial"
    ) -> Dict[str, BaseTypes]:
        params = {}

        if cls.use_weights:
            params["weight"] = trial.suggest_categorical(
                "weight",
                ["none", "variance", "distance", "cluster"]
            )
        else:
            params["weight"] = trial.suggest_categorical(
                "weight",
                ["none"]
            )

        params["train_means"] = trial.suggest_categorical(
            "train_means",
            [True, False]
        )

        params["min_maf"] = trial.suggest_float(
            "min_maf",
            0.00000001,
            0.3,
        )
        return params

    def sample_params(self, trial: "optuna.Trial"):
        raise NotImplementedError()

    def train_from_params(self, params: Mapping[str, BaseTypes], **kwargs):
        params_ = dict(params)

        data = pd.merge(
            self.experiment,
            self.markers,
            on=self.individual_columns,
            how="inner"
        )

        data["_indiv_"] = (
            data.loc[:, self.individual_columns]
            .apply(lambda g: "_".join(map(str, g)), axis=1)
            .values
        )
        data["_indiv_grouping_"] = (
            data.loc[:, self.individual_columns + self.grouping_columns]
            .apply(lambda g: "_".join(map(str, g)), axis=1)
            .values
        )

        if params_["train_means"]:
            X = (
                data
                .groupby("_indiv_grouping_")
                [self.grouping_columns + self.marker_columns]
                .mean()
            )
            y = (
                data
                .groupby("_indiv_grouping_")
                [self.response_columns]
                .mean()
                .loc[X.index.values, ]
            )
            indiv = (
                data
                .groupby("_indiv_grouping_")
                ["_indiv_"]
                .first()
                .loc[X.index.values, ]
            )
            grouping = X.index.values
        else:
            X = data.loc[:, self.grouping_columns + self.marker_columns]
            y = data.loc[:, self.response_columns]
            indiv = data.loc[:, "_indiv_"]
            grouping = data.loc[:, "_indiv_grouping_"].values

        params_["nsamples"] = len(np.unique(indiv))
        weight_fn = WEIGHT_FNS[params_.get("weight", "none")]

        if weight_fn is None:
            weights = None
        else:
            # Need to do this because variance needed.
            weights = weight_fn(
                (data
                 .loc[:, self.grouping_columns + self.marker_columns]
                 .values),
                data.loc[:, self.response_columns].values,
                data.loc[:, "_indiv_grouping_"].values,
                params_["train_means"]
            )

        if isinstance(weights, pd.Series):
            weights = weights.values

        model = self.fit(
            params_,
            X.values,
            y.values,
            indiv.values,
            grouping,
            weights,
            **kwargs
        )
        return model

    def best_model(self, study: "optuna.Study", **kwargs):
        params = study.best_params
        return self.train_from_params(params)

    def model(self, params: Dict[str, Any]):
        raise NotImplementedError()

    def fit(
        self,
        params: Dict[str, Any],
        X: "npt.ArrayLike",
        y: "npt.ArrayLike",
        individuals: Optional["npt.ArrayLike"] = None,
        grouping: Optional["npt.ArrayLike"] = None,
        sample_weights: Optional["npt.ArrayLike"] = None,
        **kwargs
    ):
        raise NotImplementedError()

    @classmethod
    def predict(
        cls,
        model,
        X: "npt.ArrayLike",
        grouping: Optional["npt.ArrayLike"] = None,
    ) -> np.ndarray:
        raise NotImplementedError()

    def starting_points(
        self
    ) -> List[Dict[str, BaseTypes]]:
        raise NotImplementedError()


class SKModel(OptimiseModel):

    @classmethod
    def predict(
        cls,
        model,
        X: "npt.ArrayLike",
        grouping: Optional["npt.ArrayLike"] = None
    ) -> np.ndarray:
        preds = model.predict(np.array(X))

        if len(preds.shape) == 1:
            preds = preds.reshape(-1, 1)

        return preds

    def fit(
        self,
        params: Dict[str, Any],
        X: "npt.ArrayLike",
        y: "npt.ArrayLike",
        individuals: Optional["npt.ArrayLike"] = None,
        grouping: Optional["npt.ArrayLike"] = None,
        sample_weights: Optional["npt.ArrayLike"] = None,
        **kwargs
    ):
        """ Default is suitable for sklearn compatible models. """

        model = self.model(params)
        indiv_kwargs = {
            "preprocessor__features__markers__individuals": individuals,
            "preprocessor__features__grouping__individuals": grouping,
            "preprocessor__interactions__interactions__individuals": grouping,
        }

        X_ = np.array(X)
        y_ = np.array(y)

        if (
            (len(self.response_columns) == 1) and
            (y_.shape[-1] == 1)
        ):
            y_ = y_.reshape(-1)

        if sample_weights is None:
            model.fit(X_, y_, **indiv_kwargs, **kwargs)
        else:
            model.fit(
                X_,
                y_,
                model__sample_weight=sample_weights,
                **indiv_kwargs,
                **kwargs
            )

        return model

    def _sample_transformed_target_params(
        self,
        trial: "optuna.Trial",
        options: List[str] = [
            "passthrough",
            "stdnorm",
            "quantile"
        ],
    ) -> Dict[str, BaseTypes]:
        params = {}

        target = trial.suggest_categorical(
            "target_transformer",
            options
        )
        params["target_transformer"] = target

        if target == "quantile":
            params["target_transformer_quantile_distribution"] = (
                trial.suggest_categorical(
                    "target_transformer_quantile_distribution",
                    ["uniform", "normal"]
                )
            )
        return params

    def _sample_transformed_target_model(self, params: Dict[str, Any]):
        from sklearn.preprocessing import StandardScaler, QuantileTransformer

        preprocessor = params["target_transformer"]

        if preprocessor == "stdnorm":
            g = StandardScaler()
        elif preprocessor == "quantile":
            d = params["target_transformer_quantile_distribution"]
            g = QuantileTransformer(
                output_distribution=d,
                n_quantiles=min([1000, round(params["nsamples"] / 2)])
            )
        else:
            assert preprocessor == "passthrough"
            g = None  # Unity function

        return g

    def _sample_marker_preprocessing_params(
        self,
        trial: "optuna.Trial",
        options: List[str] = [
            "drop",
            "passthrough",
            "maf",
            "onehot",
        ],
    ) -> Dict[str, BaseTypes]:
        params = {}
        preprocessor = trial.suggest_categorical(
            "marker_preprocessor",
            options
        )
        params["marker_preprocessor"] = preprocessor
        return params

    def _sample_marker_preprocessing_model(self, params: Dict[str, Any]):
        from sklearn.preprocessing import OneHotEncoder
        from selectml.sk.preprocessor import (
            MAFScaler
        )

        preprocessor = params["marker_preprocessor"]

        if preprocessor == "drop":
            g = "drop"

        elif preprocessor == "passthrough":
            g = "passthrough"

        elif preprocessor == "onehot":
            g = OneHotEncoder(
                categories="auto",
                drop=None,
                handle_unknown="ignore"
            )

        elif preprocessor == "maf":
            g = MAFScaler(ploidy=self.ploidy)

        return g

    def _sample_dist_preprocessing_params(
        self,
        trial: "optuna.Trial",
        options: List[str] = [
            "drop",
            "vanraden",
            "hamming",
            "manhattan",
            "euclidean"
        ],
    ) -> Dict[str, BaseTypes]:
        params = {}
        preprocessor = trial.suggest_categorical(
            "dist_preprocessor",
            options
        )
        params["dist_preprocessor"] = preprocessor
        return params

    def _sample_dist_preprocessing_model(self, params: Dict[str, Any]):
        from sklearn.preprocessing import RobustScaler
        from sklearn.pipeline import Pipeline
        from ..distance import (
            VanRadenSimilarity,
            ManhattanDistance,
            EuclideanDistance,
            HammingDistance,
        )

        preprocessor = params["dist_preprocessor"]

        if preprocessor == "drop":
            return "drop"

        if preprocessor == "vanraden":
            p = VanRadenSimilarity(ploidy=self.ploidy, distance=True)

        elif preprocessor == "hamming":
            p = HammingDistance()

        elif preprocessor == "manhattan":
            p = ManhattanDistance()

        elif preprocessor == "euclidean":
            p = EuclideanDistance()

        if p is None:
            return p
        elif p == "drop":
            return p
        else:
            return Pipeline([("dist", p), ("scaler", RobustScaler())])

    def _sample_nonlinear_preprocessing_params(
        self,
        trial: "optuna.Trial",
        options: List[str] = [
            "drop",
            "rbf",
            "laplacian",
            "poly"
        ],
    ) -> Dict[str, BaseTypes]:
        params = {}
        preprocessor = trial.suggest_categorical(
            "nonlinear_preprocessor",
            options
        )
        params["nonlinear_preprocessor"] = preprocessor

        nsamples = floor(self.markers.shape[0] / 2)

        ncomponents = min([
            nsamples,
            floor(log10(nsamples) * 50)
        ])

        if preprocessor in ("rbf", "laplacian", "poly"):
            params["nonlinear_ncomponents"] = trial.suggest_categorical(
                "nonlinear_ncomponents",
                [ncomponents]
            )

        if preprocessor == "rbf":
            params["rbf_gamma"] = trial.suggest_float(
                "rbf_gamma",
                1e-15,
                0.5
            )
        elif preprocessor == "laplacian":
            params["laplacian_gamma"] = trial.suggest_float(
                "laplacian_gamma",
                1e-15,
                0.5
            )
        elif preprocessor == "poly":
            params["poly_gamma"] = trial.suggest_float(
                "poly_gamma",
                0.1,
                20
            )
        return params

    def _sample_nonlinear_preprocessing_model(self, params: Dict[str, Any]):
        from sklearn.kernel_approximation import Nystroem
        from sklearn.preprocessing import RobustScaler
        from sklearn.pipeline import Pipeline

        preprocessor = params["nonlinear_preprocessor"]

        if preprocessor == "drop":
            p = "drop"

        elif preprocessor == "rbf":
            ncomponents = params["nonlinear_ncomponents"]
            p = Nystroem(
                kernel="rbf",
                gamma=params["rbf_gamma"],
                n_components=ncomponents
            )

        elif preprocessor == "laplacian":
            ncomponents = params["nonlinear_ncomponents"]
            p = Nystroem(
                kernel="laplacian",
                gamma=params["laplacian_gamma"],
                n_components=ncomponents
            )

        elif preprocessor == "poly":
            ncomponents = params["nonlinear_ncomponents"]
            p = Nystroem(
                kernel="poly",
                gamma=params["poly_gamma"],
                n_components=ncomponents,
                degree=2
            )

        if p is None:
            return None
        elif p == "drop":
            return p
        else:
            return Pipeline([("nonlinear", p), ("scaler", RobustScaler())])

    def _sample_feature_selection_params(
        self,
        trial: "optuna.Trial",
        options: List[str] = ["drop", "passthrough", "rf", "relief"]
    ) -> Dict[str, BaseTypes]:
        params = {}

        selector = trial.suggest_categorical("feature_selector", options)
        params["feature_selector"] = selector
        nmarkers = len(self.marker_columns)

        if selector == "rf":
            params["feature_selection_rf_min_impurity_decrease"] = (
                trial.suggest_float(
                    "feature_selection_rf_min_impurity_decrease",
                    0,
                    10
                )
            )

            params["feature_selection_nfeatures"] = (
                trial.suggest_int(
                    "feature_selection_nfeatures",
                    min([100, round(nmarkers / 2)]),
                    nmarkers - 1,
                    step=min([100, round(nmarkers / 4)])
                )
            )

        elif selector == "relief":
            params["feature_selection_nfeatures"] = (
                trial.suggest_int(
                    "feature_selection_nfeatures",
                    min([100, round(nmarkers / 2)]),
                    nmarkers - 1,
                    step=min([100, round(nmarkers / 4)]),
                )
            )

        elif selector == "maf":
            params["feature_selection_maf_threshold"] = (
                trial.suggest_uniform(
                    "feature_selection_maf_threshold",
                    0.0001,
                    0.2,
                )
            )

        return params

    def _sample_feature_selection_model(self, params: Dict[str, Any]):
        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import RandomForestRegressor
        from ..feature_selection import (MAFSelector, MultiSURF)

        selector = params["feature_selector"]

        wrap_sfm = False
        if selector == "drop":
            s = "drop"

        elif selector == "passthrough":
            s = "passthrough"

        elif selector == "rf":
            wrap_sfm = True
            s = RandomForestRegressor(
                criterion="mae",
                max_depth=5,
                n_estimators=500,
                max_features=0.1,
                min_samples_split=2,
                min_samples_leaf=1,
                min_impurity_decrease=(
                    params["feature_selection_rf_min_impurity_decrease"]
                ),
                bootstrap=False,
                oob_score=False,
                n_jobs=1,
                random_state=self.seed,
            )

        elif selector == "relief":
            s = MultiSURF(
                n=params["feature_selection_nfeatures"],
                nepoch=10,
                sd=1
            )
        elif selector == "maf":
            s = MAFSelector(
                threshold=params["feature_selection_maf_threshold"],
                ploidy=self.ploidy
            )

        if wrap_sfm:
            nfeatures = params["feature_selection_nfeatures"]
            s = SelectFromModel(
                estimator=s,
                prefit=False,
                max_features=nfeatures
            )

        return s

    def _sample_grouping_preprocessing_params(
        self,
        trial: "optuna.Trial",
        grouping_options: List[str] = ["drop", "passthrough",
                                       "onehot", "pca"]
    ) -> Dict[str, BaseTypes]:
        params = {}

        if len(self.grouping_columns) > 0:
            preprocessor = trial.suggest_categorical(
                "grouping_preprocessor",
                grouping_options
            )
        else:
            preprocessor = trial.suggest_categorical(
                "grouping_preprocessor",
                ["drop"]
            )
        params["grouping_preprocessor"] = preprocessor

        if preprocessor in ("factor", "pca"):
            # 1->1 10->3 100->10
            ncomponents = floor(sqrt(len(self.grouping_columns)))
            params["grouping_ncomponents"] = trial.suggest_categorical(
                "grouping_ncomponents",
                [ncomponents]
            )
        return params

    def _sample_grouping_preprocessing_model(self, params: Dict[str, Any]):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.decomposition import FactorAnalysis, TruncatedSVD

        preprocessor = params["grouping_preprocessor"]

        if preprocessor == "drop":
            g = "drop"

        elif preprocessor == "passthrough":
            g = "passthrough"

        elif preprocessor == "onehot":
            g = OneHotEncoder(
                categories="auto",
                drop="if_binary",
                handle_unknown="error",
                sparse=False,
            )

        elif preprocessor == "factor":
            g = Pipeline([
                (
                    "ohe",
                    OneHotEncoder(
                        categories="auto",
                        drop=None,
                        handle_unknown="ignore",
                        sparse=False
                    )
                ),
                (
                    "factor",
                    FactorAnalysis(n_components=params["grouping_ncomponents"])
                )
            ])
        elif preprocessor == "pca":
            g = Pipeline([
                (
                    "ohe",
                    OneHotEncoder(
                        categories="auto",
                        drop=None,
                        handle_unknown="ignore"
                    )
                ),
                (
                    "pca",
                    TruncatedSVD(n_components=params["grouping_ncomponents"])
                )
            ])

        return g

    def _sample_interactions_preprocessing_params(
        self,
        trial: "optuna.Trial",
        options: List[str] = ["drop", "poly"]
    ) -> Dict[str, BaseTypes]:
        params = {}
        preprocessor = trial.suggest_categorical(
            "interactions",
            options
        )

        params["interactions"] = preprocessor

        nsamples = floor(self.markers.shape[0] / 2)
        ncomponents = min([
            nsamples,
            floor(log10(nsamples) * 50)
        ])

        if preprocessor == "poly":
            params["interactions_ncomponents"] = trial.suggest_categorical(
                "interactions_ncomponents",
                [ncomponents]
            )

        return params

    def _sample_interactions_preprocessing_model(
        self,
        params: Dict[str, Any],
    ):
        from sklearn.kernel_approximation import Nystroem

        preprocessor = params["interactions"]

        if preprocessor == "drop":
            p = "drop"
        elif preprocessor == "poly":
            ncomponents = params["interactions_ncomponents"]
            p = Nystroem(
                kernel="poly",
                n_components=ncomponents,
                degree=2
            )
        return p

    def sample_preprocessing_params(
        self,
        trial: "optuna.Trial",
        target_options: List[str] = [
            "passthrough",
            "stdnorm",
            "quantile",
        ],
        marker_options: List[str] = [
            "drop",
            "maf",
            "passthrough",
            "onehot",
        ],
        dist_options: List[str] = [
            "drop",
            "vanraden",
            "hamming",
            "manhattan",
            "euclidean",
        ],
        nonlinear_options: List[str] = [
            "drop",
            "rbf",
            "laplacian",
            "poly"
        ],
        feature_selection_options: List[str] = [
            "drop",
            "passthrough",
            "rf",
            "relief",
        ],
        grouping_options: List[str] = ["drop", "passthrough",
                                       "onehot", "pca"],
        grouping_interaction_options: List[str] = ["drop", "poly"]
    ) -> Dict[str, BaseTypes]:
        params = {}
        params.update(
            self._sample_transformed_target_params(trial, target_options)
        )
        params.update(self._sample_marker_preprocessing_params(
            trial,
            marker_options
        ))
        params.update(self._sample_dist_preprocessing_params(
            trial,
            dist_options
        ))
        params.update(self._sample_nonlinear_preprocessing_params(
            trial,
            nonlinear_options
        ))
        params.update(self._sample_feature_selection_params(
            trial,
            feature_selection_options
        ))
        params.update(self._sample_grouping_preprocessing_params(
            trial,
            grouping_options
        ))
        params.update(
            self._sample_interactions_preprocessing_params(
                trial,
                grouping_interaction_options
            )
        )

        nsamples = self.markers.shape[0]
        nparams = 0
        if (
            params["marker_preprocessor"] == "drop" or
            params["feature_selector"] == "drop"
        ):
            pass
        elif params["feature_selector"] == "passthrough":
            nparams += len(self.marker_columns)
        elif params["feature_selector"] == "rf":
            assert isinstance(params["feature_selector_nfeatures"], int)
            nparams += params["feature_selector_nfeatures"]

        if params["dist_preprocessor"] != "drop":
            nparams += round(nsamples / 2)

        if params["nonlinear_preprocessor"] != "drop":
            assert isinstance(params["nonlinear_ncomponents"], int)
            nparams += params["nonlinear_ncomponents"]

        if params["grouping_preprocessor"] == "drop":
            pass
        elif params["grouping_preprocessor"] in ("passthrough", "onehot"):
            nparams += len(self.grouping_columns)
        elif params["grouping_preprocessor"] in ("factor", "pca"):
            assert isinstance(params["grouping_ncomponents"], int)
            nparams += params["grouping_ncomponents"]

        if params["interactions"] == "poly":
            assert isinstance(params["interactions_ncomponents"], int)
            nparams += params["interactions_ncomponents"]

        params["nparams"] = nparams
        return params

    def sample_preprocessing_model(self, params: Dict[str, Any]):
        from sklearn.pipeline import Pipeline

        from selectml.sk.compose import Aggregated
        from selectml.sk.fixes.sklearn_coltransformer import (
            MyColumnTransformer
        )

        from ..feature_selection import MAFSelector

        target = self._sample_transformed_target_model(params)

        marker = self._sample_marker_preprocessing_model(params)

        feature_selection = self._sample_feature_selection_model(params)
        dist = self._sample_dist_preprocessing_model(params)
        nonlinear = self._sample_nonlinear_preprocessing_model(params)

        def use_all_columns(X):
            return np.repeat(True, X.shape[1])

        transformers = [
            ("feature_selection", feature_selection),
            ("dist", dist),
            ("nonlinear", nonlinear),
        ]

        transformers = MyColumnTransformer([
            (k, v, use_all_columns) for k, v in transformers
        ])

        marker_pipeline = Pipeline([
            (
                "maf_filter",
                MAFSelector(threshold=params["min_maf"], ploidy=self.ploidy)
            ),
            ("marker_scaler", marker),
            ("transformers", transformers)
        ])

        grouping = self._sample_grouping_preprocessing_model(params)

        if grouping is None:
            grouping_trans = "drop"
        elif isinstance(grouping, str):
            grouping_trans = grouping
        else:
            grouping_trans = Aggregated(
                grouping,
                agg_train=True,
                agg_test=False
            )

        n_markers = len(self.marker_columns)
        n_groups = len(self.grouping_columns)
        trans = [
            (
                "features",
                MyColumnTransformer([
                    (
                        "grouping",
                        grouping_trans,
                        np.arange(0, n_groups)
                    ),
                    (
                        "markers",
                        Aggregated(
                            marker_pipeline,
                            agg_train=True,
                            agg_test=False
                        ),
                        np.arange(n_groups, n_groups + n_markers)
                    )
                ])
            )
        ]

        interactions = (
            self._sample_interactions_preprocessing_model(params)
        )

        if interactions is None:
            interactions = "drop"
        elif interactions not in ("drop", "passthrough"):
            interactions = Aggregated(interactions, agg_test=False)

        trans.append(
            (
                "interactions",
                MyColumnTransformer([
                    ("features", "passthrough", use_all_columns),
                    (
                        "interactions",
                        interactions,
                        use_all_columns
                    )
                ]),
            )
        )

        return target, Pipeline(trans)


class XGBBaseModel(OptimiseModel):

    @classmethod
    def predict(cls, model, X, grouping):
        import xgboost as xgb

        if grouping is not None:
            X = np.concatenate([X, grouping], axis=1)

        dtrain = xgb.DMatrix(X)
        return model.predict(dtrain)

    def fit(
        self,
        params: Dict[str, Any],
        X: "npt.ArrayLike",
        y: "npt.ArrayLike",
        individuals: Optional["npt.ArrayLike"] = None,
        grouping: Optional["npt.ArrayLike"] = None,
        sample_weights: Optional["npt.ArrayLike"] = None,
        **kwargs
    ):

        import xgboost as xgb
        from copy import deepcopy
        params = deepcopy(params)

        if grouping is not None:
            X = np.concatenate([X, grouping], axis=1)

        if sample_weights is None:
            dtrain = xgb.DMatrix(X, label=y)
        else:
            dtrain = xgb.DMatrix(
                X,
                label=y,
                weight=sample_weights
            )

        num_boost_round = params.pop("num_boost_round")

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            verbose_eval=True
        )

        return model


class TFBaseModel(OptimiseModel):

    def predict(self, model, X):
        X_markers, X_grouping = self._split_groups(np.array(X))
        preds = model.predict((X_markers, X_grouping))
        return preds

    def fit(
        self,
        params: Dict[str, Any],
        X: "npt.ArrayLike",
        y: "npt.ArrayLike",
        individuals: Optional["npt.ArrayLike"] = None,
        grouping: Optional["npt.ArrayLike"] = None,
        sample_weights: Optional["npt.ArrayLike"] = None,
        **kwargs
    ):
        model = self.model(params)
        X_markers, X_grouping = self._split_groups(np.array(X))
        model.fit(
            (X_markers, X_grouping),
            y,
            sample_weights=sample_weights,
            individuals=grouping
        )
        return model

    def _split_groups(
        self,
        X: "npt.ArrayLike"
    ) -> "Tuple[np.ndarray, Optional[np.ndarray]]":
        if len(self.grouping_columns) == 0:
            return np.array(X), None

        grouping_columns = slice(0, len(self.grouping_columns))
        marker_columns = slice(len(self.grouping_columns), None)

        X_ = np.array(X)

        X_grouping = X_[:, grouping_columns]
        X_markers = X_[:, marker_columns]
        assert X_.shape[1] == (X_grouping.shape[1] + X_markers.shape[1])
        return X_markers, X_grouping

    def _sample_transformed_target_params(
        self,
        trial: "optuna.Trial",
        options: List[str] = [
            "passthrough",
            "stdnorm",
            "quantile"
        ],
    ) -> Dict[str, BaseTypes]:
        params = {}

        target = trial.suggest_categorical(
            "target_transformer",
            options
        )
        params["target_transformer"] = target

        if target == "quantile":
            params["target_transformer_quantile_distribution"] = (
                trial.suggest_categorical(
                    "target_transformer_quantile_distribution",
                    ["uniform", "normal"]
                )
            )
        return params

    def _sample_transformed_target_model(self, params: Dict[str, Any]):
        from sklearn.preprocessing import StandardScaler, QuantileTransformer
        from selectml.sk.compose import Aggregated

        preprocessor = params["target_transformer"]

        if preprocessor == "stdnorm":
            g = StandardScaler()
        elif preprocessor == "quantile":
            d = params["target_transformer_quantile_distribution"]
            g = QuantileTransformer(
                output_distribution=d,
                n_quantiles=min([1000, round(params["nsamples"] / 2)])
            )
        else:
            assert preprocessor == "passthrough"
            g = None  # Unity function

        if g is None:
            return None
        else:
            return Aggregated(g, agg_train=True, agg_test=False)

    def _sample_marker_preprocessing_params(
        self,
        trial: "optuna.Trial",
        options: List[str] = [
            "drop",
            "passthrough",
            "maf",
            "onehot",
        ],
    ) -> Dict[str, BaseTypes]:
        params = {}
        preprocessor = trial.suggest_categorical(
            "marker_preprocessor",
            options
        )
        params["marker_preprocessor"] = preprocessor
        return params

    def _sample_marker_preprocessing_model(self, params: Dict[str, Any]):
        from sklearn.preprocessing import OneHotEncoder
        from selectml.sk.preprocessor import (
            MAFScaler
        )

        preprocessor = params["marker_preprocessor"]

        if preprocessor == "drop":
            g = "drop"

        elif preprocessor == "passthrough":
            g = "passthrough"

        elif preprocessor == "onehot":
            g = OneHotEncoder(
                categories="auto",
                drop=None,
                handle_unknown="ignore"
            )

        elif preprocessor == "maf":
            g = MAFScaler(ploidy=self.ploidy)

        return g

    def _sample_feature_selection_params(
        self,
        trial: "optuna.Trial",
        options: List[str] = ["drop", "passthrough", "rf", "relief"]
    ) -> Dict[str, BaseTypes]:
        params = {}

        selector = trial.suggest_categorical("feature_selector", options)
        params["feature_selector"] = selector
        nmarkers = len(self.marker_columns)

        if selector == "rf":
            params["feature_selection_rf_min_impurity_decrease"] = (
                trial.suggest_float(
                    "feature_selection_rf_min_impurity_decrease",
                    0,
                    10
                )
            )

            params["feature_selection_nfeatures"] = (
                trial.suggest_int(
                    "feature_selection_nfeatures",
                    min([100, round(nmarkers / 2)]),
                    nmarkers - 1,
                    step=min([100, round(nmarkers / 4)])
                )
            )

        elif selector == "relief":
            params["feature_selection_nfeatures"] = (
                trial.suggest_int(
                    "feature_selection_nfeatures",
                    min([100, round(nmarkers / 2)]),
                    nmarkers - 1,
                    step=min([100, round(nmarkers / 4)]),
                )
            )

        elif selector == "maf":
            params["feature_selection_maf_threshold"] = (
                trial.suggest_uniform(
                    "feature_selection_maf_threshold",
                    0.01,
                    0.49,
                )
            )

        return params

    def _sample_feature_selection_model(self, params: Dict[str, Any]):
        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import RandomForestRegressor
        from ..feature_selection import (MAFSelector, MultiSURF)

        selector = params["feature_selector"]

        wrap_sfm = False
        if selector == "drop":
            s = "drop"

        elif selector == "passthrough":
            s = "passthrough"

        elif selector == "rf":
            wrap_sfm = True
            s = RandomForestRegressor(
                criterion="mae",
                max_depth=5,
                n_estimators=500,
                max_features=0.1,
                min_samples_split=2,
                min_samples_leaf=1,
                min_impurity_decrease=(
                    params["feature_selection_rf_min_impurity_decrease"]
                ),
                bootstrap=False,
                oob_score=False,
                n_jobs=1,
                random_state=self.seed,
            )

        elif selector == "relief":
            s = MultiSURF(
                n=params["feature_selection_nfeatures"],
                nepoch=10,
                sd=1
            )
        elif selector == "maf":
            s = MAFSelector(
                threshold=params["feature_selection_maf_threshold"],
                ploidy=self.ploidy
            )

        if wrap_sfm:
            nfeatures = params["feature_selection_nfeatures"]
            s = SelectFromModel(
                estimator=s,
                prefit=False,
                max_features=nfeatures
            )

        return s

    def sample_preprocessing_params(
        self,
        trial: "optuna.Trial",
        target_options: List[str] = [
            "passthrough",
            "stdnorm",
            "quantile",
        ],
        marker_options: List[str] = [
            "maf",
            "passthrough",
            "onehot",
        ],
        feature_selection_options: List[str] = [
            "drop",
            "passthrough",
            "rf",
        ],
    ) -> Dict[str, BaseTypes]:
        params = {}
        params.update(
            self._sample_transformed_target_params(trial, target_options)
        )
        params.update(self._sample_marker_preprocessing_params(
            trial,
            marker_options
        ))
        params.update(self._sample_feature_selection_params(
            trial,
            feature_selection_options
        ))

        return params

    def sample_preprocessing_model(self, params: Dict[str, Any]):
        from sklearn.pipeline import Pipeline

        from selectml.sk.compose import Aggregated
        from sklearn.preprocessing import StandardScaler

        from ..feature_selection import MAFSelector

        target = self._sample_transformed_target_model(params)
        marker = self._sample_marker_preprocessing_model(params)
        feature_selection = self._sample_feature_selection_model(params)

        def use_all_columns(X):
            return np.repeat(True, X.shape[1])

        marker_trans = Aggregated(
            Pipeline([
                (
                    "maf_filter",
                    MAFSelector(
                        threshold=params["min_maf"],
                        ploidy=self.ploidy
                    )
                ),
                ("marker_scaler", marker),
                ("feature_selection", feature_selection),
            ]),
            agg_train=True,
            agg_test=False,
            xaggregator="first"
        )

        grouping_trans = StandardScaler()

        return target, grouping_trans, marker_trans
