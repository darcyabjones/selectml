#!/usr/bin/env python3


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Any, Union
    from typing import Dict, List
    import optuna


BaseTypes = Union[None, bool, str, int, float]


class OptimiseBase(object):

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, BaseTypes]:
        raise NotImplementedError()

    def model(self, params: "Dict[str, Any]"):
        raise NotImplementedError()

    def starting_points(
        self
    ) -> List[Dict[str, BaseTypes]]:
        raise NotImplementedError()


class OptimiseRegressionTarget(OptimiseBase):

    def __init__(
        self,
        options: "List[str]" = [
            "passthrough",
            "stdnorm",
            "quantile",
        ],
        name: str = "regression_target",
    ):
        self.options = options
        self.name = name
        return

    def sample_params(self, trial: "optuna.Trial"):
        params = {}
        target = trial.suggest_categorical(
            f"{self.name}_transformer",
            self.options
        )
        params[f"{self.name}_transformer"] = target

        if target == "quantile":
            params[f"{self.name}_transformer_quantile_distribution"] = (
                trial.suggest_categorical(
                    f"{self.name}_transformer_quantile_distribution",
                    ["uniform", "normal"]
                )
            )
        return params

    def model(self, params: "Dict[str, Any]"):
        from sklearn.preprocessing import StandardScaler, QuantileTransformer
        from .preprocessing import Unity

        preprocessor = params[f"{self.name}_transformer"]

        if preprocessor == "stdnorm":
            g = StandardScaler()
        elif preprocessor == "quantile":
            d = params[f"{self.name}_transformer_quantile_distribution"]
            g = QuantileTransformer(
                output_distribution=d,
                n_quantiles=min([1000, round(params["nsamples"] / 2)])
            )
        elif preprocessor == "passthrough":
            g = Unity()  # Unity function
        else:
            raise ValueError(f"Got unexpected preprocessor {preprocessor}")

        return g


class OptimiseFeatureSelector(OptimiseBase):

    def __init__(
        self,
        nmarkers: int,
        options: "List[str]" = [
            "passthrough",
            "relief",
            "gemma"
        ],
        name: str = "feature_selector",
        seed=None
    ):
        self.nmarkers = nmarkers
        self.options = options
        self.cache: "Dict[str, Any]" = {}
        self.seed = seed
        return

    def sample_params(self, trial: "optuna.Trial"):
        params = {}
        selector = trial.suggest_categorical(
            f"{self.name}_feature_selector",
            self.feature_selectors
        )

        params[f"{self.name}_feature_selector"] = selector

        if selector != "passthrough":
            params[f"{self.name}_feature_selector_nfeatures"] = (
                trial.suggest_int(
                    f"{self.name}_feature_selector_nfeatures",
                    min([100, round(self.nmarkers / 2)]),
                    self.nmarkers - 1,
                )
            )

        return params

    def model(self, params: "Dict[str, Any]"):
        from .feature_selection import (
            MultiSURF,
            GEMMASelector,
        )
        from copy import deepcopy

        selector = params[f"{self.name}_feature_selector"]
        if selector == "passthrough":
            return None

        if selector in self.cache:
            s = deepcopy(self.cache[selector])
            s.n = params[f"{self.name}_feature_selector_nfeatures"]

        elif selector == "relief":
            s = MultiSURF(
                n=params[f"{self.name}_feature_selector_nfeatures"],
                nepoch=10,
                sd=1,
                random_state=self.seed
            )

        elif selector == "gemma":
            s = GEMMASelector(
                n=params[f"{self.name}_feature_selector_nfeatures"],
            )

        elif selector != "passthrough":
            raise ValueError(f"Got unexpected feature selector {selector}")

        return s


class OptimiseMarkerTransformer(OptimiseBase):

    def __init__(
        self,
        ploidy,
        max_ncomponents,
        options: "List[str]" = [
            "drop",
            "passthrough",
            "maf",
            "onehot",
            "noia_add"
            "pca"
        ],
        name: str = "marker"
    ):
        if (ploidy != 2) and ("noia_add" in options):
            raise ValueError("noia_add models are only valid for ploidy==2")

        self.ploidy = ploidy

        # Should be min([nsamples, nmarkers])
        self.max_ncomponents = max_ncomponents

        if max_ncomponents < 10:
            options = [o for o in options if o != "pca"]

        self.options = options
        self.name = name
        return

    def sample_params(self, trial: "optuna.Trial"):
        params = {}
        preprocessor = trial.suggest_categorical(
            f"{self.name}_transformer",
            self.options
        )
        params[f"{self.name}_transformer"] = preprocessor

        if preprocessor == "pca":
            params[f"{self.name}_pca_ncomponents"] = trial.suggest_int(
                f"{self.name}_pca_ncomponents",
                min([5, self.max_ncomponents - 1]),
                self.max_ncomponents
            )
        return params

    def model(self, params: "Dict[str, Any]"):
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import TruncatedSVD
        from .preprocessor import (
            Unity,
            MAFScaler,
            NOIAAdditiveScaler
        )

        preprocessor = params[f"{self.name}_transformer"]

        if preprocessor == "drop":
            g = None

        elif preprocessor == "passthrough":
            g = Unity()

        elif preprocessor == "onehot":
            g = OneHotEncoder(
                categories="auto",
                drop=None,
                handle_unknown="ignore"
            )

        elif preprocessor == "maf":
            g = MAFScaler(ploidy=self.ploidy)

        elif preprocessor == "noia_add":
            g = NOIAAdditiveScaler()

        elif preprocessor == "pca":
            g = Pipeline([
                ("prescaler", MAFScaler(ploidy=self.ploidy)),
                (
                    "pca",
                    TruncatedSVD(n_components=params["grouping_ncomponents"])
                )
            ])

        else:
            raise ValueError(f"Encountered invalid selection {preprocessor}")

        return g


class OptimiseSKDistTransformer(OptimiseBase):

    def __init__(
        self,
        ploidy: int,
        options: "List[str]" = [
            "drop",
            "vanraden",
            "manhattan",
            "euclidean"
        ],
        name: str = "sk_dist"
    ):
        self.ploidy = ploidy
        self.options = options
        self.cache: "Dict[str, Any]" = {}
        return

    def sample_params(self, trial: "optuna.Trial"):
        params = {}
        preprocessor = trial.suggest_categorical(
            f"{self.name}_transformer",
            self.options
        )
        params[f"{self.name}_transformer"] = preprocessor

        if preprocessor == "drop":
            return params

        return params

    def model(self, params: "Dict[str, Any]"):
        from sklearn.preprocessing import RobustScaler
        from sklearn.pipeline import Pipeline
        from .preprocessing import MAFScaler
        from .distance import (
            VanRadenSimilarity,
            ManhattanDistance,
            EuclideanDistance,
        )

        preprocessor = params[f"{self.name}_transformer"]

        if preprocessor == "drop":
            return None

        if preprocessor == "vanraden":
            p = VanRadenSimilarity(
                ploidy=self.ploidy,
                distance=True
            )

        elif preprocessor == "manhattan":
            p = ManhattanDistance()

        elif preprocessor == "euclidean":
            p = EuclideanDistance()

        else:
            raise ValueError(f"Got unexpected preprocessor {preprocessor}")

        steps = []
        if preprocessor in ["manhattan", "euclidean"]:
            steps.append(("prescaler", MAFScaler(ploidy=self.ploidy)))
        else:
            steps.append(("prescaler", "passthrough"))

        steps.append(("transformer", p))
        steps.append(("postscaler", RobustScaler()))
        return Pipeline(steps)


class OptimiseNonLinear(OptimiseBase):

    def __init__(
        self,
        nsamples: int,
        options: List[str] = [
            "drop",
            "rbf",
            "laplacian",
            "poly"
        ],
        name: str = "nonlinear",
        seed=None
    ):
        self.nsamples = nsamples
        self.options = options
        self.name = name
        self.seed = seed
        return

    def sample_params(self, trial: "optuna.Trial"):
        params = {}
        preprocessor = trial.suggest_categorical(
            f"{self.name}_transformer",
            self.options
        )
        params[f"{self.name}_transformer"] = preprocessor

        if preprocessor in ("rbf", "laplacian", "poly"):
            params[f"{self.name}_ncomponents"] = trial.suggest_categorical(
                f"{self.name}_ncomponents",
                [self.nsamples - 1]
            )

        if preprocessor == "rbf":
            params[f"{self.name}_rbf_gamma"] = trial.suggest_float(
                f"{self.name}_rbf_gamma",
                1e-15,
                0.5
            )
        elif preprocessor == "laplacian":
            params[f"{self.name}_laplacian_gamma"] = trial.suggest_float(
                f"{self.name}_laplacian_gamma",
                1e-15,
                0.5
            )
        elif preprocessor == "poly":
            params[f"{self.name}_poly_gamma"] = trial.suggest_float(
                f"{self.name}_poly_gamma",
                0.1,
                20
            )
        return params

    def model(self, params: "Dict[str, Any]"):
        from sklearn.pipeline import Pipeline
        from sklearn.kernel_approximation import Nystroem
        from sklearn.preprocessing import RobustScaler
        from .preprocessing import MAFScaler

        preprocessor = params[f"{self.name}_transformer"]

        if preprocessor == "drop":
            return None

        ncomponents = params[f"{self.name}_ncomponents"]

        if preprocessor == "rbf":
            p = Nystroem(
                kernel="rbf",
                gamma=params[f"{self.name}_rbf_gamma"],
                n_components=ncomponents,
                random_state=self.seed,
            )
        elif preprocessor == "laplacian":
            p = Nystroem(
                kernel="laplacian",
                gamma=params[f"{self.name}_laplacian_gamma"],
                n_components=ncomponents,
                random_state=self.seed,
            )
        elif preprocessor == "poly":
            p = Nystroem(
                kernel="poly",
                gamma=params[f"{self.name}_poly_gamma"],
                n_components=ncomponents,
                random_state=self.seed,
                degree=2
            )
        else:
            raise ValueError(f"Got invalid transformer {preprocessor}.")

        return Pipeline([
            ("prescaler", MAFScaler(ploidy=self.ploidy))
            ("nonlinear", p),
            ("scaler", RobustScaler())
        ])


class OptimiseGrouping(OptimiseBase):

    def __init__(
        self,
        max_ncomponents,
        options: "List[str]" = [
            "drop",
            "passthrough",
            "onehot",
            "pca"
        ],
        name: str = "grouping"
    ):
        self.max_ncomponents = max_ncomponents

        if self.max_ncomponents < 5:
            options = [o for o in options if o not in ("pca", "factor")]

        self.options = options
        self.name = name
        return

    def sample_params(self, trial: "optuna.Trial"):
        params = {}

        preprocessor = trial.suggest_categorical(
            f"{self.name}_transformer",
            self.options
        )

        params[f"{self.name}_transformer"] = preprocessor

        if preprocessor in ("factor", "pca"):
            params[f"{self.name}_{preprocessor}_ncomponents"] = trial.suggest_int(  # noqa
                f"{self.name}_{preprocessor}_ncomponents",
                3,
                self.max_ncomponents - 1
            )
        return params

    def model(self, params: "Dict[str, Any]"):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.decomposition import FactorAnalysis, TruncatedSVD
        from .preprocessing import Unity

        preprocessor = params[f"{self.name}_transformer"]

        if preprocessor == "drop":
            return None

        elif preprocessor == "passthrough":
            g = Unity()

        elif preprocessor == "onehot":
            g = OneHotEncoder(
                categories="auto",
                drop="if_binary",
                handle_unknown="ignore",
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
                    FactorAnalysis(
                        n_components=params[f"{self.name}_{preprocessor}_ncomponents"]  # noqa
                    )
                )
            ])
        elif preprocessor == "pca":
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
                    "pca",
                    TruncatedSVD(
                        n_components=params[f"{self.name}_{preprocessor}_ncomponents"]  # noqa
                    )
                )
            ])

        return g


class OptimiseInteractions(OptimiseBase):

    def __init__(
        self,
        options: List[str] = ["drop", "poly"],
        name: str = "interactions"
    ):
        self.options = options
        self.name = name
        return

    def sample_params(self, trial: "optuna.Trial"):
        return

    def model(self, params: "Dict[str, Any]"):
        raise NotImplementedError()


class OptimiseModel(OptimiseBase):

    def __init__(self):
        return

    def sample_params(self, trial: "optuna.Trial"):
        raise NotImplementedError()

    def model(self, params: "Dict[str, Any]"):
        raise NotImplementedError()
