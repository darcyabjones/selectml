#!/usr/bin/env python3

import random
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Optional
    from typing import Dict, List, Literal, Sequence
    from typing import Iterable
    from typing import Tuple
    import optuna
    import numpy.typing as npt
    BaseTypes = Union[None, bool, str, int, float]
    Params = Dict[str, BaseTypes]
    from baikal import Model
    ODatasetIn = Union[npt.ArrayLike, Sequence[npt.ArrayLike], None]
    ODatasetOut = Union[np.ndarray, List[np.ndarray], None]
    FitModel = Optional[Tuple[Model, ODatasetOut]]


def ndistinct(x, keep_all: bool = False):
    """ Roughly finds out how many features I expect a one hot encoded
    copy of the matrix to have.
    """
    if keep_all:
        return np.apply_along_axis(
            lambda y: max([np.unique(y).shape[0] - 1, 1]),
            0,
            x
        )
    else:
        return np.apply_along_axis(
            lambda y: np.unique(y).shape[0] - 1,
            0,
            x
        )


class OptimiseBase(object):

    def sample(
        self,
        trial: "optuna.Trial",
        Xs: "Iterable[Optional[npt.ArrayLike]]",
        **kwargs
    ) -> "Params":

        first = True

        if any([X is None for X in Xs]):
            nfeatures = 0
            nsamples = 0
            onehot_nfeatures = 0

        else:
            for X in Xs:
                X = np.array(X)
                this_nsamples = X.shape[0]
                this_nfeatures = X.shape[1]
                this_onehot_nfeatures = np.sum(ndistinct(X))

                if first:
                    first = False
                    nsamples = this_nsamples
                    nfeatures = this_nfeatures
                    onehot_nfeatures = this_onehot_nfeatures
                else:
                    nsamples = min([nsamples, this_nsamples])
                    nfeatures = min([nfeatures, this_nfeatures])
                    onehot_nfeatures = min([
                        onehot_nfeatures,
                        this_onehot_nfeatures
                    ])

        params = self.sample_params(
            trial,
            nsamples,
            nfeatures,
            onehot_nfeatures=onehot_nfeatures,
            **kwargs
        )
        return params

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        raise NotImplementedError()

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        """ This type def is mostly just for making sure it's
        clear that output should handle None.
        """
        raise NotImplementedError()

    def fit(
        self,
        params: "Params",
        Xs: "ODatasetIn",
        y: "Optional[npt.ArrayLike]" = None,
        **kwargs,
    ) -> "Optional[Model]":
        # Example implementation
        model = self.model(params)

        if (model is None) or (Xs is None):
            return None

        assert isinstance(Xs, np.ndarray)

        if y is None:
            model.fit(Xs)
        else:
            model.fit(Xs, y)

        return model

    def predict(
        self,
        model: "Model",
        Xs: "ODatasetIn",
    ) -> "ODatasetOut":
        if (model is None) or (Xs is None):
            return None

        assert isinstance(Xs, np.ndarray)
        return model(Xs)

    def fit_predict(
        self,
        params: "Params",
        Xs: "ODatasetIn",
        y: "Optional[npt.ArrayLike]" = None,
        **kwargs,
    ) -> "FitModel":
        model = self.model(params)
        if (model is None) or (Xs is None):
            return None

        model.fit(Xs, y)
        preds = model(Xs)
        return model, preds

    def starting_points(
        self
    ) -> "List[Params]":
        return []


class OptimiseTarget(OptimiseBase):

    def __init__(
        self,
        options: "Sequence[str]" = [
            "passthrough",
            "stdnorm",
            "quantile",
            "ordinal"
        ],
        ordinal_boundaries: "Union[Literal['auto'], Sequence[npt.ArrayLike]]" = "auto",  # noqa
        name: str = "target",
    ):
        self.options = list(options)
        self.ordinal_boundaries = ordinal_boundaries
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}

        if nfeatures == 0:
            raise ValueError(
                "We need at least one target feature."
            )

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
            params[f"{self.name}_nquantiles"] = (
                trial.suggest_int(
                    f"{self.name}_nquantiles",
                    min([100, round(nsamples / 2)]),
                    min([1000, nsamples])
                )
            )

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            StandardScaler,
            QuantileTransformer,
            Unity,
            OrdinalTransformer,
        )

        preprocessor = params.get(f"{self.name}_transformer", "drop")

        if preprocessor == "drop":
            return None
        elif preprocessor == "stdnorm":
            g = StandardScaler()
        elif preprocessor == "quantile":
            d = params[f"{self.name}_transformer_quantile_distribution"]
            g = QuantileTransformer(
                output_distribution=d,
                n_quantiles=params[f"{self.name}_nquantiles"]
            )
        elif preprocessor == "ordinal":
            g = OrdinalTransformer(boundaries=self.ordinal_boundaries)
        elif preprocessor == "passthrough":
            g = Unity()  # Unity function
        else:
            raise ValueError(f"Got unexpected preprocessor {preprocessor}")

        return g


class OptimiseCovariates(OptimiseBase):

    def __init__(
        self,
        options: "Sequence[str]" = [
            "passthrough",
            "stdnorm",
            "robust",
            "quantile",
            "power",
        ],
        name: str = "covariate",
    ):
        self.options = list(options)
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}

        if nfeatures == 0:
            options = ["drop"]
        else:
            options = self.options

        target = trial.suggest_categorical(
            f"{self.name}_transformer",
            options
        )
        params[f"{self.name}_transformer"] = target

        if target == "quantile":
            params[f"{self.name}_transformer_quantile_distribution"] = (
                trial.suggest_categorical(
                    f"{self.name}_transformer_quantile_distribution",
                    ["uniform", "normal"]
                )
            )
            params[f"{self.name}_nquantiles"] = (
                trial.suggest_int(
                    f"{self.name}_nquantiles",
                    min([100, round(nsamples / 2)]),
                    min([1000, nsamples])
                )
            )

        if target != "drop":
            p = trial.suggest_categorical(  # noqa
                f"{self.name}_polynomial_degree",
                [1, 2, 3]
            )
            assert isinstance(p, int)
            params[f"{self.name}_polynomial_degree"] = p

            if p > 1:
                params[f"{self.name}_polynomial_interaction_only"] = trial.suggest_categorical(  # noqa
                    f"{self.name}_polynomial_interaction_only",
                    [True, False]
                )

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            Pipeline,
            StandardScaler,
            RobustScaler,
            QuantileTransformer,
            PowerTransformer,
            PolynomialFeatures,
            Unity,
        )

        preprocessor = params.get(f"{self.name}_transformer", "drop")

        if preprocessor == "stdnorm":
            g = StandardScaler()
        elif preprocessor == "quantile":
            d = params[f"{self.name}_transformer_quantile_distribution"]
            g = QuantileTransformer(
                output_distribution=d,
                n_quantiles=params[f"{self.name}_nquantiles"]
            )
        elif preprocessor == "robust":
            g = RobustScaler()
        elif preprocessor == "power":
            g = Pipeline([
                ("scale", RobustScaler()),
                ("power", PowerTransformer())
            ])
        elif preprocessor == "passthrough":
            g = Unity()  # Unity function
        elif preprocessor == "drop":
            g = None
        else:
            raise ValueError(f"Got unexpected preprocessor {preprocessor}")

        if f"{self.name}_polynomial_degree" in params:
            p = params[f"{self.name}_polynomial_degree"]
            assert isinstance(p, int)
            if p > 1:
                g = Pipeline([
                    ("scale", g),
                    ("power", PolynomialFeatures(
                        degree=p,
                        interaction_only=params[f"{self.name}_polynomial_interaction_only"],  # noqa
                        include_bias=False
                    )),
                ])

        return g


class OptimiseFeatureSelector(OptimiseBase):

    def __init__(
        self,
        options: "Sequence[str]" = [
            "passthrough",
            "drop",
            "maf",
            "relief",
            "gemma"
        ],
        name: str = "feature_selector",
        gemma_exe: str = "gemma",
        seed=None
    ):
        from threading import Lock

        if not self._check_if_has_gemma(gemma_exe):
            options = [o for o in options if o != "gemma"]

        self.options = list(options)
        self.rng = random.Random(seed)
        self.name = name

        self.lock = Lock()
        # key is [id(data), option]
        self.cache: "Dict[Tuple[int, str], Model]" = dict()
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}

        if nfeatures == 0:
            raise ValueError(
                "We need at least one marker feature."
            )

        selector = trial.suggest_categorical(
            f"{self.name}_selector",
            self.options
        )

        params[f"{self.name}_selector"] = selector

        if selector != "passthrough":
            params[f"{self.name}_nfeatures"] = (
                trial.suggest_int(
                    f"{self.name}_nfeatures",
                    min([100, round(nfeatures / 2)]),
                    nfeatures - 1,
                )
            )

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            MultiSURF,
            GEMMASelector,
            MAFSelector,
            Unity,
        )

        selector = params.get(f"{self.name}_selector", "drop")
        if selector == "drop":
            return None

        elif selector == "passthrough":
            s = Unity()

        elif selector == "relief":
            n = params[f"{self.name}_nfeatures"]
            assert isinstance(n, int)
            s = MultiSURF(
                n=n,
                nepoch=10,
                sd=1,
                random_state=self.rng.getrandbits(32)
            )

        elif selector == "gemma":
            n = params[f"{self.name}_nfeatures"]
            assert isinstance(n, int)
            s = GEMMASelector(
                n=n,
            )

        elif selector == "maf":
            n = params[f"{self.name}_nfeatures"]
            assert isinstance(n, int)
            s = MAFSelector(
                n=n,
            )

        else:
            raise ValueError(f"Got unexpected feature selector {selector}")

        return s

    @staticmethod
    def _check_if_has_gemma(exe: str):
        from shutil import which
        return which(exe) is not None

    def fit(
        self,
        params: "Params",
        Xs: "ODatasetIn",
        y: "Optional[npt.ArrayLike]" = None,
        **kwargs,
    ) -> "Optional[Model]":
        # Example implementation
        from copy import deepcopy
        from .wrapper import (
            MultiSURF,
            GEMMASelector,
            MAFSelector
        )

        assert y is not None

        selector = params[f"{self.name}_selector"]

        assert isinstance(Xs, np.ndarray)

        assert isinstance(selector, str)
        key = (id(Xs), selector)

        """ This isn't a great solution, it repeats some work from the
        model selection bit. Would be good to lookup cache in there,
        but we need the data to do the lookup sooo.
        """
        with self.lock:
            if key in self.cache:
                model = deepcopy(self.cache[key])
                if model is None:
                    return None
                elif isinstance(
                    model,
                    (MultiSURF, GEMMASelector, MAFSelector)
                ):
                    n = params[f"{self.name}_nfeatures"]
                    assert isinstance(n, int)
                    model.n = n
            else:
                model_ = self.model(params)
                if model_ is None:
                    return None
                else:
                    model = model_

                model.fit(Xs, y)
                self.cache[key] = model

        return model


class OptimiseMarkerTransformer(OptimiseBase):

    def __init__(
        self,
        ploidy: int,
        max_ncomponents: int,
        options: "Sequence[str]" = [
            "passthrough",
            "maf",
            "onehot",
            "noia_add",
            "pca"
        ],
        name: str = "marker",
        seed: "Optional[int]" = None,
    ):
        if (ploidy != 2) and ("noia_add" in options):
            raise ValueError("noia_add models are only valid for ploidy==2")

        self.ploidy = ploidy
        self.max_ncomponents = max_ncomponents

        self.options = list(options)
        self.name = name
        self.rng = random.Random(seed)
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}
        preprocessor = trial.suggest_categorical(
            f"{self.name}_transformer",
            self.options
        )
        params[f"{self.name}_transformer"] = preprocessor

        max_ncomponents = min([
            nsamples - 1,
            nfeatures - 1,
            self.max_ncomponents
        ])
        min_ncomponents = min([
            5,
            max_ncomponents,
            round(max_ncomponents / 2)
        ])

        if preprocessor == "pca":
            params[f"{self.name}_pca_ncomponents"] = trial.suggest_int(
                f"{self.name}_pca_ncomponents",
                min_ncomponents,
                max_ncomponents
            )

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            OneHotEncoder,
            Pipeline,
            TruncatedSVD,
            Unity,
            MAFScaler,
            NOIAAdditiveScaler,
        )

        preprocessor = params.get(f"{self.name}_transformer", "drop")

        if preprocessor == "drop":
            return None

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
                    TruncatedSVD(
                        n_components=params[f"{self.name}_pca_ncomponents"],
                        random_state=self.rng.getrandbits(32)
                    )
                )
            ])

        else:
            raise ValueError(f"Encountered invalid selection {preprocessor}")

        return g


class OptimiseDistTransformer(OptimiseBase):

    def __init__(
        self,
        ploidy: int,
        options: "Sequence[str]" = [
            "vanraden",
            "manhattan",
            "euclidean"
        ],
        name: str = "dist"
    ):
        self.ploidy = ploidy
        self.options = list(options)
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}
        preprocessor = trial.suggest_categorical(
            f"{self.name}_transformer",
            self.options
        )
        params[f"{self.name}_transformer"] = preprocessor

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            RobustScaler,
            Pipeline,
            MAFScaler,
            VanRadenSimilarity,
            ManhattanDistance,
            EuclideanDistance,
        )

        preprocessor = params.get(f"{self.name}_transformer", "drop")

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

        steps.append(("transformer", p))
        steps.append(("postscaler", RobustScaler()))
        return Pipeline(steps)


class OptimiseNonLinear(OptimiseBase):

    def __init__(
        self,
        ploidy: int,
        max_ncomponents: int,
        options: "List[str]" = [
            "rbf",
            "laplacian",
            "poly"
        ],
        name: str = "nonlinear",
        seed: "Optional[int]" = None
    ):
        self.ploidy = ploidy
        self.max_ncomponents = max_ncomponents
        self.options = options
        self.name = name
        self.rng = random.Random(seed)
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}
        preprocessor = trial.suggest_categorical(
            f"{self.name}_transformer",
            self.options
        )
        params[f"{self.name}_transformer"] = preprocessor

        if preprocessor in ("rbf", "laplacian", "poly"):
            max_ncomponents = min([
                nsamples - 1,
                nfeatures - 1,
                self.max_ncomponents
            ])

            min_ncomponents = min([
                5,
                max_ncomponents,
                round(max_ncomponents / 2)
            ])

            params[f"{self.name}_ncomponents"] = trial.suggest_int(
                f"{self.name}_ncomponents",
                min_ncomponents,
                max_ncomponents
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

        params[f"{self.name}_nquantiles"] = (
            trial.suggest_int(
                f"{self.name}_nquantiles",
                min([100, round(nsamples / 2)]),
                min([1000, nsamples])
            )
        )

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            Pipeline,
            Nystroem,
            QuantileTransformer,
            MAFScaler,
        )

        preprocessor = params.get(f"{self.name}_transformer", "drop")

        if preprocessor == "drop":
            return None

        ncomponents = params[f"{self.name}_ncomponents"]

        if preprocessor == "rbf":
            p = Nystroem(
                kernel="rbf",
                gamma=params[f"{self.name}_rbf_gamma"],
                n_components=ncomponents,
                random_state=self.rng.getrandbits(32),
            )
        elif preprocessor == "laplacian":
            p = Nystroem(
                kernel="laplacian",
                gamma=params[f"{self.name}_laplacian_gamma"],
                n_components=ncomponents,
                random_state=self.rng.getrandbits(32),
            )
        elif preprocessor == "poly":
            p = Nystroem(
                kernel="poly",
                gamma=params[f"{self.name}_poly_gamma"],
                n_components=ncomponents,
                random_state=self.rng.getrandbits(32),
                degree=2
            )
        else:
            raise ValueError(f"Got invalid transformer {preprocessor}.")

        return Pipeline([
            ("prescaler", MAFScaler(ploidy=self.ploidy)),
            ("nonlinear", p),
            ("scaler", QuantileTransformer(
                n_quantiles=params[f"{self.name}_nquantiles"]
            ))
        ])


class OptimiseGrouping(OptimiseBase):

    def __init__(
        self,
        max_ncomponents,
        options: "List[str]" = [
            "passthrough",
            "onehot",
            "pca"
        ],
        name: str = "grouping",
        seed: "Optional[int]" = None,
    ):
        self.max_ncomponents = max_ncomponents
        self.options = options
        self.name = name
        self.rng = random.Random(seed)
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}

        if nfeatures == 0:
            options = ["drop"]
        else:
            options = self.options

        preprocessor = trial.suggest_categorical(
            f"{self.name}_transformer",
            options
        )

        params[f"{self.name}_transformer"] = preprocessor

        if preprocessor in ("factor", "pca"):
            max_ncomponents = min([
                nsamples - 1,
                nfeatures - 1,
                self.max_ncomponents
            ])

            min_ncomponents = min([
                3,
                max_ncomponents,
                round(max_ncomponents / 2)
            ])

            params[f"{self.name}_{preprocessor}_ncomponents"] = trial.suggest_int(  # noqa
                f"{self.name}_{preprocessor}_ncomponents",
                min_ncomponents,
                max_ncomponents
            )

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            Pipeline,
            OneHotEncoder,
            FactorAnalysis,
            TruncatedSVD,
            Unity,
        )

        preprocessor = params.get(f"{self.name}_transformer", "drop")

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
                        n_components=params[f"{self.name}_{preprocessor}_ncomponents"],  # noqa
                        random_state=self.rng.getrandbits(32)
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
                        n_components=params[f"{self.name}_{preprocessor}_ncomponents"],  # noqa
                        random_state=self.rng.getrandbits(32)
                    )
                )
            ])

        return g


class OptimiseInteractions(OptimiseBase):

    def __init__(
        self,
        max_ncomponents: int,
        options: "List[str]" = ["drop", "rbf", "laplacian", "poly"],
        name: str = "interactions",
        seed: "Optional[int]" = None,
    ):
        self.max_ncomponents = max_ncomponents
        self.options = options
        self.name = name

        self.rng = random.Random(seed)
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}
        preprocessor = trial.suggest_categorical(
            f"{self.name}_preprocessor",
            self.options
        )
        params[f"{self.name}_preprocessor"] = preprocessor

        if preprocessor in ("rbf", "laplacian", "poly"):
            max_ncomponents = min([
                nsamples - 1,
                nfeatures - 1,
                self.max_ncomponents
            ])
            params[f"{self.name}_ncomponents"] = trial.suggest_int(
                f"{self.name}_ncomponents",
                min([5, round(max_ncomponents / 2)]),
                max_ncomponents
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

        params[f"{self.name}_nquantiles"] = (
            trial.suggest_int(
                f"{self.name}_nquantiles",
                min([100, round(nsamples / 2)]),
                min([1000, nsamples])
            )
        )

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            Nystroem,
            QuantileTransformer,
            Pipeline,
        )

        preprocessor = params.get(f"{self.name}_preprocessor", "drop")

        if preprocessor == "drop":
            return None

        ncomponents = params[f"{self.name}_ncomponents"]
        if preprocessor == "rbf":
            p = Nystroem(
                kernel="rbf",
                gamma=params[f"{self.name}_rbf_gamma"],
                n_components=ncomponents,
                random_state=self.rng.getrandbits(32),
            )

        elif preprocessor == "laplacian":
            p = Nystroem(
                kernel="laplacian",
                gamma=params[f"{self.name}_laplacian_gamma"],
                n_components=ncomponents,
                random_state=self.rng.getrandbits(32),
            )

        elif preprocessor == "poly":
            p = Nystroem(
                kernel="poly",
                gamma=params[f"{self.name}_poly_gamma"],
                n_components=ncomponents,
                random_state=self.rng.getrandbits(32),
                degree=2,
            )
        else:
            raise ValueError("Invalid preprocessor")

        return Pipeline([
            ("nonlinear", p),
            ("scaler", QuantileTransformer(
                n_quantiles=params[f"{self.name}_nquantiles"]
            ))
        ])


class OptimiseSK(OptimiseBase):

    def fit(
        self,
        params: "Params",
        Xs: "ODatasetIn",
        y: "Optional[npt.ArrayLike]" = None,
        **kwargs,
    ) -> "Optional[Model]":
        if isinstance(Xs, np.ndarray):
            X = np.asarray(Xs)
        elif isinstance(Xs, list):
            assert len(Xs) == 1
            X = np.asarray(Xs[0])
        else:
            raise ValueError("Invalid data")

        model = self.model(params)

        if model is None:
            return None

        model.fit(X, y)
        return model


class OptimiseXGB(OptimiseSK):

    def __init__(
        self,
        objectives: "Sequence[str]" = [
            "reg:squarederror",
            "reg:logistic",
            "binary:logistic",
            "count:poisson",
            "rank:pairwise",
            "reg:gamma",
            "reg:tweedie"
        ],
        seed: "Optional[int]" = None,
        name: str = "xgb",
    ):
        self.objectives = objectives
        self.rng = random.Random(seed)
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}
        params.update({
            f"{self.name}_objective": trial.suggest_categorical(
                f"{self.name}_objective",
                self.objectives
            ),
            f"{self.name}_n_estimators": trial.suggest_int(
                f"{self.name}_n_estimators",
                10,
                1000
            ),
            f"{self.name}_booster": trial.suggest_categorical(
                f"{self.name}_booster",
                ["gbtree", "gblinear", "dart"]
            ),
            f"{self.name}_gamma": trial.suggest_float(
                f"{self.name}_gamma",
                0,
                100
            ),
            f"{self.name}_min_child_weight": trial.suggest_int(
                f"{self.name}_min_child_weight",
                1,
                20
            ),
            f"{self.name}_subsample": trial.suggest_float(
                f"{self.name}_subsample",
                0.1,
                1
            ),
            f"{self.name}_colsample_bytree": trial.suggest_float(
                f"{self.name}_colsample_bytree",
                0.1,
                1
            ),
            f"{self.name}_colsample_bylevel": trial.suggest_float(
                f"{self.name}_colsample_bylevel",
                0.1,
                1
            ),
            f"{self.name}_colsample_bynode": trial.suggest_float(
                f"{self.name}_colsample_bynode",
                0.1,
                1
            ),
            f"{self.name}_reg_alpha": trial.suggest_float(
                f"{self.name}_reg_alpha",
                0,
                50
            ),
            f"{self.name}_reg_lambda": trial.suggest_float(
                f"{self.name}_reg_lambda",
                0,
                50
            ),
            f"{self.name}_max_depth": trial.suggest_int(
                f"{self.name}_max_depth",
                3,
                10
            ),
            f"{self.name}_learning_rate": trial.suggest_float(
                f"{self.name}_learning_rate",
                1e-4,
                0.5,
                log=True
            ),
        })
        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            XGBRanker,
            XGBRegressor,
            XGBClassifier,
        )

        objective = params[f"{self.name}_objective"]
        assert isinstance(objective, str)
        if objective.startswith("reg") or objective.startswith("count"):
            cls = XGBRegressor
        elif objective.startswith("binary") or objective.startswith("multi"):
            cls = XGBClassifier
        elif objective.startswith("rank"):
            cls = XGBRanker
        else:
            raise ValueError("This shouldn't be reachable")

        model = cls(
            objective=objective,
            n_estimators=params[f"{self.name}_n_estimators"],
            booster=params[f"{self.name}_booster"],
            gamma=params[f"{self.name}_gamma"],
            min_child_weight=params[f"{self.name}_min_child_weight"],
            subsample=params[f"{self.name}_subsample"],
            colsample_bytree=params[f"{self.name}_colsample_bytree"],
            colsample_bylevel=params[f"{self.name}_colsample_bylevel"],
            colsample_bynode=params[f"{self.name}_colsample_bynode"],
            reg_alpha=params[f"{self.name}_reg_alpha"],
            reg_lambda=params[f"{self.name}_reg_lambda"],
            random_state=self.rng.getrandbits(32),
            max_depth=params[f"{self.name}_max_depth"],
            learning_rate=params[f"{self.name}_learning_rate"],
            n_jobs=1,
            verbosity=1,
        )

        return model

    def starting_points(self) -> "List[Params]":
        out: "List[Params]" = []
        for objective in self.objectives:
            out.extend([
                {
                    f"{self.name}_objective": objective,
                    f"{self.name}_n_estimators": 500,
                    f"{self.name}_booster": "gbtree",
                    f"{self.name}_gamma": 10,
                    f"{self.name}_min_child_weight": 1,
                    f"{self.name}_subsample": 1,
                    f"{self.name}_colsample_bytree": 1,
                    f"{self.name}_colsample_bylevel": 1,
                    f"{self.name}_colsample_bynode": 1,
                    f"{self.name}_reg_alpha": 1,
                    f"{self.name}_reg_lambda": 1,
                    f"{self.name}_max_depth": 4,
                    f"{self.name}_learning_rate": 1e-3,
                },
                {
                    f"{self.name}_objective": objective,
                    f"{self.name}_n_estimators": 500,
                    f"{self.name}_booster": "gbtree",
                    f"{self.name}_gamma": 10,
                    f"{self.name}_min_child_weight": 1,
                    f"{self.name}_subsample": 1,
                    f"{self.name}_colsample_bytree": 1,
                    f"{self.name}_colsample_bylevel": 1,
                    f"{self.name}_colsample_bynode": 1,
                    f"{self.name}_reg_alpha": 1,
                    f"{self.name}_reg_lambda": 1,
                    f"{self.name}_max_depth": 9,
                    f"{self.name}_learning_rate": 1e-3,
                }
            ])
        return out


class OptimiseKNN(OptimiseSK):

    def __init__(
        self,
        seed: "Optional[int]" = None,
        name: str = "knn",
    ):
        self.rng = random.Random(seed)
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}
        params.update({
            f"{self.name}_n_neighbors": trial.suggest_int(
                f"{self.name}_n_neighbors",
                2,
                100
            ),
            f"{self.name}_weights": trial.suggest_categorical(
                f"{self.name}_weights",
                ["distance", "uniform"]
            ),
            f"{self.name}_leaf_size": trial.suggest_int(
                f"{self.name}_leaf_size",
                10,
                80
            ),
            f"{self.name}_algorithm": trial.suggest_categorical(
                f"{self.name}_algorithm",
                ["kd_tree", "ball_tree"]
            ),
            f"{self.name}_p": trial.suggest_categorical(
                f"{self.name}_p",
                [1, 2]
            ),
        })
        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import KNeighborsRegressor

        model = KNeighborsRegressor(
            n_neighbors=params[f"{self.name}_n_neighbors"],
            weights=params[f"{self.name}_weights"],
            leaf_size=params[f"{self.name}_leaf_size"],
            algorithm=params[f"{self.name}_algorithm"],
            p=params[f"{self.name}_p"],
            n_jobs=-1
        )
        return model

    def starting_points(self) -> "List[Params]":
        out: "List[Params]" = []
        out.extend([
            {
                f"{self.name}_n_neighbors": 10,
                f"{self.name}_weights": "distance",
                f"{self.name}_leaf_size": 10,
                f"{self.name}_algorithm": "kd_tree",
                f"{self.name}_p": 1,
            },
            {
                f"{self.name}_n_neighbors": 10,
                f"{self.name}_weights": "distance",
                f"{self.name}_leaf_size": 50,
                f"{self.name}_algorithm": "kd_tree",
                f"{self.name}_p": 1,
            },
        ])
        return out


class OptimiseRF(OptimiseSK):

    def __init__(
        self,
        criterion: "Sequence[str]" = [
            "squared_error",
            "absolute_error",
            "poisson",
            "gini",
            "entropy",
        ],
        seed: "Optional[int]" = None,
        name: str = "rf",
    ):
        self.criterion = criterion
        self.rng = random.Random(seed)
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}
        params.update({
            f"{self.name}_criterion": trial.suggest_categorical(
                f"{self.name}_criterion",
                self.criterion
            ),
            f"{self.name}_max_features": trial.suggest_float(
                f"{self.name}_max_features",
                0.01,
                1.0
            ),
            f"{self.name}_max_samples": trial.suggest_float(
                f"{self.name}_max_samples",
                0.01,
                1.0
            ),
            f"{self.name}_bootstrap": trial.suggest_categorical(
                f"{self.name}_bootstrap",
                [True, False]
            ),
            f"{self.name}_max_depth": trial.suggest_int(
                f"{self.name}_max_depth",
                3,
                30
            ),
            f"{self.name}_n_estimators": trial.suggest_int(
                f"{self.name}_n_estimators",
                50,
                1000
            ),
            f"{self.name}_min_samples_split": trial.suggest_int(
                f"{self.name}_min_samples_split",
                2,
                50
            ),
            f"{self.name}_min_samples_leaf": trial.suggest_int(
                f"{self.name}_min_samples_leaf",
                1,
                20
            ),
            f"{self.name}_min_impurity_decrease": trial.suggest_float(
                "{self.name}_min_impurity_decrease",
                0,
                1
            ),
        })

        if params[f"{self.name}_bootstrap"]:
            params[f"{self.name}_oob_score"] = trial.suggest_categorical(
                f"{self.name}_oob_score",
                [True, False]
            )
        else:
            params[f"{self.name}_oob_score"] = False
        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            RandomForestRegressor,
            RandomForestClassifier,
        )

        criterion = params[f"{self.name}_criterion"]
        assert isinstance(criterion, str)
        if criterion in ("squared_error", "absolute_error", "poisson"):
            cls = RandomForestRegressor
        elif criterion in ("gini", "entropy"):
            cls = RandomForestClassifier
        else:
            raise ValueError("This shouldn't be reachable")

        # Prevents key error when selecting from "best params"
        if f"{self.name}_oob_score" in params:
            oob_score = params[f"{self.name}_oob_score"]
        else:
            oob_score = False

        model = cls(
            criterion=criterion,
            max_depth=params[f"{self.name}_max_depth"],
            max_samples=params[f"{self.name}_max_samples"],
            n_estimators=params[f"{self.name}_n_estimators"],
            max_features=params[f"{self.name}_max_features"],
            min_samples_split=params[f"{self.name}_min_samples_split"],
            min_samples_leaf=params[f"{self.name}_min_samples_leaf"],
            min_impurity_decrease=params[f"{self.name}_min_impurity_decrease"],
            bootstrap=params[f"{self.name}_bootstrap"],
            oob_score=oob_score,
            n_jobs=-1,
            random_state=self.rng.getrandbits(32),
        )
        return model

    def starting_points(self) -> "List[Params]":
        out: "List[Params]" = []
        for c in self.criterion:
            out.append({
                f"{self.name}_criterion": c,
                f"{self.name}_max_depth": 3,
                f"{self.name}_max_samples": 1.0,
                f"{self.name}_n_estimators": 500,
                f"{self.name}_max_features": 0.5,
                f"{self.name}_min_samples_split": 2,
                f"{self.name}_min_samples_leaf": 10,
                f"{self.name}_min_impurity_decrease": 0.1,
                f"{self.name}_bootstrap": False,
            })
        return out


class OptimiseExtraTrees(OptimiseSK):

    def __init__(
        self,
        criterion: "Sequence[str]" = [
            "squared_error",
            "absolute_error",
            "gini",
            "entropy",
        ],
        seed: "Optional[int]" = None,
        name: str = "extratrees",
    ):
        self.criterion = criterion
        self.rng = random.Random(seed)
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":

        params = {}
        params.update({
            f"{self.name}_criterion": trial.suggest_categorical(
                f"{self.name}_criterion",
                self.criterion
            ),
            f"{self.name}_max_features": trial.suggest_float(
                f"{self.name}_max_features",
                0.01,
                1.0
            ),
            f"{self.name}_bootstrap": trial.suggest_categorical(
                f"{self.name}_bootstrap",
                [True, False]
            ),
            f"{self.name}_max_depth": trial.suggest_int(
                f"{self.name}_max_depth",
                3,
                30
            ),
            f"{self.name}_n_estimators": trial.suggest_int(
                f"{self.name}_n_estimators",
                50,
                1000
            ),
            f"{self.name}_max_samples": trial.suggest_float(
                f"{self.name}_max_samples",
                0.01,
                1.0
            ),
            f"{self.name}_min_samples_split": trial.suggest_int(
                f"{self.name}_min_samples_split",
                2,
                50
            ),
            f"{self.name}_min_samples_leaf": trial.suggest_int(
                f"{self.name}_min_samples_leaf",
                1,
                20
            ),
            f"{self.name}_min_impurity_decrease": trial.suggest_float(
                "{self.name}_min_impurity_decrease",
                0,
                1
            ),
        })

        if params[f"{self.name}_bootstrap"]:
            params[f"{self.name}_oob_score"] = trial.suggest_categorical(
                f"{self.name}_oob_score",
                [True, False]
            )
        else:
            params[f"{self.name}_oob_score"] = False
        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            ExtraTreesRegressor,
            ExtraTreesClassifier,
        )

        criterion = params[f"{self.name}_criterion"]
        assert isinstance(criterion, str)
        if criterion in ("squared_error", "absolute_error"):
            cls = ExtraTreesRegressor
        elif criterion in ("gini", "entropy"):
            cls = ExtraTreesClassifier
        else:
            raise ValueError("This shouldn't be reachable")

        # Prevents key error when selecting from "best params"
        if f"{self.name}_oob_score" in params:
            oob_score = params[f"{self.name}_oob_score"]
        else:
            oob_score = False

        model = cls(
            criterion=criterion,
            max_depth=params[f"{self.name}_max_depth"],
            max_samples=params[f"{self.name}_max_samples"],
            n_estimators=params[f"{self.name}_n_estimators"],
            max_features=params[f"{self.name}_max_features"],
            min_samples_split=params[f"{self.name}_min_samples_split"],
            min_samples_leaf=params[f"{self.name}_min_samples_leaf"],
            min_impurity_decrease=params[f"{self.name}_min_impurity_decrease"],
            bootstrap=params[f"{self.name}_bootstrap"],
            oob_score=oob_score,
            n_jobs=-1,
            random_state=self.rng.getrandbits(32),
        )
        return model

    def starting_points(self) -> "List[Params]":
        out: "List[Params]" = []
        for c in self.criterion:
            out.append({
                f"{self.name}_criterion": c,
                f"{self.name}_max_depth": 3,
                f"{self.name}_max_samples": 1.0,
                f"{self.name}_n_estimators": 500,
                f"{self.name}_max_features": 0.5,
                f"{self.name}_min_samples_split": 2,
                f"{self.name}_min_samples_leaf": 10,
                f"{self.name}_min_impurity_decrease": 0.1,
                f"{self.name}_bootstrap": False,
            })
        return out


class OptimiseNGB(OptimiseSK):

    def __init__(
        self,
        distribution: "Sequence[str]" = [
            "normal",
            "lognormal",
            "exponential",
            "bernoulli"
        ],
        seed: "Optional[int]" = None,
        name: str = "ngb",
    ):
        self.distribution = distribution
        self.rng = random.Random(seed)
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}
        params.update({
            f"{self.name}_distribution": trial.suggest_categorical(
                f"{self.name}_distribution",
                self.distribution
            ),
            f"{self.name}_max_features": trial.suggest_float(
                f"{self.name}_max_features",
                0.01,
                1.0
            ),
            f"{self.name}_n_estimators": trial.suggest_int(
                f"{self.name}_n_estimators",
                50,
                1000
            ),
            f"{self.name}_max_depth": trial.suggest_int(
                f"{self.name}_max_depth",
                3,
                30
            ),
            f"{self.name}_min_samples_split": trial.suggest_int(
                f"{self.name}_min_samples_split",
                2,
                50
            ),
            f"{self.name}_min_samples_leaf": trial.suggest_int(
                f"{self.name}_min_samples_leaf",
                1,
                20
            ),
            f"{self.name}_min_impurity_decrease": trial.suggest_float(
                "{self.name}_min_impurity_decrease",
                0,
                1
            ),
            f"{self.name}_col_sample": trial.suggest_float(
                f"{self.name}_col_sample",
                0.01,
                1.0
            ),
            f"{self.name}_learning_rate": trial.suggest_float(
                f"{self.name}_learning_rate",
                1e-6,
                1.0,
                log=True
            ),
            f"{self.name}_natural_gradient": trial.suggest_categorical(
                f"{self.name}_natural_gradient",
                [True, False]
            ),
            f"{self.name}_minibatch_frac": trial.suggest_float(
                f"{self.name}_minibatch_frac",
                0.1,
                1.0
            )
        })
        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":

        from .wrapper import (
            NGBRegressor,
            NGBClassifier
        )
        from ngboost.distns import (
            Normal,
            Exponential,
            LogNormal,
            Bernoulli
        )
        from sklearn.tree import DecisionTreeRegressor

        distribution = params[f"{self.name}_distribution"]
        assert isinstance(distribution, str)
        if distribution in ("normal", "lognormal", "exponential"):
            cls = NGBRegressor
        elif distribution in ("bernoulli"):
            cls = NGBClassifier
        else:
            raise ValueError("This shouldn't be reachable")

        dist = {
            "normal": Normal,
            "exponential": Exponential,
            "lognormal": LogNormal,
            "bernoulli": Bernoulli
        }[distribution]

        model = cls(
            Dist=dist,
            Base=DecisionTreeRegressor(
                criterion="friedman_mse",
                max_depth=params[f"{self.name}_max_depth"],
                max_features=params[f"{self.name}_max_features"],
                min_samples_split=params[f"{self.name}_min_samples_split"],
                min_samples_leaf=params[f"{self.name}_min_samples_leaf"],
                min_impurity_decrease=params[f"{self.name}_min_impurity_decrease"],  # noqa
            ),
            n_estimators=params[f"{self.name}_n_estimators"],
            col_sample=params[f"{self.name}_col_sample"],
            learning_rate=params[f"{self.name}_learning_rate"],
            natural_gradient=params[f"{self.name}_natural_gradient"],
            n_jobs=-1,
            verbose=False,
            random_state=self.rng.getrandbits(32),
        )
        return model

    def starting_points(self) -> "List[Params]":
        out: "List[Params]" = []
        for d in self.distribution:
            out.append({
                f"{self.name}_distribution": d,
                f"{self.name}_max_depth": 3,
                f"{self.name}_n_estimators": 500,
                f"{self.name}_max_features": 0.5,
                f"{self.name}_min_samples_split": 2,
                f"{self.name}_min_samples_leaf": 10,
                f"{self.name}_min_impurity_decrease": 0.1,
                f"{self.name}_col_sample": 0.5,
                f"{self.name}_learning_rate": 0.1,
                f"{self.name}_natural_gradient": True,
            })
        return out


class OptimiseSVM(OptimiseSK):

    def __init__(
        self,
        loss: "Sequence[str]" = [
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
            "hinge",
            "squared_hinge"
        ],
        seed: "Optional[int]" = None,
        name: str = "svm",
    ):
        self.loss = loss
        self.rng = random.Random(seed)
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}
        params.update({
            f"{self.name}_loss": trial.suggest_categorical(
                f"{self.name}_loss",
                self.loss,
            ),
            f"{self.name}_epsilon": trial.suggest_float(
                f"{self.name}_epsilon",
                0.0,
                5.0,
            ),
            f"{self.name}_C": trial.suggest_float(
                f"{self.name}_C",
                1e-10,
                5,
            ),
            f"{self.name}_intercept_scaling": trial.suggest_float(
                f"{self.name}_intercept_scaling",
                1e-10,
                5,
            ),
            f"{self.name}_dual": trial.suggest_categorical(
                f"{self.name}_dual",
                [True, False],
            ),
            f"{self.name}_fit_intercept": trial.suggest_categorical(
                f"{self.name}_fit_intercept",
                [True, False],
            ),
            f"{self.name}_max_iter": trial.suggest_categorical(
                f"{self.name}_max_iter",
                [nfeatures * 10],
            )
        })

        if params[f"{self.name}_loss"] in ("hinge", "squared_hinge"):
            params.update({
                f"{self.name}_penalty": trial.suggest_categorical(
                    f"{self.name}_penalty",
                    ["l1", "l2"],
                ),
                f"{self.name}_multi_class": trial.suggest_categorical(
                    f"{self.name}_multi_class",
                    ["ovr", "crammer_singer"]
                ),
                f"{self.name}_class_weight": trial.suggest_categorical(
                    f"{self.name}_class_weight",
                    [None, "balanced"]
                ),
            })

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            LinearSVR,
            LinearSVC
        )

        loss = params[f"{self.name}_loss"]
        assert isinstance(loss, str)

        if loss in ("epsilon_insensitive", "squared_epsilon_insensitive"):
            model = LinearSVR(
                fit_intercept=params[f"{self.name}_fit_intercept"],
                intercept_scaling=params[f"{self.name}_intercept_scaling"],
                max_iter=params[f"{self.name}_max_iter"],
                dual=params[f"{self.name}_dual"],
                C=params[f"{self.name}_C"],
                epsilon=params[f"{self.name}_epsilon"],
                loss=params[f"{self.name}_loss"],
                random_state=self.rng.getrandbits(32),
            )
        elif loss in ("hinge", "squared_hinge"):
            model = LinearSVC(
                fit_intercept=params[f"{self.name}_fit_intercept"],
                intercept_scaling=params[f"{self.name}_intercept_scaling"],
                max_iter=params[f"{self.name}_max_iter"],
                dual=params[f"{self.name}_dual"],
                C=params[f"{self.name}_C"],
                epsilon=params[f"{self.name}_epsilon"],
                loss=params[f"{self.name}_loss"],
                random_state=self.rng.getrandbits(32),
                penalty=params[f"{self.name}_penalty"],
                multi_class=params[f"{self.name}_multi_class"],
                class_weight=params[f"{self.name}_class_weight"]
            )
        else:
            raise ValueError("This shouldn't be reachable")
        return model

    def starting_points(self) -> "List[Params]":
        out: "List[Params]" = []
        for loss in self.loss:
            d: "Params" = {
                f"{self.name}_loss": loss,
                f"{self.name}_epsilon": 0,
                f"{self.name}_C": 1,
                f"{self.name}_dual": True,
                f"{self.name}_intercept_scaling": 0.1,
                f"{self.name}_fit_intercept": True,
            }
            if loss in ("hinge", "squared_hinge"):
                d.update({
                    f"{self.name}_penalty": "l2",
                    f"{self.name}_multi_class": "ovr",
                    f"{self.name}_class_weight": None,
                })

            out.append(d)
        return out


class OptimiseSGD(OptimiseSK):

    def __init__(
        self,
        loss: "Sequence[str]" = [
            "squared_error",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
            "hinge",
            "log",
            "modified_huber",
            "squared_hinge",
            "perceptron",
        ],
        seed: "Optional[int]" = None,
        name: str = "sgd",
    ):
        self.loss = loss
        self.rng = random.Random(seed)
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {}
        params.update({
            f"{self.name}_loss": trial.suggest_categorical(
                f"{self.name}_loss",
                self.loss,
            ),
            f"{self.name}_penalty": trial.suggest_categorical(
                f"{self.name}_penalty",
                ["l1", "l2", "elasticnet"],
            ),
            f"{self.name}_alpha": trial.suggest_float(
                f"{self.name}_alpha",
                1e-9,
                1,
                log=True
            ),
            f"{self.name}_fit_intercept": trial.suggest_categorical(
                f"{self.name}_fit_intercept",
                [True, False],
            ),
            f"{self.name}_max_iter": trial.suggest_categorical(
                f"{self.name}_max_iter",
                [max([1000, 2 * nsamples])],
            ),
            f"{self.name}_learning_rate": trial.suggest_categorical(
                f"{self.name}_learning_rate",
                ["constant", "optimal", "invscaling", "adaptive"],
            ),
        })

        if params[f"{self.name}_penalty"] == "elasticnet":
            params[f"{self.name}_l1_ratio"] = trial.suggest_float(
                f"{self.name}_l1_ratio",
                0.0,
                1.0,
            )

        if params[f"{self.name}_loss"] in (
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive"
        ):
            params[f"{self.name}_epsilon"] = trial.suggest_float(
                f"{self.name}_epsilon",
                0.0,
                5.0,
            )

        if params[f"{self.name}_learning_rate"] != "optimal":
            params[f"{self.name}_eta0"] = trial.suggest_float(
                f"{self.name}_eta0",
                1e-10,
                0.1,
                log=True
            )

        if params[f"{self.name}_learning_rate"] == "invscaling":
            params[f"{self.name}_power_t"] = trial.suggest_float(
                f"{self.name}_power_t",
                0.05,
                1.0,
            )

        if params[f"{self.name}_loss"] in (
            "hinge",
            "log",
            "modified_huber",
            "squared_hinge",
            "perceptron",
        ):
            params.update({
                f"{self.name}_class_weight": trial.suggest_categorical(
                    f"{self.name}_class_weight",
                    [None, "balanced"]
                ),
            })

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            SGDRegressor,
            SGDClassifier
        )

        loss = params[f"{self.name}_loss"]
        assert isinstance(loss, str)

        if loss in (
            "squared_error",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ):
            model = SGDRegressor(
                penalty=params[f"{self.name}_penalty"],
                alpha=params[f"{self.name}_alpha"],
                l1_ratio=params.get(f"{self.name}_l1_ratio", 0.15),
                fit_intercept=params[f"{self.name}_fit_intercept"],
                max_iter=params.get(f"{self.name}_max_iter", 1000),
                epsilon=params.get(f"{self.name}_epsilon", 0.1),
                learning_rate=params[f"{self.name}_learning_rate"],
                eta0=params.get(f"{self.name}_eta0", 0.01),
                power_t=params.get(f"{self.name}_power_t", 0.25),
                loss=params[f"{self.name}_loss"],
                random_state=self.rng.getrandbits(32),
            )
        elif loss in (
            "hinge",
            "log",
            "modified_huber",
            "squared_hinge",
            "perceptron",
        ):
            model = SGDClassifier(
                penalty=params[f"{self.name}_penalty"],
                alpha=params[f"{self.name}_alpha"],
                l1_ratio=params.get(f"{self.name}_l1_ratio", 0.15),
                fit_intercept=params[f"{self.name}_fit_intercept"],
                max_iter=params.get(f"{self.name}_max_iter", 1000),
                epsilon=params.get(f"{self.name}_epsilon", 0.1),
                learning_rate=params[f"{self.name}_learning_rate"],
                eta0=params.get(f"{self.name}_eta0", 0.01),
                power_t=params.get(f"{self.name}_power_t", 0.25),
                loss=params[f"{self.name}_loss"],
                random_state=self.rng.getrandbits(32),
                class_weight=params[f"{self.name}_class_weight"]
            )
        else:
            raise ValueError("This shouldn't be reachable")
        return model

    def starting_points(self) -> "List[Params]":
        from itertools import product
        from copy import copy
        out: "List[Params]" = []

        for loss, p, a in product(
            self.loss,
            ["l1", "l2", "elasticnet"],
            [1, 0.001, 1e-6]
        ):
            d: "Params" = {
                f"{self.name}_loss": loss,
                f"{self.name}_penalty": "l2",
                f"{self.name}_alpha": a,
                f"{self.name}_learning_rate": "optimal",
                f"{self.name}_epsilon": 0,
                f"{self.name}_C": 1,
                f"{self.name}_dual": True,
            }

            if loss in (
                "hinge",
                "log",
                "modified_huber",
                "squared_hinge",
                "perceptron",
            ):
                d.update({
                    f"{self.name}_class_weight": None,
                })

            if p == "elasticnet":
                for r in [0.15, 0.5, 0.85]:
                    d = copy(d)
                    d[f"{self.name}_l1_ratio"] = r
                    out.append(d)
            else:
                out.append(d)
        return out


class OptimiseLassoLars(OptimiseSK):

    def __init__(
        self,
        seed: "Optional[int]" = None,
        name: str = "lassolars",
    ):
        self.rng = random.Random(seed)
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {
            f"{self.name}_alpha": trial.suggest_float(
                f"{self.name}_alpha",
                1e-9,
                10,
                log=True
            ),
            f"{self.name}_fit_intercept": trial.suggest_categorical(
                f"{self.name}_fit_intercept",
                [True, False],
            ),
            f"{self.name}_max_iter": trial.suggest_categorical(
                f"{self.name}_max_iter",
                [min([1000, nfeatures])],
            ),
            f"{self.name}_jitter": trial.suggest_float(
                f"{self.name}_jitter",
                1e-9,
                1,
                log=True
            ),
        }
        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import LassoLars

        model = LassoLars(
            alpha=params[f"{self.name}_alpha"],
            fit_intercept=params[f"{self.name}_fit_intercept"],
            jitter=params.get(f"{self.name}_jitter", 0.0),
            max_iter=params.get(f"{self.name}_max_iter", 1000),
            random_state=self.rng.getrandbits(32),
        )
        return model

    def starting_points(self) -> "List[Params]":
        out: "List[Params]" = []

        for a in [10, 1, 1e-3, 1e-6]:
            d: "Params" = {
                f"{self.name}_alpha": a,
                f"{self.name}_fit_intercept": True,
                f"{self.name}_max_iter": 1000,
                f"{self.name}_jitter": 0,
            }
            out.append(d)
        return out


class OptimiseLars(OptimiseSK):

    def __init__(
        self,
        seed: "Optional[int]" = None,
        name: str = "lars",
    ):
        self.rng = random.Random(seed)
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":
        params = {
            f"{self.name}_fit_intercept": trial.suggest_categorical(
                f"{self.name}_fit_intercept",
                [True, False],
            ),
            f"{self.name}_n_nonzero_coefs": trial.suggest_int(
                f"{self.name}_n_nonzero_coefs",
                1,
                nfeatures
            ),
            f"{self.name}_jitter": trial.suggest_float(
                f"{self.name}_jitter",
                1e-20,
                1,
                log=True
            ),
        }
        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import Lars

        model = Lars(
            fit_intercept=params[f"{self.name}_fit_intercept"],
            jitter=params.get(f"{self.name}_jitter", 0.0),
            n_nonzero_coefs=params.get(f"{self.name}_n_nonzero_coefs", 500),
            random_state=self.rng.getrandbits(32),
        )
        return model

    def starting_points(self) -> "List[Params]":
        out: "List[Params]" = []

        for i in [50, 100, 500, 1000]:
            d: "Params" = {
                f"{self.name}_fit_intercept": True,
                f"{self.name}_n_nonzero_coefs": i,
                f"{self.name}_jitter": 0,
            }
            out.append(d)
        return out
