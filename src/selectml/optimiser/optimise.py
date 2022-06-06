#!/usr/bin/env python3

import random
import numpy as np

from baikal import Model

from typing import cast, TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union, Optional
    from typing import Dict, List, Literal, Sequence, Mapping
    from typing import Iterable
    from typing import Tuple
    from typing import TypeVar
    T = TypeVar("T")

    import optuna
    import numpy.typing as npt
    BaseTypes = Union[None, bool, str, int, float]
    Params = Dict[str, BaseTypes]
    ODatasetIn = Union[
        npt.ArrayLike,
        Sequence[npt.ArrayLike],
        Mapping[str, npt.ArrayLike],
        None
    ]
    ODatasetOut = Union[
        np.ndarray,
        List[np.ndarray],
        Dict[str, np.ndarray],
        None
    ]
    FitModel = Optional[Tuple[Model, ODatasetOut]]

    from ..bglr.wrapper import BGLR_MODELS


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


class MyModel(Model):

    @staticmethod
    def listify(x: "Union[T, List[T], Tuple[T, ...]]") -> "List[T]":
        if isinstance(x, list):
            pass
        elif isinstance(x, tuple):
            x = list(x)
        else:
            x = [x]
        return x

    @staticmethod
    def unlistify(x: "List[T]") -> "Union[List[T], T]":
        if not isinstance(x, list):
            raise ValueError("x must be a list.")
        if len(x) == 1:
            return x[0]
        return x

    def transform(
        self,
        X,
        output_names=None,
    ):
        """Predict by applying the model on the given input data.
        Parameters
        ----------
        X
            Input data. It follows the same format as in the ``fit`` method.
        output_names
            Names of required outputs (optional). You can specify any final or
            intermediate output by passing the name of its associated data
            placeholder. This is useful for debugging. If not specified, it
            will return the outputs specified at instantiation.
        Returns
        -------
        array-like or list of array-like
            The computed outputs.
        """

        # Intermediate results are stored here
        results_cache = dict()  # type: Dict[DataPlaceholder, ArrayLike]

        # Normalize inputs
        X_norm = self._normalize_data(X, self._internal_inputs)

        # Get required outputs
        if output_names is None:
            outputs = self._internal_outputs
        else:
            output_names = self.listify(output_names)
            if len(set(output_names)) != len(output_names):
                raise ValueError("output_names must be unique.")
            outputs = [
                self.get_data_placeholder(output)
                for output
                in output_names
            ]

        # We allow unused inputs to allow debugging different outputs
        # without having to change the inputs accordingly.
        nodes = self._get_required_nodes(
            X_norm, [], outputs, allow_unused_inputs=True, follow_targets=False
        )

        # Compute
        results_cache.update(X_norm)

        for node in nodes:
            Xs = [results_cache[i] for i in node.inputs]
            self._compute_node(node, Xs, results_cache)

        output_data = [results_cache[o] for o in outputs]
        if len(output_data) == 1:
            return output_data[0]
        else:
            return output_data

    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)


class OptimiseBase(object):

    name: str

    def sample(
        self,
        trial: "optuna.Trial",
        Xs: "Iterable[Optional[npt.ArrayLike]]",
        **kwargs
    ) -> "Params":
        from selectml.higher import or_else, fmap

        first = True
        groups: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "groups",
            [None for _ in Xs]
        )
        covariates: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "covariates",
            [None for _ in Xs]
        )

        if any([X is None for X in Xs]):
            nfeatures = 0
            nsamples = 0
            onehot_nfeatures = 0
            group_nfeatures = 0
            covariate_nfeatures = 0

        else:
            groups_none = [x is None for x in groups]
            if any(groups_none):
                assert all(groups_none)

            covariates_none = [x is None for x in covariates]
            if any(covariates_none):
                assert all(covariates_none)

            for X, group, covariate in zip(
                Xs,
                groups,
                covariates
            ):
                X = np.array(X)
                if len(X.shape) == 1:
                    X = X.reshape(-1, 1)

                this_nsamples = X.shape[0]
                this_nfeatures = X.shape[1]
                this_onehot_nfeatures = np.sum(ndistinct(X))

                this_group_nfeatures = or_else(0, fmap(
                    lambda h: np.asarray(h).shape[1],
                    group
                ))
                this_covariate_nfeatures = or_else(0, fmap(
                    lambda h: np.asarray(h).shape[1],
                    covariate
                ))

                if first:
                    first = False
                    nsamples = this_nsamples
                    nfeatures = this_nfeatures
                    onehot_nfeatures = this_onehot_nfeatures
                    group_nfeatures = this_group_nfeatures
                    covariate_nfeatures = this_covariate_nfeatures
                else:
                    nsamples = min([nsamples, this_nsamples])
                    nfeatures = min([nfeatures, this_nfeatures])
                    onehot_nfeatures = min([
                        onehot_nfeatures,
                        this_onehot_nfeatures
                    ])
                    group_nfeatures = min([
                        group_nfeatures,
                        this_group_nfeatures
                    ])
                    covariate_nfeatures = min([
                        covariate_nfeatures,
                        this_covariate_nfeatures
                    ])

        params = self.sample_params(
            trial,
            nsamples,
            nfeatures,
            onehot_nfeatures=onehot_nfeatures,
            ngroups=group_nfeatures,
            ncovariates=covariate_nfeatures,
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

        return model.predict(Xs)

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
        preds = model.predict(Xs)
        return model, preds

    def transform(
        self,
        model: "Model",
        Xs: "ODatasetIn",
    ) -> "ODatasetOut":
        if (model is None) or (Xs is None):
            return None

        return model.transform(Xs)

    def fit_transform(
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
        preds = model.transform(Xs)
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
        params: "Params" = {}

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
            params[f"{self.name}_nquantiles"] = trial.suggest_int(
                f"{self.name}_nquantiles",
                min([100, round(nsamples / 2)]),
                min([1000, nsamples])
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
            Make2D,
            Pipeline,
        )

        preprocessor = params.get(f"{self.name}_transformer", "drop")

        if preprocessor == "drop":
            return None
        elif preprocessor == "stdnorm":
            g = StandardScaler(name=f"{self.name}_preprocessor")
        elif preprocessor == "quantile":
            d = params[f"{self.name}_transformer_quantile_distribution"]
            g = QuantileTransformer(
                output_distribution=d,
                n_quantiles=params[f"{self.name}_nquantiles"],
                name=f"{self.name}_preprocessor",
            )
        elif preprocessor == "ordinal":
            g = OrdinalTransformer(
                boundaries=self.ordinal_boundaries,
                name=f"{self.name}_preprocessor",
            )
        elif preprocessor == "passthrough":
            g = Unity(
                name=f"{self.name}_preprocessor",
            )
        else:
            raise ValueError(f"Got unexpected preprocessor {preprocessor}")

        return Pipeline([
            ("twod", Make2D()),
            ("transformer", g),
        ], name=f"{self.name}_target_transformer")


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
        use_polynomial: bool = True,
        name: str = "covariate",
    ):
        self.options = list(options)
        self.name = name
        self.use_polynomial = use_polynomial
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

        if (target == "drop") and self.use_polynomial:
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
            g = StandardScaler(name=f"{self.name}_preprocessor")
        elif preprocessor == "quantile":
            d = params[f"{self.name}_transformer_quantile_distribution"]
            g = QuantileTransformer(
                output_distribution=d,
                n_quantiles=params[f"{self.name}_nquantiles"],
                name=f"{self.name}_preprocessor"
            )
        elif preprocessor == "robust":
            g = RobustScaler()
        elif preprocessor == "power":
            g = Pipeline(
                [
                    ("scale", RobustScaler()),
                    ("power", PowerTransformer())
                ],
                name=f"{self.name}_preprocessor"
            )
        elif preprocessor == "passthrough":
            g = Unity(
                name=f"{self.name}_preprocessor"
            )
        elif preprocessor == "drop":
            g = None
        else:
            raise ValueError(f"Got unexpected preprocessor {preprocessor}")

        if (
            (f"{self.name}_polynomial_degree" in params)
            and self.use_polynomial
        ):
            p = params[f"{self.name}_polynomial_degree"]
            assert isinstance(p, int)
            if p > 1:
                g = Pipeline(
                    [
                        ("scale", g),
                        ("power", PolynomialFeatures(
                            degree=p,
                            interaction_only=params[f"{self.name}_polynomial_interaction_only"],  # noqa
                            include_bias=False
                        )),
                    ],
                    name=f"{self.name}_preprocessor"
                )

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
        use_cache: bool = True,
        seed: "Optional[int]" = None
    ):
        from threading import Lock

        if not self._check_if_has_gemma(gemma_exe):
            options = [o for o in options if o != "gemma"]

        self.options = list(options)
        self.rng = random.Random(seed)
        self.name = name

        self.lock = Lock()
        self.use_cache = use_cache
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

        ngroups = kwargs.get("ngroups", 0)
        ncovariates = kwargs.get("ncovariates", 0)

        selector = trial.suggest_categorical(
            f"{self.name}_selector",
            self.options
        )
        params[f"{self.name}_selector"] = selector

        if selector == "drop":
            return params

        if selector == "gemma":

            if ngroups > 0:
                group_options = [True]
            else:
                group_options = [False]

            params[f"{self.name}_use_groups"] = trial.suggest_categorical(  # noqa
                f"{self.name}_use_groups",
                group_options
            )

            if ncovariates > 0:
                covariate_options = [True]
            else:
                covariate_options = [False]

            params[f"{self.name}_use_covariates"] = trial.suggest_categorical(  # noqa
                f"{self.name}_use_covariates",
                covariate_options
            )

            max_pcs = max([0, min([nsamples, nfeatures]) - 1])

            params[f"{self.name}_gemma_pcs"] = trial.suggest_int(
                f"{self.name}_gemma_pcs",
                0,
                min([4, max_pcs]),
            )

        if selector != "passthrough":
            params[f"{self.name}_nfeatures"] = (
                trial.suggest_int(
                    f"{self.name}_nfeatures",
                    min([100, round(nfeatures / 2)]),
                    nfeatures - 1,
                )
            )

        return params

    def model(  # noqa
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            MultiSURF,
            GEMMASelector,
            MAFSelector,
            Unity,
            TruncatedSVD,
            StandardScaler,
        )
        from baikal.steps import Input, ColumnStack

        selector = params.get(f"{self.name}_selector", "drop")
        if selector == "drop":
            return None

        elif selector == "passthrough":
            s = Unity(name=f"{self.name}_preprocessor")

        elif selector == "relief":
            n = params[f"{self.name}_nfeatures"]
            assert isinstance(n, int)
            s = MultiSURF(
                n=n,
                nepoch=10,
                sd=1,
                random_state=self.rng.getrandbits(32),
                name=f"{self.name}_preprocessor",
            )

        elif selector == "gemma":
            n = params[f"{self.name}_nfeatures"]
            assert isinstance(n, int)

            use_covariates = params.get(
                f"{self.name}_use_covariates",
                False
            )
            assert isinstance(use_covariates, bool)

            use_groups = params.get(
                f"{self.name}_use_groups",
                False
            )
            assert isinstance(use_groups, bool)

            pcs = params.get(f"{self.name}_gemma_pcs", 0)
            assert isinstance(pcs, int)

            marker_input = Input(name="markers")
            target_input = Input(name="y")
            inputs = [marker_input]
            gemma_inputs = [marker_input]

            if use_covariates:
                covariates_input = Input(name="covariates")
            else:
                covariates_input = None

            if use_groups:
                groups_input = Input(name="groups")
                inputs.append(groups_input)
                gemma_inputs.append(groups_input)
            else:
                groups_input = None

            cov_inputs = []

            if pcs >= 1:
                scaler = StandardScaler(
                    name=f"{self.name}_gemma_pc_scaler"
                )(marker_input)
                pc_model = TruncatedSVD(
                    n_components=pcs,
                    random_state=self.rng.getrandbits(32),
                    name=f"{self.name}_gemma_pc"
                )(scaler)
            else:
                pc_model = None

            if covariates_input is not None:
                cov_inputs.append(covariates_input)
                inputs.append(covariates_input)

            if pc_model is not None:
                cov_inputs.append(pc_model)

            if len(cov_inputs) == 0:
                cov_model = None
            elif len(cov_inputs) == 1:
                cov_model = cov_inputs[0]
            else:
                cov_model = ColumnStack()(cov_inputs)

            if cov_model is not None:
                gemma_inputs.append(cov_model)

            gemma_model = GEMMASelector(
                n=n,
                name=f"{self.name}_preprocessor",
                use_groups=groups_input is not None,
                use_covariates=cov_model is not None,
            )(gemma_inputs, target_input)
            s = MyModel(
                inputs,
                gemma_model,
                target_input,
                name=f"{self.name}_preprocessor"
            )

        elif selector == "maf":
            n = params[f"{self.name}_nfeatures"]
            assert isinstance(n, int)
            s = MAFSelector(
                n=n,
                name=f"{self.name}_preprocessor",
            )

        else:
            raise ValueError(f"Got unexpected feature selector {selector}")

        return s

    @staticmethod
    def _check_if_has_gemma(exe: str):
        from shutil import which
        return which(exe) is not None

    @staticmethod
    def select_inputs(
        model,
        data: "Union[List[np.ndarray], np.ndarray]",
    ) -> "Dict[str, Union[List[np.ndarray], np.ndarray]]":
        from copy import copy

        out: "Dict[str, Union[List[np.ndarray], np.ndarray]]" = {}
        if isinstance(data, np.ndarray):
            out["markers"] = data
            return out
        elif isinstance(data, list) and len(data) == 1:
            out["markers"] = data
            return out

        data = copy(data)
        for key in ["markers", "groups", "covariates"]:
            if len(data) == 0:
                break

            try:
                model.get_step(key)
                out[key] = np.asarray(data.pop(0))
            except ValueError:
                pass

        return out

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

        if (
            isinstance(Xs, list) and not
            any(isinstance(xi, (float, int)) for xi in Xs)
        ):
            Xs_: "Union[np.ndarray, List[np.ndarray]]" = list(map(np.asarray, Xs))  # noqa: E501
        elif isinstance(Xs, np.ndarray):
            Xs_ = Xs
        else:
            raise ValueError(
                "Xs needs to be a list of arrays "
                "or single np array"
            )

        assert isinstance(selector, str)
        key = (id(Xs_), selector)

        """ This isn't a great solution, it repeats some work from the
        model selection bit. Would be good to lookup cache in there,
        but we need the data to do the lookup sooo.
        """
        with self.lock:
            if (key in self.cache) and self.use_cache:
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
                model_ = self.model(params, **kwargs)

                if model_ is None:
                    return None
                else:
                    model = model_

                if selector == "gemma":
                    i = self.select_inputs(model, Xs_)
                    model.fit(i, y)
                else:
                    if isinstance(Xs_, list):
                        Xs_ = Xs_[0]
                    model.fit(Xs_, y)

                if self.use_cache:
                    self.cache[key] = model

        return model

    def predict(
        self,
        model: "Model",
        Xs: "ODatasetIn",
        **kwargs
    ) -> "ODatasetOut":
        if (model is None) or (Xs is None):
            return None

        if (
            isinstance(Xs, list) and not
            any(isinstance(xi, (float, int)) for xi in Xs)
        ):
            Xs_: "Union[np.ndarray, List[np.ndarray]]" = list(map(np.asarray, Xs))  # noqa: E501
        elif isinstance(Xs, np.ndarray):
            Xs_ = np.asarray(Xs)
        else:
            raise ValueError(
                "Xs needs to be a list of arrays "
                "or single np array"
            )

        if isinstance(model, MyModel):
            return model.predict(self.select_inputs(model, Xs_))
        else:
            return model.predict(Xs_)

    def transform(
        self,
        model: "Model",
        Xs: "ODatasetIn",
        **kwargs
    ) -> "ODatasetOut":
        if (model is None) or (Xs is None):
            return None

        if (
            isinstance(Xs, list) and not
            any(isinstance(xi, (float, int)) for xi in Xs)
        ):
            Xs_: "Union[np.ndarray, List[np.ndarray]]" = list(map(np.asarray, Xs))  # noqa: E501
        elif isinstance(Xs, np.ndarray):
            Xs_ = np.asarray(Xs)
        else:
            raise ValueError(
                "Xs needs to be a list of arrays "
                "or single np array"
            )

        if isinstance(model, MyModel):
            return model.transform(self.select_inputs(model, Xs_))
        else:
            return model.transform(Xs_)


class OptimisePostFeatureSelector(OptimiseBase):

    def __init__(
        self,
        options: "Sequence[str]" = [
            "passthrough",
            "f_classif",
            "chi2",
            "f_regression",
            "mutual_info_regression",
            "mutual_info_classif",
        ],
        name: str = "post_feature_selector",
        seed=None
    ):
        self.options = list(options)
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
        from math import floor
        params = {}

        if nfeatures == 0:
            raise ValueError(
                "We need at least one feature."
            )

        selector = trial.suggest_categorical(
            f"{self.name}_selector",
            self.options
        )
        params[f"{self.name}_selector"] = selector

        if selector == "drop":
            return params

        if selector != "passthrough":
            params[f"{self.name}_nfeatures"] = (
                trial.suggest_int(
                    f"{self.name}_nfeatures",
                    min([10, floor(nfeatures / 2)]),
                    nfeatures - 1,
                )
            )

        return params

    def model(  # noqa
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import (
            SelectKBest,
            Unity,
        )
        from sklearn.feature_selection import (
            f_classif,
            chi2,
            mutual_info_classif,
            mutual_info_regression,
            f_regression,
        )

        selector = params.get(f"{self.name}_selector", "drop")
        if selector == "drop":
            return None

        elif selector == "passthrough":
            s = Unity(name=f"{self.name}_selector")
            return s

        n = params[f"{self.name}_nfeatures"]

        if selector == "f_classif":
            s = SelectKBest(
                score_func=f_classif,
                k=n,
                name=f"{self.name}_selector"
            )
        elif selector == "chi2":
            s = SelectKBest(
                score_func=chi2,
                k=n,
                name=f"{self.name}_selector"
            )
        elif selector == "f_regression":
            s = SelectKBest(
                score_func=f_regression,
                k=n,
                name=f"{self.name}_selector"
            )
        elif selector == "mutual_info_regression":
            s = SelectKBest(
                score_func=lambda X, y: mutual_info_regression(
                    X,
                    y,
                    random_state=self.rng.getrandbits(32)
                ),
                k=n,
                name=f"{self.name}_selector"
            )
        elif selector == "mutual_info_classif":
            s = SelectKBest(
                score_func=lambda X, y: mutual_info_classif(
                    X,
                    y,
                    random_state=self.rng.getrandbits(32)
                ),
                k=n,
                name=f"{self.name}_selector"
            )
        else:
            raise ValueError(f"Got unexpected feature selector {selector}")

        return s


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
                max([0, min_ncomponents]),
                max([0, max_ncomponents])
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
            g = Unity(name=f"{self.name}_preprocessor")

        elif preprocessor == "onehot":
            g = OneHotEncoder(
                categories="auto",
                drop=None,
                sparse=False,
                handle_unknown="ignore",
                name=f"{self.name}_preprocessor",
            )

        elif preprocessor == "maf":
            g = MAFScaler(
                ploidy=self.ploidy,
                name=f"{self.name}_preprocessor",
            )

        elif preprocessor == "noia_add":
            g = NOIAAdditiveScaler(name=f"{self.name}_preprocessor")

        elif preprocessor == "pca":
            g = Pipeline(
                [
                    ("prescaler", MAFScaler(ploidy=self.ploidy)),
                    (
                        "pca",
                        TruncatedSVD(
                            n_components=params[f"{self.name}_pca_ncomponents"],  # noqa: E501
                            random_state=self.rng.getrandbits(32)
                        )
                    )
                ],
                name=f"{self.name}_preprocessor",
            )

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
            "euclidean",
            "noia_additive",
            "noia_dominance",
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
            NOIAAdditiveKernel,
            NOIADominanceKernel,
        )

        preprocessor = params.get(f"{self.name}_transformer", "drop")

        if preprocessor == "drop":
            return None

        if preprocessor == "vanraden":
            p = VanRadenSimilarity(
                ploidy=self.ploidy,
                distance=True,
                name=f"{self.name}_preprocessor",
            )

        elif preprocessor == "manhattan":
            p = ManhattanDistance(name=f"{self.name}_preprocessor")

        elif preprocessor == "euclidean":
            p = EuclideanDistance(name=f"{self.name}_preprocessor")

        elif preprocessor == "noia_additive":
            p = NOIAAdditiveKernel(name=f"{self.name}_preprocessor")

        elif preprocessor == "noia_dominance":
            p = NOIADominanceKernel(name=f"{self.name}_preprocessor")

        else:
            raise ValueError(f"Got unexpected preprocessor {preprocessor}")

        steps = []
        if preprocessor in ["manhattan", "euclidean"]:
            steps.append(("prescaler", MAFScaler(ploidy=self.ploidy)))

        steps.append(("transformer", p))

        if preprocessor not in ["noia_additive", "noia_dominance"]:
            steps.append(("postscaler", RobustScaler()))

        if len(steps) > 1:
            return Pipeline(steps, name=f"{self.name}_preprocessor")
        else:
            return p


class OptimiseAddEpistasis(OptimiseBase):

    def __init__(
        self,
        allow: bool = True,
        AA: float = 2.,
        Aa: float = 1.,
        aa: float = 0.,
        name: str = "additive_epistasis"
    ):
        self.allow = allow
        self.AA = AA
        self.Aa = Aa
        self.aa = aa
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":

        if self.allow:
            options = [True]
        else:
            options = [False]

        params = {
            f"{self.name}_use": trial.suggest_categorical(
                f"{self.name}_use",
                options
            )
        }

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        use = params.get(f"{self.name}_use", False)

        if not use:
            return None

        from .wrapper import NOIAAdditiveKernel, HadamardCovariance

        k = NOIAAdditiveKernel(AA=self.AA, Aa=self.Aa, aa=self.aa)
        h = HadamardCovariance(
            a=k,
            b=k,
            fit_b=False,
            name=f"{self.name}_preprocessor"
        )
        return h


class OptimiseDomEpistasis(OptimiseBase):

    def __init__(
        self,
        allow: bool = True,
        AA: float = 2.,
        Aa: float = 1.,
        aa: float = 0.,
        name: str = "dominance_epistasis"
    ):
        self.allow = allow
        self.AA = AA
        self.Aa = Aa
        self.aa = aa
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":

        if self.allow:
            options = [True]
        else:
            options = [False]

        params = {
            f"{self.name}_use": trial.suggest_categorical(
                f"{self.name}_use",
                options
            )
        }

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        use = params.get(f"{self.name}_use", False)

        if not use:
            return None

        from .wrapper import NOIADominanceKernel, HadamardCovariance

        k = NOIADominanceKernel(AA=self.AA, Aa=self.Aa, aa=self.aa)
        h = HadamardCovariance(
            a=k,
            b=k,
            fit_b=False,
            name=f"{self.name}_preprocessor"
        )
        return h


class OptimiseAddDomEpistasis(OptimiseBase):

    def __init__(
        self,
        allow: bool = True,
        AA: float = 2.,
        Aa: float = 1.,
        aa: float = 0.,
        name: str = "addxdom_epistasis"
    ):
        self.allow = allow
        self.AA = AA
        self.Aa = Aa
        self.aa = aa
        self.name = name
        return

    def sample_params(
        self,
        trial: "optuna.Trial",
        nsamples: int,
        nfeatures: int,
        **kwargs
    ) -> "Params":

        if self.allow:
            options = [True]
        else:
            options = [False]

        params = {
            f"{self.name}_use": trial.suggest_categorical(
                f"{self.name}_use",
                options
            )
        }

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        use = params.get(f"{self.name}_use", False)

        if not use:
            return None

        from .wrapper import (
            NOIAAdditiveKernel,
            NOIADominanceKernel,
            HadamardCovariance
        )

        a = NOIAAdditiveKernel(AA=self.AA, Aa=self.Aa, aa=self.aa)
        b = NOIADominanceKernel(AA=self.AA, Aa=self.Aa, aa=self.aa)
        h = HadamardCovariance(a=a, b=b, name=f"{self.name}_preprocessor")
        return h


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

        if preprocessor == "drop":
            return params

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
                name=f"{self.name}_preprocessor",
            )
        elif preprocessor == "laplacian":
            p = Nystroem(
                kernel="laplacian",
                gamma=params[f"{self.name}_laplacian_gamma"],
                n_components=ncomponents,
                random_state=self.rng.getrandbits(32),
                name=f"{self.name}_preprocessor",
            )
        elif preprocessor == "poly":
            p = Nystroem(
                kernel="poly",
                gamma=params[f"{self.name}_poly_gamma"],
                n_components=ncomponents,
                random_state=self.rng.getrandbits(32),
                degree=2,
                name=f"{self.name}_preprocessor",
            )
        else:
            raise ValueError(f"Got invalid transformer {preprocessor}.")

        return Pipeline(
            [
                ("prescaler", MAFScaler(ploidy=self.ploidy)),
                ("nonlinear", p),
                ("scaler", QuantileTransformer(
                    n_quantiles=params[f"{self.name}_nquantiles"]
                ))
            ],
            name=f"{self.name}_preprocessor",
        )


class OptimiseGrouping(OptimiseBase):

    def __init__(
        self,
        max_ncomponents,
        allow_pca: bool = True,
        name: str = "grouping",
        seed: "Optional[int]" = None,
    ):
        self.max_ncomponents = max_ncomponents
        self.allow_pca = allow_pca
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
            options = ["onehot"]

        params[f"{self.name}_transformer"] = trial.suggest_categorical(  # noqa
            f"{self.name}_transformer",
            options,
        )

        if (nfeatures > 0) and self.allow_pca:
            max_ncomponents = min([
                nsamples - 1,
                nfeatures - 1,
                self.max_ncomponents
            ])

            min_ncomponents = max([
                0,
                min([
                    3,
                    max_ncomponents,
                    round(max_ncomponents / 2)
                ])
            ])

            if max_ncomponents > 3:
                params[f"{self.name}_pca_ncomponents"] = trial.suggest_int(  # noqa
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
            Pipeline,
            OneHotEncoder,
            TruncatedSVD,
        )

        preprocessor = params.get(f"{self.name}_transformer", "drop")

        if preprocessor == "drop":
            return None

        elif preprocessor == "onehot":
            g = OneHotEncoder(
                categories="auto",
                drop="if_binary",
                handle_unknown="ignore",
                sparse=False,
                name=f"{self.name}_onehot",
            )
        else:
            raise ValueError("received invalid preprocessor.")

        if f"{self.name}_pca_ncomponents" in params:
            g = Pipeline(
                [
                    ("ohe", g),
                    (
                        "pca",
                        TruncatedSVD(
                            n_components=params[f"{self.name}_pca_ncomponents"],  # noqa
                            random_state=self.rng.getrandbits(32),
                            name=f"{self.name}_pca",
                        )
                    )
                ],
                name=f"{self.name}_preprocessor",
            )

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

        if preprocessor == "drop":
            return params

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

        return Pipeline(
            [
                ("nonlinear", p),
                ("scaler", QuantileTransformer(
                    n_quantiles=params[f"{self.name}_nquantiles"]
                ))
            ],
            name=f"{self.name}_preprocessor",
        )


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
        n_jobs: "Optional[int]" = None,
    ):
        self.objectives = objectives
        self.rng = random.Random(seed)
        self.n_jobs = n_jobs
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
            n_jobs=self.n_jobs,
            verbosity=1,
            name=f"{self.name}",
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
        objective: "Literal['classification', 'regression']",
        seed: "Optional[int]" = None,
        name: str = "knn",
        n_jobs: "Optional[int]" = None,
    ):
        self.rng = random.Random(seed)
        self.objective = objective
        self.name = name
        self.n_jobs = n_jobs
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
        params.update({
            f"{self.name}_n_neighbors": trial.suggest_int(
                f"{self.name}_n_neighbors",
                2,
                max([0, min([100, nsamples - 1])]),
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
        from .wrapper import KNeighborsRegressor, KNeighborsClassifier

        if self.objective == "regression":
            cls = KNeighborsRegressor
        elif self.objective == "classification":
            cls = KNeighborsClassifier
        else:
            raise ValueError("task must be classification or regression")

        model = cls(
            n_neighbors=params[f"{self.name}_n_neighbors"],
            weights=params[f"{self.name}_weights"],
            leaf_size=params[f"{self.name}_leaf_size"],
            algorithm=params[f"{self.name}_algorithm"],
            p=params[f"{self.name}_p"],
            n_jobs=self.n_jobs,
            name=f"{self.name}",
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
                f"{self.name}_p": 2,
            },
            {
                f"{self.name}_n_neighbors": 10,
                f"{self.name}_weights": "distance",
                f"{self.name}_leaf_size": 50,
                f"{self.name}_algorithm": "kd_tree",
                f"{self.name}_p": 2,
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
        n_jobs: "Optional[int]" = None,
        name: str = "rf",
    ):
        self.criterion = criterion
        self.rng = random.Random(seed)
        self.n_jobs = n_jobs
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

        bootstrap = trial.suggest_categorical(
            f"{self.name}_bootstrap",
            [True, False]
        )

        params[f"{self.name}_bootstrap"] = bootstrap
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
                f"{self.name}_min_impurity_decrease",
                0,
                1
            ),
        })

        if bootstrap:
            params[f"{self.name}_oob_score"] = trial.suggest_categorical(
                f"{self.name}_oob_score",
                [True, False]
            )
            params[f"{self.name}_max_samples"] = trial.suggest_float(
                f"{self.name}_max_samples",
                0.01,
                1.0
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
            max_samples=params.get(f"{self.name}_max_samples", None),
            n_estimators=params[f"{self.name}_n_estimators"],
            max_features=params[f"{self.name}_max_features"],
            min_samples_split=params[f"{self.name}_min_samples_split"],
            min_samples_leaf=params[f"{self.name}_min_samples_leaf"],
            min_impurity_decrease=params[f"{self.name}_min_impurity_decrease"],
            bootstrap=params[f"{self.name}_bootstrap"],
            oob_score=oob_score,
            n_jobs=self.n_jobs,
            random_state=self.rng.getrandbits(32),
            name=f"{self.name}",
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
        n_jobs: "Optional[int]" = None,
        name: str = "extratrees",
    ):
        self.criterion = criterion
        self.rng = random.Random(seed)
        self.n_jobs = n_jobs
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
            n_jobs=self.n_jobs,
            random_state=self.rng.getrandbits(32),
            name=f"{self.name}",
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
            verbose=False,
            random_state=self.rng.getrandbits(32),
            name=f"{self.name}",
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
            f"{self.name}_fit_intercept": trial.suggest_categorical(
                f"{self.name}_fit_intercept",
                [True, False],
            ),
            f"{self.name}_max_iter": trial.suggest_int(
                f"{self.name}_max_iter",
                nfeatures * 10,
                (nfeatures * 10)
            )
        })

        if nsamples > nfeatures:
            params[f"{self.name}_dual"] = trial.suggest_categorical(
                f"{self.name}_dual",
                [False],
            )

        if params[f"{self.name}_loss"] in (
            "epsilon_insensitive",
            "squared_epsilon_insensitive"
        ):
            params[f"{self.name}_epsilon"] = trial.suggest_float(
                    f"{self.name}_epsilon",
                    0.0,
                    5.0,
                )

        if params[f"{self.name}_loss"] in ("hinge", "squared_hinge"):
            params.update({
                f"{self.name}_multi_class": trial.suggest_categorical(
                    f"{self.name}_multi_class",
                    ["ovr", "crammer_singer"]
                ),
                f"{self.name}_class_weight": trial.suggest_categorical(
                    f"{self.name}_class_weight",
                    [None, "balanced"]
                ),
            })
            if params[f"{self.name}_loss"] == "squared_hinge":
                # L1 not supported with hinge
                params[f"{self.name}_penalty"] = trial.suggest_categorical(
                    f"{self.name}_penalty",
                    ["l1", "l2"],
                )

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

        dual = params.get(f"{self.name}_dual", True)
        loss = params[f"{self.name}_loss"]

        if loss in ("epsilon_insensitive", "squared_epsilon_insensitive"):
            if loss == "epsilon_insensitive":
                dual = True

            model = LinearSVR(
                fit_intercept=params[f"{self.name}_fit_intercept"],
                intercept_scaling=params[f"{self.name}_intercept_scaling"],
                max_iter=params[f"{self.name}_max_iter"],
                dual=dual,
                C=params[f"{self.name}_C"],
                epsilon=params[f"{self.name}_epsilon"],
                loss=loss,
                random_state=self.rng.getrandbits(32),
                name=f"{self.name}",
            )
        elif loss in ("hinge", "squared_hinge"):
            penalty = params.get(f"{self.name}_penalty", "l2")
            if loss == "squared_hinge":
                dual = False

            model = LinearSVC(
                fit_intercept=params[f"{self.name}_fit_intercept"],
                intercept_scaling=params[f"{self.name}_intercept_scaling"],
                max_iter=params[f"{self.name}_max_iter"],
                dual=dual,
                C=params[f"{self.name}_C"],
                loss=loss,
                random_state=self.rng.getrandbits(32),
                penalty=penalty,
                multi_class=params[f"{self.name}_multi_class"],
                class_weight=params[f"{self.name}_class_weight"],
                name=f"{self.name}",
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
            f"{self.name}_learning_rate": trial.suggest_categorical(
                f"{self.name}_learning_rate",
                ["constant", "optimal", "invscaling", "adaptive"],
            ),
        })

        max_iter: int = cast(int, trial.suggest_categorical(
            f"{self.name}_max_iter",
            [max([1000, 2 * nsamples])],
        ))
        params[f"{self.name}_max_iter"] = max_iter

        use_average = trial.suggest_categorical(
            f"{self.name}_use_average",
            [True, False],
        )
        params[f"{self.name}_use_average"] = use_average

        if use_average:
            assert isinstance(max_iter, int)
            params[f"{self.name}_average"] = trial.suggest_int(
                f"{self.name}_average",
                1,
                round(max_iter / 2),
            )

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
                average=params.get(f"{self.name}_average", False),
                name=str(self.name),
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
                class_weight=params[f"{self.name}_class_weight"],
                average=params.get(f"{self.name}_average", False),
                name=str(self.name),
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
            name=f"{self.name}",
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
            name=f"{self.name}",
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


class OptimiseBGLR(OptimiseSK):

    def __init__(
        self,
        objective: "Literal['gaussian', 'ordinal']",
        seed: "Optional[int]" = None,
        name: str = "bglr",
    ):
        self.objective = objective
        self.rng = random.Random(seed)
        self.name = name
        return

    def sample(  # noqa: C901
        self,
        trial: "optuna.Trial",
        Xs: "Iterable[Optional[npt.ArrayLike]]",
        **kwargs
    ) -> "Params":
        from selectml.higher import or_else, fmap

        first = True
        adds: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "adds",
            [None for _ in Xs]
        )
        doms: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "doms",
            [None for _ in Xs]
        )
        epiadds: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "epiadds",
            [None for _ in Xs]
        )
        epidoms: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "epidoms",
            [None for _ in Xs]
        )
        epiaddxdoms: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "epiaddxdoms",
            [None for _ in Xs]
        )
        groups: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "groups",
            [None for _ in Xs]
        )
        covariates: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "covariates",
            [None for _ in Xs]
        )

        add_none = [x is None for x in adds]
        if any(add_none):
            assert all(add_none)

        dom_none = [x is None for x in doms]
        if any(dom_none):
            assert all(dom_none)

        epiadds_none = [x is None for x in epiadds]
        if any(epiadds_none):
            assert all(epiadds_none)

        epidoms_none = [x is None for x in epidoms]
        if any(epidoms_none):
            assert all(epidoms_none)

        epiaddxdoms_none = [x is None for x in epiaddxdoms]
        if any(epiaddxdoms_none):
            assert all(epiaddxdoms_none)

        groups_none = [x is None for x in groups]
        if any(groups_none):
            assert all(groups_none)

        covariates_none = [x is None for x in covariates]
        if any(covariates_none):
            assert all(covariates_none)

        for X, add, dom, epiadd, epidom, epiaddxdom, group, covariate in zip(  # noqa: E501
            Xs,
            adds,
            doms,
            epiadds,
            epidoms,
            epiaddxdoms,
            groups,
            covariates,
        ):
            X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)

            this_nsamples = X.shape[0]
            this_nfeatures = X.shape[1]
            this_onehot_nfeatures = np.sum(ndistinct(X))

            this_add_nfeatures = or_else(0, fmap(
                lambda h: np.asarray(h).shape[1],
                add
            ))

            this_dom_nfeatures = or_else(0, fmap(
                lambda h: np.asarray(h).shape[1],
                dom
            ))

            this_epiadd_nfeatures = or_else(0, fmap(
                lambda h: np.asarray(h).shape[1],
                epiadd
            ))

            this_epidom_nfeatures = or_else(0, fmap(
                lambda h: np.asarray(h).shape[1],
                epidom
            ))

            this_epiaddxdom_nfeatures = or_else(0, fmap(
                lambda h: np.asarray(h).shape[1],
                epiaddxdom
            ))

            this_group_nfeatures = or_else(0, fmap(
                lambda h: np.asarray(h).shape[1],
                group
            ))

            this_covariate_nfeatures = or_else(0, fmap(
                lambda h: np.asarray(h).shape[1],
                covariate
            ))

            if first:
                first = False
                nsamples = this_nsamples
                nfeatures = this_nfeatures
                onehot_nfeatures = this_onehot_nfeatures
                add_nfeatures = this_add_nfeatures
                dom_nfeatures = this_dom_nfeatures
                epiadd_nfeatures = this_epiadd_nfeatures
                epidom_nfeatures = this_epidom_nfeatures
                epiaddxdom_nfeatures = this_epiaddxdom_nfeatures
                group_nfeatures = this_group_nfeatures
                covariate_nfeatures = this_covariate_nfeatures
            else:
                nsamples = min([nsamples, this_nsamples])
                nfeatures = min([nfeatures, this_nfeatures])
                onehot_nfeatures = min([
                    onehot_nfeatures,
                    this_onehot_nfeatures
                ])
                add_nfeatures = min([
                    add_nfeatures,
                    this_add_nfeatures
                ])
                dom_nfeatures = min([
                    dom_nfeatures,
                    this_dom_nfeatures
                ])
                epiadd_nfeatures = min([
                    epiadd_nfeatures,
                    this_epiadd_nfeatures
                ])
                epidom_nfeatures = min([
                    epidom_nfeatures,
                    this_epidom_nfeatures
                ])
                epiaddxdom_nfeatures = min([
                    epiaddxdom_nfeatures,
                    this_epiaddxdom_nfeatures
                ])
                group_nfeatures = min([
                    group_nfeatures,
                    this_group_nfeatures
                ])
                covariate_nfeatures = min([
                    covariate_nfeatures,
                    this_covariate_nfeatures
                ])

        params: "Params" = {}

        params[f"{self.name}_response_type"] = trial.suggest_categorical(
            f"{self.name}_response_type",
            [self.objective],
        )

        params[f"{self.name}_r2"] = trial.suggest_float(
            f"{self.name}_r2",
            0.2,
            0.8,
        )

        BGLR_MODELS = [
            'FIXED', 'BRR', 'BL',
            'BayesA', 'BayesB', 'BayesC',
        ]  # excluded 'RKHS'

        if nfeatures > 0:
            params[f"{self.name}_markers"] = trial.suggest_categorical(
                f"{self.name}_markers",
                BGLR_MODELS,
            )

        if add_nfeatures > 0:
            params[f"{self.name}_add"] = trial.suggest_categorical(
                f"{self.name}_add",
                ["RKHS"],
            )

        if dom_nfeatures > 0:
            params[f"{self.name}_dom"] = trial.suggest_categorical(
                f"{self.name}_dom",
                ["RKHS"],
            )

        if epiadd_nfeatures > 0:
            params[f"{self.name}_epiadd"] = trial.suggest_categorical(
                f"{self.name}_epiadd",
                ["RKHS"],
            )

        if epidom_nfeatures > 0:
            params[f"{self.name}_epidom"] = trial.suggest_categorical(
                f"{self.name}_epidom",
                ["RKHS"],
            )

        if epiaddxdom_nfeatures > 0:
            params[f"{self.name}_epiaddxdom"] = trial.suggest_categorical(
                f"{self.name}_epiaddxdom",
                ["RKHS"],
            )

        if group_nfeatures > 0:
            params[f"{self.name}_groups"] = trial.suggest_categorical(
                f"{self.name}_groups",
                BGLR_MODELS,
            )

        if covariate_nfeatures > 0:
            params[f"{self.name}_covariates"] = trial.suggest_categorical(
                f"{self.name}_covariates",
                ["FIXED"],
            )

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import BGLRRegressor
        valid_models = [
            'FIXED', 'BRR', 'BL',
            'BayesA', 'BayesB', 'BayesC', 'RKHS'
        ]

        bglr_models: "List[BGLR_MODELS]" = []
        bglr_names: "List[str]" = []
        if f"{self.name}_markers" in params:
            markers = params[f"{self.name}_markers"]
            assert markers in valid_models
            bglr_models.append(cast("BGLR_MODELS", markers))
            bglr_names.append("markers")

        if f"{self.name}_add" in params:
            add = params[f"{self.name}_add"]
            assert add in valid_models
            bglr_models.append(cast("BGLR_MODELS", add))
            bglr_names.append("add")

        if f"{self.name}_dom" in params:
            dom = params[f"{self.name}_dom"]
            assert dom in valid_models
            bglr_models.append(cast("BGLR_MODELS", dom))
            bglr_names.append("dom")

        if f"{self.name}_epiadd" in params:
            epiadd = params[f"{self.name}_epiadd"]
            assert epiadd in valid_models
            bglr_models.append(cast("BGLR_MODELS", epiadd))
            bglr_names.append("epiadd")

        if f"{self.name}_epidom" in params:
            epidom = params[f"{self.name}_epidom"]
            assert epidom in valid_models
            bglr_models.append(cast("BGLR_MODELS", epidom))
            bglr_names.append("epidom")

        if f"{self.name}_epiaddxdom" in params:
            epiaddxdom = params[f"{self.name}_epiaddxdom"]
            assert epiaddxdom in valid_models
            bglr_models.append(cast("BGLR_MODELS", epiaddxdom))
            bglr_names.append("epiaddxdom")

        if f"{self.name}_groups" in params:
            groups = params[f"{self.name}_groups"]
            assert groups in valid_models
            bglr_models.append(cast("BGLR_MODELS", groups))
            bglr_names.append("groups")

        if f"{self.name}_covariates" in params:
            covariates = params[f"{self.name}_covariates"]
            assert covariates in valid_models
            bglr_models.append(cast("BGLR_MODELS", covariates))
            bglr_names.append("covariates")

        response_type = params.get(f"{self.name}_response_type", "gaussian")
        assert response_type in ('gaussian', 'ordinal')
        r2 = params.get(f"{self.name}_r2", 0.5)
        assert isinstance(r2, float)

        model = BGLRRegressor(
            models=bglr_models,
            component_names=bglr_names,
            niter=10000,
            burnin=1000,
            response_type=cast(
                "Literal['gaussian', 'ordinal']",
                response_type
            ),
            R2=r2,
            random_state=self.rng.getrandbits(32),
            verbose=False,
        )
        return model

    def starting_points(self) -> "List[Params]":
        out: "List[Params]" = []
        return out

    def fit(
        self,
        params: "Params",
        Xs: "ODatasetIn",
        y: "Optional[npt.ArrayLike]" = None,
        **kwargs,
    ) -> "Optional[Model]":
        if isinstance(Xs, np.ndarray):
            X: ODatasetOut = np.asarray(Xs)
        elif isinstance(Xs, list):
            X = [np.asarray(xi) for xi in Xs]
        elif isinstance(Xs, dict):
            X = {k: np.asarray(xi) for k, xi in Xs.items()}
        else:
            raise ValueError("Invalid data")

        model = self.model(params)

        if model is None:
            return None

        model.fit(X, y)
        return model


class OptimiseSKBGLR(OptimiseSK):

    def __init__(
        self,
        objective: "Literal['gaussian', 'ordinal']",
        seed: "Optional[int]" = None,
        name: str = "sk_bglr",
    ):
        self.objective = objective
        self.rng = random.Random(seed)
        self.name = name
        return

    def sample(  # noqa: C901
        self,
        trial: "optuna.Trial",
        Xs: "Iterable[Optional[npt.ArrayLike]]",
        **kwargs
    ) -> "Params":
        from selectml.higher import or_else, fmap

        first = True
        dists: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "dists",
            [None for _ in Xs]
        )
        nonlinear: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "nonlinear",
            [None for _ in Xs]
        )
        groups: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "groups",
            [None for _ in Xs]
        )
        covariates: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "covariates",
            [None for _ in Xs]
        )
        interactions: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "interactions",
            [None for _ in Xs]
        )

        if any([X is None for X in Xs]):
            nfeatures = 0
            nsamples = 0
            dists_nfeatures = 0
            nonlinear_nfeatures = 0
            onehot_nfeatures = 0
            group_nfeatures = 0
            covariate_nfeatures = 0
            interactions_nfeatures = 0

        else:
            dists_none = [x is None for x in dists]
            if any(dists_none):
                assert all(dists_none)

            nonlinear_none = [x is None for x in nonlinear]
            if any(nonlinear_none):
                assert all(nonlinear_none)

            groups_none = [x is None for x in groups]
            if any(groups_none):
                assert all(groups_none)

            covariates_none = [x is None for x in covariates]
            if any(covariates_none):
                assert all(covariates_none)

            interactions_none = [x is None for x in interactions]
            if any(interactions_none):
                assert all(interactions_none)

            for X, dist, nl, group, covariate, interaction in zip(
                Xs,
                dists,
                nonlinear,
                groups,
                covariates,
                interactions,
            ):
                X = np.array(X)
                if len(X.shape) == 1:
                    X = X.reshape(-1, 1)

                this_nsamples = X.shape[0]
                this_nfeatures = X.shape[1]
                this_onehot_nfeatures = np.sum(ndistinct(X))

                this_dists_nfeatures = or_else(0, fmap(
                    lambda h: np.asarray(h).shape[1],
                    dist
                ))

                this_nonlinear_nfeatures = or_else(0, fmap(
                    lambda h: np.asarray(h).shape[1],
                    nl
                ))

                this_group_nfeatures = or_else(0, fmap(
                    lambda h: np.asarray(h).shape[1],
                    group
                ))
                this_covariate_nfeatures = or_else(0, fmap(
                    lambda h: np.asarray(h).shape[1],
                    covariate
                ))

                this_interactions_nfeatures = or_else(0, fmap(
                    lambda h: np.asarray(h).shape[1],
                    interaction
                ))

                if first:
                    first = False
                    nsamples = this_nsamples
                    nfeatures = this_nfeatures
                    onehot_nfeatures = this_onehot_nfeatures
                    dists_nfeatures = this_dists_nfeatures
                    nonlinear_nfeatures = this_nonlinear_nfeatures
                    group_nfeatures = this_group_nfeatures
                    covariate_nfeatures = this_covariate_nfeatures
                    interactions_nfeatures = this_interactions_nfeatures
                else:
                    nsamples = min([nsamples, this_nsamples])
                    nfeatures = min([nfeatures, this_nfeatures])
                    onehot_nfeatures = min([
                        onehot_nfeatures,
                        this_onehot_nfeatures
                    ])
                    dists_nfeatures = min([
                        dists_nfeatures,
                        this_dists_nfeatures
                    ])
                    nonlinear_nfeatures = min([
                        nonlinear_nfeatures,
                        this_nonlinear_nfeatures
                    ])
                    group_nfeatures = min([
                        group_nfeatures,
                        this_group_nfeatures
                    ])
                    covariate_nfeatures = min([
                        covariate_nfeatures,
                        this_covariate_nfeatures
                    ])
                    interactions_nfeatures = min([
                        interactions_nfeatures,
                        this_interactions_nfeatures
                    ])

        params: "Params" = {}

        params[f"{self.name}_response_type"] = trial.suggest_categorical(
            f"{self.name}_response_type",
            [self.objective],
        )

        params[f"{self.name}_r2"] = trial.suggest_float(
            f"{self.name}_r2",
            0.2,
            0.8,
        )

        BGLR_MODELS = [
            'FIXED', 'BRR', 'BL',
            'BayesA', 'BayesB', 'BayesC',
        ]  # excluded 'RKHS'

        if nfeatures > 0:
            params[f"{self.name}_markers"] = trial.suggest_categorical(
                f"{self.name}_markers",
                BGLR_MODELS,
            )

        if dists_nfeatures > 0:
            params[f"{self.name}_dists"] = trial.suggest_categorical(
                f"{self.name}_dists",
                BGLR_MODELS,
            )

        if nonlinear_nfeatures > 0:
            params[f"{self.name}_nonlinear"] = trial.suggest_categorical(
                f"{self.name}_nonlinear",
                BGLR_MODELS,
            )

        if group_nfeatures > 0:
            params[f"{self.name}_groups"] = trial.suggest_categorical(
                f"{self.name}_groups",
                BGLR_MODELS,
            )

        if covariate_nfeatures > 0:
            params[f"{self.name}_covariates"] = trial.suggest_categorical(
                f"{self.name}_covariates",
                BGLR_MODELS,
            )

        if interactions_nfeatures > 0:
            params[f"{self.name}_interactions"] = trial.suggest_categorical(
                f"{self.name}_interactions",
                BGLR_MODELS,
            )

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import BGLRRegressor

        valid_models = [
            'FIXED', 'BRR', 'BL',
            'BayesA', 'BayesB', 'BayesC', 'RKHS'
        ]

        bglr_models: "List[BGLR_MODELS]" = []
        bglr_names: "List[str]" = []
        if f"{self.name}_markers" in params:
            markers = params[f"{self.name}_markers"]
            assert markers in valid_models
            bglr_models.append(cast("BGLR_MODELS", markers))
            bglr_names.append("markers")

        if f"{self.name}_dists" in params:
            dists = params[f"{self.name}_dists"]
            assert dists in valid_models
            bglr_models.append(cast("BGLR_MODELS", dists))
            bglr_names.append("dists")

        if f"{self.name}_nonlinear" in params:
            nonlinear = params[f"{self.name}_nonlinear"]
            assert nonlinear in valid_models
            bglr_models.append(cast("BGLR_MODELS", nonlinear))
            bglr_names.append("nonlinear")

        if f"{self.name}_groups" in params:
            groups = params[f"{self.name}_groups"]
            assert groups in valid_models
            bglr_models.append(cast("BGLR_MODELS", groups))
            bglr_names.append("groups")

        if f"{self.name}_covariates" in params:
            covariates = params[f"{self.name}_covariates"]
            assert covariates in valid_models
            bglr_models.append(cast("BGLR_MODELS", covariates))
            bglr_names.append("covariates")

        if f"{self.name}_interactions" in params:
            interactions = params[f"{self.name}_interactions"]
            assert interactions in valid_models
            bglr_models.append(cast("BGLR_MODELS", interactions))
            bglr_names.append("interactions")
        response_type = params.get(f"{self.name}_response_type", "gaussian")
        assert response_type in ('gaussian', 'ordinal')

        r2 = params.get(f"{self.name}_r2", 0.5)
        assert isinstance(r2, float)

        model = BGLRRegressor(
            models=bglr_models,
            component_names=bglr_names,
            niter=10000,
            burnin=1000,
            response_type=cast(
                "Literal['gaussian', 'ordinal']",
                response_type
            ),
            R2=r2,
            random_state=self.rng.getrandbits(32),
            verbose=False,
        )
        return model

    def starting_points(self) -> "List[Params]":
        out: "List[Params]" = []
        return out

    def fit(
        self,
        params: "Params",
        Xs: "ODatasetIn",
        y: "Optional[npt.ArrayLike]" = None,
        **kwargs,
    ) -> "Optional[Model]":
        if isinstance(Xs, np.ndarray):
            X: ODatasetOut = np.asarray(Xs)
        elif isinstance(Xs, list):
            X = [np.asarray(xi) for xi in Xs]
        elif isinstance(Xs, dict):
            X = {k: np.asarray(xi) for k, xi in Xs.items()}
        else:
            raise ValueError("Invalid data")

        model = self.model(params)

        if model is None:
            return None

        model.fit(X, y)
        return model


class OptimiseConvMLP(OptimiseSK):

    def __init__(
        self,
        loss: "List[Literal['mse', 'mae', 'binary_crossentropy', 'pairwise']]",
        seed: "Optional[int]" = None,
        name: str = "conv_mlp",
    ):
        self.loss = loss
        self.rng = random.Random(seed)
        self.name = name
        return

    def _sample_conv(
        self,
        trial: "optuna.Trial",
        params: "Params",
        nfeatures: int,
    ) -> "Tuple[Params, int]":
        from math import floor

        if nfeatures < 30:
            max_nconv = 0
        elif nfeatures < 100:
            max_nconv = 1
        elif nfeatures < 200:
            max_nconv = 2
        else:
            max_nconv = 3

        conv_nlayers = trial.suggest_int(
            f"{self.name}_conv_nlayers",
            0,
            max_nconv
        )
        params[f"{self.name}_conv_nlayers"] = conv_nlayers

        if conv_nlayers < 1:
            return params, nfeatures

        def get_strides(
            nl: int,
            m: int,
            k: int,
            d: int
        ) -> int:
            for _ in range(nl):
                m = floor((m - k + d) / d)
            return m

        def get_max_strides(nl, m_min, m, k, dmax=4):
            """ I couldn't get a single function that
            dealt with the recurrence properly, so
            use a function.
            """
            if (m <= m_min) or (nl <= 0):
                return 1

            for di in range(dmax, 1, -1):
                mi = get_strides(nl, m, k, di)

                if mi >= m_min:
                    return di

            return 1

        min_features = 10
        if max_nconv > 0:
            max_kernel_size = floor(
                (nfeatures - min_features + max_nconv)
                / max_nconv
            )
            kernel_size = trial.suggest_int(
                f"{self.name}_conv_kernel_size",
                2,
                min([5, max_kernel_size])
            )
            params[f"{self.name}_conv_kernel_size"] = kernel_size

            max_strides = get_max_strides(
                conv_nlayers,
                min_features,
                nfeatures,
                kernel_size,
                dmax=4
            )
            stride_size = trial.suggest_int(
                f"{self.name}_conv_strides",
                1,
                max_strides
            )
            params[f"{self.name}_conv_strides"] = stride_size

            nfeatures = get_strides(
                conv_nlayers,
                nfeatures,
                kernel_size,
                stride_size
            )

            params[f"{self.name}_conv_activation"] = trial.suggest_categorical(
                f"{self.name}_conv_activation",
                ["linear", "relu"]
            )

        return params, nfeatures

    def _sample_adaptive(
        self,
        trial: "optuna.Trial",
        params: "Params",
    ) -> "Params":
        use_adaptive = trial.suggest_categorical(
            f"{self.name}_adaptive",
            [True, False]
        )
        params[f"{self.name}_adaptive"] = use_adaptive

        if not use_adaptive:
            return params

        params[f"{self.name}_adaptive_l1_rate"] = trial.suggest_float(
            f"{self.name}_adaptive_l1_rate",
            1e-50,
            1,
            log=True
        )

        params[f"{self.name}_adaptive_l2_rate"] = trial.suggest_float(
            f"{self.name}_adaptive_l2_rate",
            1e-50,
            1,
            log=True
        )
        return params

    def _sample_embed(
        self,
        trial: "optuna.Trial",
        params: "Params",
        nfeatures: int,
        final_units: "Optional[int]",
        min_units: int,
        max_units: int,
        min_nlayers: int,
        max_nlayers: int,
        name: str
    ) -> "Tuple[Params, int]":
        if nfeatures < 1:
            return params, nfeatures

        nlayers = trial.suggest_int(
            f"{self.name}_{name}_nlayers",
            min_nlayers,
            max_nlayers
        )
        params[f"{self.name}_{name}_nlayers"] = nlayers

        if nlayers == 0:
            assert final_units is None

        if (nlayers == 1) and (final_units is not None):
            min_units = final_units
            max_units = final_units
        else:
            max_units = min([nfeatures, max_units])

        if min_units > max_units:
            min_units = max_units

        nunits = trial.suggest_int(
            f"{self.name}_{name}_nunits",
            min_units,
            max_units
        )
        params[f"{self.name}_{name}_nunits"] = nunits

        params[f"{self.name}_{name}_0_dropout_rate"] = trial.suggest_float(
            f"{self.name}_{name}_0_dropout_rate",
            0.0,
            0.9
        )

        if nlayers > 1:
            params[f"{self.name}_{name}_1_dropout_rate"] = trial.suggest_float(
                f"{self.name}_{name}_1_dropout_rate",
                0.0,
                0.9
            )

        params[f"{self.name}_{name}_activation"] = trial.suggest_categorical(
            f"{self.name}_{name}_activation",
            ["linear", "relu"]
        )

        params[f"{self.name}_{name}_residual"] = trial.suggest_categorical(
            f"{self.name}_{name}_residual",
            [True, False]
        )

        if final_units is None:
            outfeatures: int = nunits
        else:
            outfeatures = final_units

        return params, outfeatures

    def sample(
        self,
        trial: "optuna.Trial",
        Xs: "Iterable[Optional[npt.ArrayLike]]",
        **kwargs
    ) -> "Params":
        from selectml.higher import or_else, fmap

        first = True
        groups: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "groups",
            [None for _ in Xs]
        )
        covariates: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "covariates",
            [None for _ in Xs]
        )
        dists: "List[Optional[npt.ArrayLike]]" = kwargs.get(
            "dists",
            [None for _ in Xs]
        )

        if any([X is None for X in Xs]):
            nfeatures = 0
            nsamples = 0
            onehot_nfeatures = 0
            onehot_nfeatures = 0
            dist_nfeatures = 0
            group_nfeatures = 0
            covariate_nfeatures = 0
        else:
            dists_none = [x is None for x in dists]
            if any(dists_none):
                assert all(dists_none)

            groups_none = [x is None for x in groups]
            if any(groups_none):
                assert all(groups_none)

            covariates_none = [x is None for x in covariates]
            if any(covariates_none):
                assert all(covariates_none)

            for X, dist, group, covariate in zip(
                Xs,
                dists,
                groups,
                covariates
            ):
                X = np.array(X)
                this_nsamples = X.shape[0]
                this_nfeatures = X.shape[1]
                this_onehot_nfeatures = np.sum(ndistinct(X))

                this_dist_nfeatures = or_else(0, fmap(
                    lambda h: np.asarray(h).shape[1],
                    dist
                ))
                this_group_nfeatures = or_else(0, fmap(
                    lambda h: np.asarray(h).shape[1],
                    group
                ))
                this_covariate_nfeatures = or_else(0, fmap(
                    lambda h: np.asarray(h).shape[1],
                    covariate
                ))

                if first:
                    first = False
                    nsamples = this_nsamples
                    nfeatures = this_nfeatures
                    onehot_nfeatures = this_onehot_nfeatures
                    dist_nfeatures = this_dist_nfeatures
                    group_nfeatures = this_group_nfeatures
                    covariate_nfeatures = this_covariate_nfeatures
                else:
                    nsamples = min([nsamples, this_nsamples])
                    nfeatures = min([nfeatures, this_nfeatures])
                    onehot_nfeatures = min([
                        onehot_nfeatures,
                        this_onehot_nfeatures
                    ])
                    dist_nfeatures = min([
                        dist_nfeatures,
                        this_dist_nfeatures
                    ])
                    group_nfeatures = min([
                        group_nfeatures,
                        this_group_nfeatures
                    ])
                    covariate_nfeatures = min([
                        covariate_nfeatures,
                        this_covariate_nfeatures
                    ])

        params = self.sample_params(
            trial,
            nsamples,
            nfeatures,
            onehot_nfeatures=onehot_nfeatures,
            ndists=dist_nfeatures,
            ngroups=group_nfeatures,
            ncovariates=covariate_nfeatures,
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
        params: Params = {}

        params, nfeatures = self._sample_conv(
            trial,
            params,
            nfeatures
        )

        params = self._sample_adaptive(
            trial,
            params,
        )

        ngroups = kwargs.get("ngroups", 0)
        ncovariates = kwargs.get("ncovariates", 0)
        ndists = kwargs.get("ndists", 0)

        combine_options = ["add", "concatenate"]

        combine_method = trial.suggest_categorical(
            f"{self.name}_combine_method",
            combine_options
        )
        params[f"{self.name}_combine_method"] = combine_method

        min_embed_nunits = 5
        max_embed_nunits = 100
        min_embed_nlayers = 1
        max_embed_nlayers = 10

        if combine_method == "add":
            final_nunits = trial.suggest_int(
                f"{self.name}_final_nunits",
                min_embed_nunits,
                max_embed_nunits
            )
            params[f"{self.name}_final_nunits"] = final_nunits
        else:
            final_nunits = None

        params, nmarkers = self._sample_embed(
            trial,
            params,
            nfeatures,
            final_nunits,
            min_embed_nunits,
            max_embed_nunits,
            min_embed_nlayers,
            max_embed_nlayers,
            name="marker_embed"
        )

        params, ndists = self._sample_embed(
            trial,
            params,
            ndists,
            final_nunits,
            min_embed_nunits,
            max_embed_nunits,
            min_embed_nlayers,
            max_embed_nlayers,
            name="dist_embed"
        )

        params, ngroups = self._sample_embed(
            trial,
            params,
            ngroups,
            final_nunits,
            min_embed_nunits,
            max_embed_nunits,
            min_embed_nlayers,
            max_embed_nlayers,
            name="group_embed"
        )

        params, ncovariates = self._sample_embed(
            trial,
            params,
            ncovariates,
            final_nunits,
            min_embed_nunits,
            max_embed_nunits,
            min_embed_nlayers,
            max_embed_nlayers,
            name="covariate_embed"
        )

        if combine_method == "add":
            assert final_nunits is not None
            nfeatures = final_nunits
        else:
            nfeatures = nmarkers + ndists + ngroups + ncovariates

        assert nfeatures > 0

        params, nembed = self._sample_embed(
            trial,
            params,
            nfeatures,
            None,
            min_embed_nunits,
            max_embed_nunits,
            0,
            max_embed_nlayers,
            name="post_embed"
        )

        selfsupervised_loss = trial.suggest_categorical(
            f"{self.name}_selfsupervised_loss",
            ["none", "semihard_triplet", "hard_triplet"]
        )
        params[f"{self.name}_selfsupervised_loss"] = selfsupervised_loss

        if selfsupervised_loss != "none":
            params[f"{self.name}_{selfsupervised_loss}_rate"] = trial.suggest_float(  # noqa: E501
                f"{self.name}_{selfsupervised_loss}_rate",
                1e-4,
                1
            )

        params[f"{self.name}_predictor_dropout_rate"] = trial.suggest_float(
            f"{self.name}_predictor_dropout_rate",
            0.0,
            0.9,
        )

        # For the kind of data we have, i've rarely seen it need to go
        # further than 200 epochs
        params[f"{self.name}_nepochs"] = trial.suggest_int(
            f"{self.name}_nepochs",
            10,
            200
        )

        params[f"{self.name}_learning_rate"] = trial.suggest_float(
            f"{self.name}_learning_rate",
            1e-6,
            1,
            log=True,
        )

        params[f"{self.name}_loss"] = trial.suggest_categorical(
            f"{self.name}_loss",
            self.loss
        )

        return params

    def model(
        self,
        params: "Params",
        **kwargs
    ) -> "Optional[Model]":
        from .wrapper import ConvMLPClassifier, ConvMLPRegressor, ConvMLPRanker

        loss = params[f"{self.name}_loss"]

        if loss in ("mse", "mae"):
            cls = ConvMLPRegressor
        elif loss == "binary_crossentropy":
            cls = ConvMLPClassifier
        elif loss == "pairwise":
            cls = ConvMLPRanker
        else:
            raise ValueError("Got invalid loss")

        inputs: "List[Literal['markers', 'dists', 'groups', 'covariates']]" = []  # noqa: E501

        me_nlayers = params.get(f"{self.name}_marker_embed_nlayers", 0)
        assert isinstance(me_nlayers, int)
        if me_nlayers > 0:
            inputs.append("markers")

        de_nlayers = params.get(f"{self.name}_dist_embed_nlayers", 0)
        assert isinstance(de_nlayers, int)
        if de_nlayers > 0:
            inputs.append("dists")

        ge_nlayers = params.get(f"{self.name}_group_embed_nlayers", 0)
        assert isinstance(ge_nlayers, int)
        if ge_nlayers > 0:
            inputs.append("groups")

        ce_nlayers = params.get(f"{self.name}_covariate_embed_nlayers", 0)
        assert isinstance(ce_nlayers, int)
        if ce_nlayers > 0:
            inputs.append("covariates")

        model = cls(
            loss=loss,
            optimizer__learning_rate=params.get(f"{self.name}_learning_rate", 0.001),  # noqa: E501
            epochs=params.get(f"{self.name}_nepochs", 100),
            verbose=0,
            conv_nlayers=params.get(f"{self.name}_conv_nlayers", 0),
            conv_filters=params.get(f"{self.name}_conv_filters", 1),
            conv_strides=params.get(f"{self.name}_conv_strides", 1),
            conv_kernel_size=params.get(f"{self.name}_conv_kernel_size", 2),
            conv_activation=params.get(f"{self.name}_conv_activation", "linear"),  # noqa: E501
            conv_use_batchnorm=True,
            adaptive_l1=params.get(f"{self.name}_adaptive", False),
            adaptive_l1_rate=params.get(f"{self.name}_adaptive_l1_rate", 0.0),
            adaptive_l2_rate=params.get(f"{self.name}_adaptive_l1_rate", 0.0),
            marker_embed_nlayers=params.get(f"{self.name}_marker_embed_nlayers", 1),  # noqa: E501
            marker_embed_residual=params.get(f"{self.name}_marker_embed_residual", False),  # noqa: E501
            marker_embed_nunits=params.get(f"{self.name}_marker_embed_nunits", 2),  # noqa: E501
            marker_embed_final_nunits=params.get(f"{self.name}_final_nunits", None),  # noqa: E501
            marker_embed_0_dropout_rate=params.get(f"{self.name}_marker_embed_0_dropout_rate", 0.0),  # noqa: E501
            marker_embed_1_dropout_rate=params.get(f"{self.name}_marker_embed_1_dropout_rate", 0.0),  # noqa: E501
            marker_embed_activation=params.get(f"{self.name}_marker_embed_activation", "linear"),  # noqa: E501
            dist_embed_nlayers=params.get(f"{self.name}_dist_embed_nlayers", 0),  # noqa: E501
            dist_embed_residual=params.get(f"{self.name}_dist_embed_residual", False),  # noqa: E501
            dist_embed_nunits=params.get(f"{self.name}_dist_embed_nunits", 2),  # noqa: E501
            dist_embed_final_nunits=params.get(f"{self.name}_final_nunits", None),  # noqa: E501
            dist_embed_0_dropout_rate=params.get(f"{self.name}_dist_embed_0_dropout_rate", 0.0),  # noqa: E501
            dist_embed_1_dropout_rate=params.get(f"{self.name}_dist_embed_1_dropout_rate", 0.0),  # noqa: E501
            dist_embed_activation=params.get(f"{self.name}_dist_embed_activation", "linear"),  # noqa: E501
            group_embed_nlayers=params.get(f"{self.name}_group_embed_nlayers", 0),  # noqa: E501
            group_embed_residual=params.get(f"{self.name}_group_embed_residual", False),  # noqa: E501
            group_embed_nunits=params.get(f"{self.name}_group_embed_nunits", 2),  # noqa: E501
            group_embed_final_nunits=params.get(f"{self.name}_final_nunits", None),  # noqa: E501
            group_embed_0_dropout_rate=params.get(f"{self.name}_group_embed_0_dropout_rate", 0.0),  # noqa: E501
            group_embed_1_dropout_rate=params.get(f"{self.name}_group_embed_1_dropout_rate", 0.0),  # noqa: E501
            group_embed_activation=params.get(f"{self.name}_group_embed_activation", "linear"),  # noqa: E501
            covariate_embed_nlayers=params.get(f"{self.name}_covariate_embed_nlayers", 0),  # noqa: E501
            covariate_embed_residual=params.get(f"{self.name}_covariate_embed_residual", False),  # noqa: E501
            covariate_embed_nunits=params.get(f"{self.name}_covariate_embed_nunits", 2),  # noqa: E501
            covariate_embed_final_nunits=params.get(f"{self.name}_final_nunits", None),  # noqa: E501
            covariate_embed_0_dropout_rate=params.get(f"{self.name}_covariate_embed_0_dropout_rate", 0.0),  # noqa: E501
            covariate_embed_1_dropout_rate=params.get(f"{self.name}_covariate_embed_1_dropout_rate", 0.0),  # noqa: E501
            covariate_embed_activation=params.get(f"{self.name}_covariate_embed_activation", "linear"),  # noqa: E501
            combine_method=params.get(f"{self.name}_combine_method", "concatenate"),  # noqa: E501
            post_embed_nlayers=params.get(f"{self.name}_post_embed_nlayers", 0),  # noqa: E501
            post_embed_residual=params.get(f"{self.name}_post_embed_residual", False),  # noqa: E501
            post_embed_nunits=params.get(f"{self.name}_post_embed_nunits", 2),
            post_embed_0_dropout_rate=params.get(f"{self.name}_post_embed_0_dropout_rate", 0.0),  # noqa: E501
            post_embed_1_dropout_rate=params.get(f"{self.name}_post_embed_1_dropout_rate", 0.0),  # noqa: E501
            post_embed_activation=params.get(f"{self.name}_post_embed_activation", "linear"),  # noqa: E501
            predictor_dropout_rate=params.get(f"{self.name}_predictor_dropout_rate", 0.0),  # noqa: E501
            hard_triplet_loss_rate=params.get(f"{self.name}_hard_triplet_loss_rate", None),  # noqa: E501
            semihard_triplet_loss_rate=params.get(f"{self.name}_semihard_triplet_loss_rate", None),  # noqa: E501
            input_names=inputs,
            name=f"{self.name}",
        )
        return model

    def starting_points(self) -> "List[Params]":
        out: "List[Params]" = []
        return out

    def fit(
        self,
        params: "Params",
        Xs: "ODatasetIn",
        y: "Optional[npt.ArrayLike]" = None,
        **kwargs,
    ) -> "Optional[Model]":
        if isinstance(Xs, np.ndarray):
            X: ODatasetOut = np.asarray(Xs)
        elif isinstance(Xs, list):
            X = [np.asarray(xi) for xi in Xs]
        elif isinstance(Xs, dict):
            X = {k: np.asarray(xi) for k, xi in Xs.items()}
        else:
            raise ValueError("Invalid data")

        model = self.model(params)

        if model is None:
            return None

        model.fit(X, y)
        return model
