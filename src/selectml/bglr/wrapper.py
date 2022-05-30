from os.path import join as pjoin
from tempfile import TemporaryDirectory

from dataclasses import dataclass
from ..higher import fmap, or_else
import numpy as np

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils import check_array, check_random_state

from rpy2 import robjects as ro
from rpy2 import rinterface as ri


from typing import Literal
from typing import Sequence, Tuple, List, Dict
from typing import Iterable
from typing import Union, Optional, Any
from typing import Callable
from typing import Type, TypeVar
from typing import cast
from numpy import typing as npt

T = TypeVar("T")
BGLR_MODELS = Literal[
    'FIXED', 'BRR', 'BL',
    'BayesA', 'BayesB', 'BayesC',
    'RKHS'
]


def get_str(vector: "ro.vectors.Vector") -> str:
    assert not isinstance(vector, ri.sexp.NULLType), vector

    if len(vector) > 1:
        raise ValueError(f"Expected a single value. {vector}")

    if not isinstance(vector, ro.vectors.StrVector):
        raise ValueError(f"Expected a string vector as input. {vector}")

    return str(vector[0])


def get_int(vector: "ro.vectors.Vector") -> int:
    assert not isinstance(vector, ri.sexp.NULLType), vector
    if len(vector) > 1:
        raise ValueError(f"Expected a single value. {vector}")

    if not isinstance(vector, ro.vectors.IntVector):
        raise ValueError(f"Expected a integer vector as input. {vector}")

    return int(vector[0])


def get_float(vector: "ro.vectors.Vector") -> float:
    assert not isinstance(vector, ri.sexp.NULLType), vector
    if len(vector) > 1:
        raise ValueError(f"Expected a single value. {vector}")

    if not isinstance(vector, ro.vectors.FloatVector):
        raise ValueError(f"Expected a float vector as input. {vector}")

    return float(vector[0])


def get_int_or_float(vector: "ro.vectors.Vector") -> "float":
    assert not isinstance(vector, ri.sexp.NULLType), vector
    if len(vector) > 1:
        raise ValueError(f"Expected a single value. {vector}")

    if not isinstance(vector, (ro.vectors.FloatVector, ro.vectors.IntVector)):
        raise ValueError(f"Expected an int or float vector as input. {vector}")

    return float(vector[0])


def get_float_array(vector: "ro.vectors.Vector") -> np.ndarray:
    assert not isinstance(vector, ri.sexp.NULLType), vector

    if not isinstance(vector, ro.vectors.FloatVector):
        raise ValueError(f"Expected a float vector as input. {vector}")

    return np.asarray(vector)


def get_optional(
    fn: "Callable[[ro.vectors.Vector], T]",
    vector: "Union[ri.sexp.NULLType]"
) -> "Optional[T]":
    if isinstance(vector, ri.sexp.NULLType):
        return None
    else:
        return fn(vector)


class BGLRResult:

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def from_r(self, lv: ro.ListVector) -> "BGLRResult":

        switch: "Dict[str, Type[BGLRResult]]" = {
            "FIXED": FixedResult,
            "BRR": BRRResult,
            "BL": BLResult,
            "BayesA": BayesAResult,
            "BayesB": BayesBResult,
            "BayesC": BayesCResult,
            "RKHS": RKHSResult,
        }

        model = get_str(lv.rx2("model"))
        if model not in switch:
            raise ValueError(
                f"Model {model} does not correspond to one of the "
                "known BGLR models."
            )

        return switch[model].from_r(lv)

    @staticmethod
    def _get_name(lv: ro.ListVector) -> str:
        name = get_str(lv.rx2("Name"))
        return name[len("ETA_"):]

    @staticmethod
    def _check_expected_model(lv: ro.ListVector, model: str):
        this_model = get_str(lv.rx2("model"))
        if this_model != model:
            raise ValueError(
                f"Expected to get {model}, but got results for {this_model}."
            )


@dataclass
class FixedResult(BGLRResult):

    name: str
    p: int
    b: np.ndarray
    varB: float
    SD_b: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.matmul(X, self.b)

    @classmethod
    def from_r(cls, lv: ro.ListVector):
        cls._check_expected_model(lv, "FIXED")
        return cls(
            name=cls._get_name(lv),
            p=get_int(lv.rx2("p")),
            b=get_float_array(lv.rx2("b")),
            varB=get_float(lv.rx2("varB")),
            SD_b=get_float_array(lv.rx2("SD.b")),
        )


@dataclass
class BRRResult(BGLRResult):

    name: str
    p: int
    df0: float
    R2: float
    MSx: float
    S0: float
    b: np.ndarray
    varB: float
    SD_b: np.ndarray
    SD_varB: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.matmul(X, self.b)

    @classmethod
    def from_r(cls, lv: ro.ListVector):
        cls._check_expected_model(lv, "BRR")
        return cls(
            name=cls._get_name(lv),
            p=get_int(lv.rx2("p")),
            df0=get_float(lv.rx2("df0")),
            R2=get_float(lv.rx2("R2")),
            MSx=get_float(lv.rx2("MSx")),
            S0=get_float(lv.rx2("S0")),
            b=get_float_array(lv.rx2("b")),
            varB=get_float(lv.rx2("varB")),
            SD_b=get_float_array(lv.rx2("SD.b")),
            SD_varB=get_float(lv.rx2("SD.varB")),
        )


@dataclass
class BLResult(BGLRResult):

    name: str
    minAbsBeta: float
    p: int
    MSx: float
    R2: float
    lambda_: float
    type: str
    shape: float
    rate: float
    b: np.ndarray
    tau2: np.ndarray
    SD_b: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.matmul(X, self.b)

    @classmethod
    def from_r(cls, lv: ro.ListVector):
        cls._check_expected_model(lv, "BL")
        return cls(
            name=cls._get_name(lv),
            minAbsBeta=get_float(lv.rx2("minAbsBeta")),
            p=get_int(lv.rx2("p")),
            MSx=get_float(lv.rx2("MSx")),
            R2=get_float(lv.rx2("R2")),
            lambda_=get_float(lv.rx2("lambda")),
            type=get_str(lv.rx2("type")),
            shape=get_float(lv.rx2("shape")),
            rate=get_float(lv.rx2("rate")),
            b=get_float_array(lv.rx2("b")),
            tau2=get_float_array(lv.rx2("tau2")),
            SD_b=get_float_array(lv.rx2("SD.b")),
        )


@dataclass
class BayesAResult(BGLRResult):

    name: str
    p: int
    MSx: float
    df0: float
    R2: float
    S0: float
    shape0: float
    rate0: float
    S: float
    b: np.ndarray
    varB: np.ndarray
    SD_b: np.ndarray
    SD_varB: np.ndarray
    SD_S: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.matmul(X, self.b)

    @classmethod
    def from_r(cls, lv: ro.ListVector):
        cls._check_expected_model(lv, "BayesA")
        return cls(
            name=cls._get_name(lv),
            p=get_int(lv.rx2("p")),
            MSx=get_float(lv.rx2("MSx")),
            df0=get_float(lv.rx2("df0")),
            R2=get_float(lv.rx2("R2")),
            S0=get_float(lv.rx2("S0")),
            shape0=get_float(lv.rx2("shape0")),
            rate0=get_float(lv.rx2("rate0")),
            S=get_float(lv.rx2("S")),
            b=get_float_array(lv.rx2("b")),
            varB=get_float_array(lv.rx2("varB")),
            SD_b=get_float_array(lv.rx2("SD.b")),
            SD_varB=get_float_array(lv.rx2("SD.varB")),
            SD_S=get_float(lv.rx2("SD.S")),
        )


@dataclass
class BayesBResult(BGLRResult):

    name: str
    p: int
    MSx: float
    R2: float
    df0: float
    probIn: float
    counts: float
    countsIn: float
    countsOut: float
    S0: float
    b: np.ndarray
    d: np.ndarray
    shape0: float
    rate0: float
    S: float
    varB: np.ndarray
    SD_b: np.ndarray
    SD_varB: np.ndarray
    SD_probIn: float
    SD_S: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.matmul(X, self.b)

    @classmethod
    def from_r(cls, lv: ro.ListVector):
        cls._check_expected_model(lv, "BayesB")
        return cls(
            name=cls._get_name(lv),
            p=get_int(lv.rx2("p")),
            MSx=get_float(lv.rx2("MSx")),
            R2=get_float(lv.rx2("R2")),
            df0=get_float(lv.rx2("df0")),
            probIn=get_float(lv.rx2("probIn")),
            counts=get_float(lv.rx2("counts")),
            countsIn=get_float(lv.rx2("countsIn")),
            countsOut=get_float(lv.rx2("countsOut")),
            S0=get_float(lv.rx2("S0")),
            b=get_float_array(lv.rx2("b")),
            d=get_float_array(lv.rx2("d")),
            shape0=get_float(lv.rx2("shape0")),
            rate0=get_float(lv.rx2("rate0")),
            S=get_float(lv.rx2("S")),
            varB=get_float_array(lv.rx2("varB")),
            SD_b=get_float_array(lv.rx2("SD.b")),
            SD_varB=get_float_array(lv.rx2("SD.varB")),
            SD_probIn=get_float(lv.rx2("SD.probIn")),
            SD_S=get_float(lv.rx2("SD.S")),
        )


@dataclass
class BayesCResult(BGLRResult):

    name: str
    p: int
    MSx: float
    R2: float
    df0: float
    probIn: float
    counts: float
    countsIn: float
    countsOut: float
    S0: float
    b: np.ndarray
    d: np.ndarray
    varB: np.ndarray
    SD_b: np.ndarray
    SD_varB: np.ndarray
    SD_probIn: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.matmul(X, self.b)

    @classmethod
    def from_r(cls, lv: ro.ListVector):
        cls._check_expected_model(lv, "BayesC")
        return cls(
            name=cls._get_name(lv),
            p=get_int(lv.rx2("p")),
            MSx=get_float(lv.rx2("MSx")),
            R2=get_float(lv.rx2("R2")),
            df0=get_float(lv.rx2("df0")),
            probIn=get_float(lv.rx2("probIn")),
            counts=get_float(lv.rx2("counts")),
            countsIn=get_float(lv.rx2("countsIn")),
            countsOut=get_float(lv.rx2("countsOut")),
            S0=get_float(lv.rx2("S0")),
            b=get_float_array(lv.rx2("b")),
            d=get_float_array(lv.rx2("d")),
            varB=get_float_array(lv.rx2("varB")),
            SD_b=get_float_array(lv.rx2("SD.b")),
            SD_varB=get_float_array(lv.rx2("SD.varB")),
            SD_probIn=get_float(lv.rx2("SD.probIn")),
        )


@dataclass
class RKHSResult(BGLRResult):

    name: str
    K: np.ndarray
    K_inv: np.ndarray
    V: np.ndarray
    d: np.ndarray
    tolD: float
    levelsU: int
    df0: float
    R2: float
    S0: float
    u: np.ndarray
    varU: float
    uStar: np.ndarray
    SD_u: np.ndarray
    SD_varU: float

    @classmethod
    def from_r(cls, lv: ro.ListVector):
        cls._check_expected_model(lv, "RKHS")
        K = get_float_array(lv.rx2("K"))
        return cls(
            name=cls._get_name(lv),
            K=K,
            K_inv=np.linalg.pinv(K),
            V=get_float_array(lv.rx2("V")),
            d=get_float_array(lv.rx2("d")),
            tolD=get_float(lv.rx2("tolD")),
            levelsU=get_int(lv.rx2("levelsU")),
            df0=get_float(lv.rx2("df0")),
            R2=get_float(lv.rx2("R2")),
            S0=get_float(lv.rx2("S0")),
            u=get_float_array(lv.rx2("u")),
            varU=get_float(lv.rx2("varU")),
            uStar=get_float_array(lv.rx2("uStar")),
            SD_u=get_float_array(lv.rx2("SD.u")),
            SD_varU=get_float(lv.rx2("SD.varU")),
        )

    def predict(self, X: np.ndarray):
        return X.dot(self.K_inv).dot(self.u)


class BGLRRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        models: "Union[BGLR_MODELS, Sequence[BGLR_MODELS]]",
        component_prediction: "Optional[Union[Sequence[str], Sequence[int], Sequence[bool]]]" = None,  # noqa: E501
        component_names: "Optional[Union[str, Sequence[str]]]" = None,
        niter: int = 1500,
        burnin: int = 500,
        thin: int = 5,
        saveat: str = "",
        response_type: "Literal['gaussian', 'ordinal']" = "gaussian",
        S0: "Optional[float]" = None,
        df0: int = 5,
        R2: float = 0.5,
        verbose: bool = True,
        rm_existing_files: bool = True,
        random_state: "Optional[int]" = None,
        name: str = "bglr"
    ):
        if isinstance(models, str):
            models_ = [models]
        else:
            models_ = list(models)

        self.name = name

        assert len(models) > 0, "We need to know which models you want to use."

        if isinstance(component_names, str):
            component_names_: "Union[List[str], List[int]]" = [component_names]
        elif component_names is None:
            component_names_ = list(range(len(models_)))
        else:
            component_names_ = list(component_names)

        assert len(component_names_) == len(models_)

        component_prediction_ = self.check_component_prediction_type(
            models_,
            component_names_,
            component_prediction
        )

        self.models = models_
        self.component_prediction = component_prediction_
        self.component_names = component_names_

        assert niter > 0
        self.niter = niter

        assert burnin >= 0
        self.burnin = burnin

        assert thin >= 0
        self.thin = thin
        self.saveat = saveat
        self.response_type = response_type
        self.S0 = S0

        assert df0 > 0
        self.df0 = df0

        assert 0.0 < R2 < 1.0, "R2 must be between 0 and 1 (exclusive)."
        self.R2 = R2
        self.verbose = verbose
        self.rm_existing_files = rm_existing_files
        self.random_state = random_state
        return

    @staticmethod
    def check_component_prediction_type(
        models: "Sequence[BGLR_MODELS]",
        component_names: "Union[List[str], List[int]]",
        component_prediction: "Optional[Union[Sequence[str], Sequence[int], Sequence[bool]]]",  # noqa: E501
    ) -> "Union[List[str], List[int]]":
        type_: "Optional[Literal['str', 'int', 'bool']]" = None

        if component_prediction is None:
            component_prediction_: "Union[List[str], List[bool], List[int]]" = component_names  # noqa: E501
        elif isinstance(component_prediction, (bool, int, str)):
            component_prediction_ = [component_prediction]
        else:
            component_prediction_ = cast(
                "Union[List[str], List[bool], List[int]]",
                list(component_prediction)
            )

        is_bool = [isinstance(p, bool) for p in component_prediction_]
        is_str = [isinstance(p, str) for p in component_prediction_]
        is_int = [isinstance(p, int) for p in component_prediction_]

        if any(is_bool) and not all(is_bool):
            raise ValueError(
                "Some of the component_prediction elements are booleans, "
                "but not all are."
            )
        elif all(is_bool) and not (len(component_prediction_) == len(models)):
            raise ValueError(
                "component_prediction contains boolean elements, "
                "but is not the same length as models."
            )
        elif all(is_bool):
            type_ = "bool"

        if any(is_str) and not all(is_str):
            raise ValueError(
                "Some of the component_prediction elements are strings, "
                "but not all are."
            )
        elif all(is_str):
            type_ = "str"

        if any(is_int) and not all(is_int):
            raise ValueError(
                "Some of the component_prediction elements are integers, "
                "but not all are."
            )
        elif all(is_int):
            type_ = "int"

        if type_ == "bool":
            component_prediction2_ = cast("Union[List[str], List[int]]", [
                n
                for n, b
                in zip(
                    component_names,
                    cast("List[bool]", component_prediction_)
                )
                if b
            ])
        else:
            component_prediction2_ = cast(
                "Union[List[str], List[int]]",
                component_prediction_
            )

        assert len(component_prediction2_) > 0
        return component_prediction2_

    def _reset(self):
        if hasattr(self, "eta_"):
            del self.yndim_
            del self.intercept_
            del self.sd_intercept_
            del self.S0_
            del self.df0_
            del self.vare_
            del self.df_vare_
            del self.log_likelihood_at_post_mean_
            del self.post_mean_log_lik_
            del self.pd_
            del self.dic_
            del self.eta
        return self

    def fit(
        self,
        X: "Union[npt.ArrayLike, List[npt.ArrayLike], Tuple[npt.ArrayLike, ...], Dict[str, npt.ArrayLike]]",  # noqa: E501
        y: "npt.ArrayLike",
        sample_weight: "Optional[npt.ArrayLike]" = None,
        groups: "Optional[npt.ArrayLike]" = None
    ):
        self._reset()
        self.random_state_ = check_random_state(self.random_state)

        y_ = self._check_y(
            np.asarray(y),
            y_numeric=True,
            reset=True,
        )
        self.yndim_ = y_.ndim

        X_ = self._check_X(X, reset=True)
        self._check_X_y(X_, y_)

        results = self._fit_bglr(
            X_,
            y_,
            fmap(np.asarray, sample_weight),
            fmap(np.asarray, groups)
        )
        self.intercept_ = get_float(results.rx2("mu"))
        self.sd_intercept_ = get_float(results.rx2("SD.mu"))
        self.S0_ = get_float(results.rx2("S0"))
        self.df0_ = get_int_or_float(results.rx2("df0"))
        self.vare_ = get_float(results.rx2("varE"))
        self.sd_vare_ = get_float(results.rx2("varE"))
        self.log_likelihood_at_post_mean_ = get_float(
            results.rx2("fit").rx2("logLikAtPostMean")
        )
        self.post_mean_log_lik_ = get_float(
            results.rx2("fit").rx2("postMeanLogLik")
        )
        self.pd_ = get_float(results.rx2("fit").rx2("pD"))
        self.dic_ = get_float(results.rx2("fit").rx2("DIC"))
        self.eta_ = {
            name: BGLRResult.from_r(result)
            for name, result
            in zip(self.component_names, results.rx2("ETA"))
        }
        # self.yhat = get_float_array(results.rx2("yHat"))
        return self

    def _fit_bglr(
        self,
        X: "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]",  # noqa: E501
        y: "np.ndarray",
        sample_weight: "Optional[np.ndarray]" = None,
        groups: "Optional[np.ndarray]" = None,
    ) -> ro.ListVector:
        import rpy2.robjects as ro
        import rpy2.robjects.vectors as rov
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()

        if isinstance(X, dict):
            generator: "List[Tuple[BGLR_MODELS, Union[int, str], np.ndarray]]" = [  # noqa: E501
                (m, str(k), X[str(k)])
                for m, k
                in zip(self.models, self.component_names)
            ]
        elif isinstance(X, list):
            generator = list(zip(self.models, self.component_names, X))
        else:
            assert isinstance(X, np.ndarray)
            generator = [(self.models[0], self.component_names[0], X)]

        y_ = np.asarray(y)

        BGLR = importr("BGLR")
        base = importr("base")

        base.set_seed(self.random_state_.randint(0, 2**16))

        if all(isinstance(i, int) for _, i, _ in generator):
            ETA: "Union[List[ro.ListVector], Dict[str, ro.ListVector]]" = []
            assert isinstance(ETA, list)
            for mtype, name, data in generator:
                if mtype == "RKHS":
                    ETA.append(ro.ListVector({"K": data, "model": mtype}))
                else:
                    ETA.append(ro.ListVector({"X": data, "model": mtype}))
        else:
            assert all(isinstance(i, str) for _, i, _ in generator)
            ETA = {}
            for mtype, name, data in generator:
                assert isinstance(name, str)
                if mtype == "RKHS":
                    ETA[name] = ro.ListVector({"K": data, "model": mtype})
                else:
                    ETA[name] = ro.ListVector({"X": data, "model": mtype})

        # TODO: read the output files to allow trace introspection
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
            results = BGLR.BGLR(
                y=y_,
                response_type=self.response_type,
                ETA=ro.ListVector(ETA) if isinstance(ETA, dict) else ETA,
                nIter=self.niter,
                burnIn=self.burnin,
                thin=self.thin,
                saveAt=pjoin(tmpdirname, self.saveat),
                S0=or_else(ro.NULL, self.S0),
                df0=self.df0,
                R2=self.R2,
                weights=or_else(ro.NULL, fmap(np.asarray, sample_weight)),
                verbose=self.verbose,
                rmExistingFiles=self.rm_existing_files,
                groups=or_else(
                    ro.NULL,
                    fmap(rov.FactorVector, fmap(np.asarray, groups))
                ),
            )
        numpy2ri.deactivate()

        return results

    def predict(
        self,
        X
    ):
        X_ = self._check_X(X, reset=False)
        yhat = self.intercept_

        if isinstance(X_, dict):
            generator: "Iterable[Union[Tuple[str, np.ndarray], Tuple[int, np.ndarray]]]" = (  # noqa: E501
                (k, X_[k]) for k in self.component_names
            )
        elif isinstance(X_, list):
            generator = zip(self.component_names, X_)
        else:
            assert isinstance(X_, np.ndarray)
            generator = [(self.component_names[0], X_)]

        for name, Xi in generator:
            if name not in self.component_prediction:
                continue

            component = self.eta_[name]
            yhat += component.predict(Xi)

        if (yhat.ndim == 1) and (self.yndim_ == 2):
            yhat = yhat.reshape((-1, 1))
        elif self.yndim_ == 1:
            yhat = yhat.ravel()

        return yhat

    def get_params(self, deep: bool = True) -> "Dict[str, Any]":
        from copy import deepcopy
        return dict(
            models=deepcopy(self.models),
            component_prediction=deepcopy(self.component_prediction),
            component_names=deepcopy(self.component_names),
            niter=self.niter,
            burnin=self.burnin,
            thin=self.thin,
            saveat=self.saveat,
            response_type=self.response_type,
            S0=self.S0,
            df0=self.df0,
            R2=self.R2,
            verbose=self.verbose,
            rm_existing_files=self.rm_existing_files,
            random_state=self.random_state,
            name=self.name
        )

    @staticmethod
    def _all_sample_first_dim(
        arrs: "Sequence[np.ndarray]"
    ) -> "Tuple[bool, Optional[int]]":
        s = set(it.shape[0] for it in arrs)
        is_single = len(s) <= 1

        if len(s) == 1:
            length: "Optional[int]" = s.pop()
        else:
            length = None
        return is_single, length

    def _all_sample_same_first_dim(self, arrs: "Sequence[np.ndarray]") -> bool:
        b, _ = self._all_sample_first_dim(arrs)
        return b

    def _check_X(  # noqa: C901
        self,
        X: "Union[npt.ArrayLike, List[npt.ArrayLike], Tuple[npt.ArrayLike, ...], Dict[str, npt.ArrayLike]]",  # noqa: E501
        reset: bool,
        estimator: "Optional[str]" = None,
    ) -> "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]":  # noqa: E501
        multi_input = (
            isinstance(X, (list, tuple, dict))
            and not any(isinstance(xi, (int, float)) for xi in X)
        )

        if multi_input and isinstance(X, tuple):
            X = list(X)

        def _check_X_array(arr: "npt.ArrayLike") -> "np.ndarray":
            return np.asarray(check_array(
                arr,
                accept_sparse=False,
                accept_large_sparse=False,
                force_all_finite=True,
                ensure_min_samples=1,
                ensure_min_features=1,
                allow_nd=True,
                ensure_2d=True,
                dtype=float,
                estimator=estimator,
                input_name="X",
            ))

        if multi_input and isinstance(X, list):
            X_: "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]" = [  # noqa: E501
                _check_X_array(xi)
                for xi
                in X
            ]

            assert isinstance(X_, list)
            if not self._all_sample_same_first_dim(X_):
                raise ValueError(
                    "The X-arrays have different numbers of samples."
                )

            X_dtype_: "Union[List[np.dtype], Dict[str, np.dtype], np.dtype]" = [  # noqa: E501
                xi.dtype for xi in X_
            ]
            n_features_in_: "Union[int, List[int], Dict[str, int]]" = [
                xi.shape[1] for xi in X_
            ]
            X_shape_: "Union[List[Tuple[int, ...]], Dict[str, Tuple[int, ...]], Tuple[int, ...]]" = [  # noqa: E501
                xi.shape for xi in X_
            ]
        elif multi_input and isinstance(X, dict):
            X_ = {k: _check_X_array(xi) for k, xi in X.items()}

            assert isinstance(X_, dict)
            if not self._all_sample_same_first_dim(list(X_.values())):
                raise ValueError(
                    "The X-arrays have different numbers of samples."
                )

            X_dtype_ = {k: xi.dtype for k, xi in X_.items()}
            n_features_in_ = {k: xi.shape[1] for k, xi in X_.items()}
            X_shape_ = {k: xi.shape for k, xi in X_.items()}
        else:
            assert isinstance(X, np.ndarray)
            X_ = _check_X_array(X)
            X_dtype_ = X_.dtype
            n_features_in_ = X_.shape[1]
            X_shape_ = X_.shape

        if reset:
            self.X_dtype_ = X_dtype_
            self.X_shape_ = X_shape_
            self.n_features_in_ = n_features_in_

        elif multi_input:
            isdict = isinstance(X_, dict)

            if isdict:
                assert isinstance(X_, dict)
                keys: "Iterable[Union[int, str]]" = X_.keys()
            else:
                assert isinstance(X_, list)
                keys = range(len(X_))

            for k in keys:
                if isdict:
                    assert isinstance(k, str)
                    assert isinstance(X_dtype_, dict)
                    assert isinstance(X_shape_, dict)
                    assert isinstance(self.X_dtype_, dict)
                    assert isinstance(self.X_shape_, dict)
                    xi_dtype_ = cast("Dict", X_dtype_)[k]
                    dtype_ = cast("Dict", self.X_dtype_)[k]
                    xi_shape_ = cast("Dict", X_shape_)[k]
                    shape_ = cast("Dict", self.X_shape_)[k]
                else:
                    assert isinstance(k, int)
                    assert isinstance(X_dtype_, list)
                    assert isinstance(X_shape_, list)
                    assert isinstance(self.X_dtype_, list)
                    assert isinstance(self.X_shape_, list)
                    xi_dtype_ = cast("List", X_dtype_)[k]
                    dtype_ = cast("List", self.X_dtype_)[k]
                    xi_shape_ = cast("List", X_shape_)[k]
                    shape_ = cast("List", self.X_shape_)[k]

                assert isinstance(xi_dtype_, np.dtype)
                assert isinstance(dtype_, np.dtype)
                if not np.can_cast(xi_dtype_, dtype_):
                    raise ValueError(
                        f"Got y input {k} with dtype {xi_dtype_},"
                        f" but this {self.name} expected {dtype_}"
                        f" and casting from {xi_dtype_} to {dtype_} "
                        "is not safe!"
                    )
                if len(shape_) != len(xi_shape_):
                    raise ValueError(
                        f"X input {k} has {len(xi_shape_)} dimensions, "
                        f"but this {self.name} is expecting "
                        f"{len(shape_)} dimensions in X {k}."
                    )
                if shape_[1:] != xi_shape_[1:]:
                    raise ValueError(
                        f"X has shape {xi_shape_[1:]}, but this "
                        f"{self.name} is expecting X of shape "
                        f"{shape_[1:]}"
                    )
        else:
            assert isinstance(X_dtype_, np.dtype)
            assert isinstance(self.X_dtype_, np.dtype), self.X_dtype_
            if not np.can_cast(X_dtype_, self.X_dtype_):
                raise ValueError(
                    f"Got X with dtype {X_dtype_},"
                    f" but this {self.name} expected {self.X_dtype_}"
                    f" and casting from {X_dtype_} to {self.X_dtype_} "
                    "is not safe!"
                )
            if len(self.X_shape_) != len(X_shape_):
                raise ValueError(
                    f"X has {len(X_shape_)} dimensions, but this "
                    f"{self.name} is expecting {len(self.X_shape_)} "
                    "dimensions in X."
                )

            assert isinstance(X_shape_, tuple)
            assert isinstance(self.X_shape_, tuple)
            if X_shape_[1:] != self.X_shape_[1:]:
                raise ValueError(
                    f"X has shape {X_shape_[1:]}, but this {self.name}"
                    f" is expecting X of shape {self.X_shape_[1:]}"
                )
        return X_

    def _check_y(
        self,
        y: "Union[None, np.ndarray]",
        reset: bool,
        y_numeric: bool = False,
        estimator: "Optional[str]" = None,
    ):
        if y is None:
            if estimator is None:
                estimator_name = "estimator"
            else:
                estimator_name = ""

            raise ValueError(
                f"{estimator_name} requires y to be passed, "
                "but the target y is None."
            )

        return np.asarray(check_array(
            y,
            accept_sparse="csr",
            force_all_finite=True,
            input_name="y",
            ensure_2d=False,
            allow_nd=False,
            dtype=float,
        ))

    def _check_X_y(
        self,
        X: "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]",  # noqa: E501
        y: "np.ndarray",  # noqa: E501
        accept_sparse=False,
        *,
        accept_large_sparse=True,
        dtype="numeric",
        order=None,
        copy=False,
        force_all_finite=True,
        ensure_2d=True,
        allow_nd=False,
        multi_output=False,
        ensure_min_samples=1,
        ensure_min_features=1,
        y_numeric=False,
        estimator=None,
    ) -> """Tuple[
            Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],  # noqa: E501
            np.ndarray,
    ]""":
        """
        A rewrite of
        https://github.com/scikit-learn/scikit-learn/blob/4685cf624582cbc9a35d646f239347e54db798dc/sklearn/utils/validation.py#L941"
        to allow for the multi-input bits
        """
        from sklearn.utils.validation import check_consistent_length

        arrays: "List[np.ndarray]" = []

        if isinstance(X, list):
            arrays.extend(X)
        elif isinstance(X, dict):
            arrays.extend(X.values())
        elif X is not None:
            arrays.append(X)

        if y is not None:
            arrays.append(y)

        check_consistent_length(*arrays)
        return X, y

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _more_tags(self):
        return {
            'multioutput': False,
        }
