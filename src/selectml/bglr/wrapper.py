
from dataclasses import dataclass
from ..higher import fmap, or_else
import numpy as np

from rpy2 import robjects as ro
from rpy2 import rinterface as ri


from typing import Literal
from typing import Sequence, Tuple, List, Dict
from typing import Union, Optional
from typing import Callable
from typing import TypeVar
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

    def predict(self, X: np.ndarray):
        return np.matmul(X, self.b)

    @classmethod
    def from_r(self, lv: ro.ListVector) -> "BGLRResult":

        switch = {
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

        return switch.get(model).from_r(lv)

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

        return cls(
            name=cls._get_name(lv),
            K=get_float_array(lv.rx2("K")),
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

        K = self.K
        K_inv = np.linalg.pinv(K)
        X_solved = X.dot(K_inv)
        return X_solved.dot(self.u)


class BGLR(object):

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
            component_prediction_ = list(component_prediction)

        is_bool = [isinstance(p, bool) for p in component_prediction_]
        is_str = [isinstance(p, str) for p in component_prediction_]
        is_int = [isinstance(p, int) for p in component_prediction_]

        if any(is_bool) and not all(is_bool):
            raise ValueError(
                "Some of the component_prediction elements are booleans, "
                "but not all are."
            )
        elif all(is_bool) and not (len(component_prediction) == len(models)):
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
            component_prediction2_: "Union[List[str], List[int]]" = [
                n
                for n, b
                in zip(
                    component_names,
                    cast("List[bool]", component_prediction_)
                )
                if b
            ]
        else:
            component_prediction2_ = cast(
                "Union[List[str], List[int]]",
                component_prediction_
            )

        assert len(component_prediction2_) > 0
        return component_prediction2_

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
    ):
        if isinstance(models, str):
            models_ = [models]
        else:
            models_ = list(models)

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
        return

    def fit(
        self,
        X: "Union[List[npt.ArrayLike], Tuple[npt.ArrayLike, ...]]",
        y: "npt.ArrayLike",
        sample_weight: "Optional[npt.ArrayLike]" = None,
        groups: "Optional[npt.ArrayLike]" = None
    ):
        results = self.fit_bglr(X, y, sample_weight, groups)
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
        self.yhat = get_float_array(results.rx2("yHat"))

        return self

    def fit_bglr(
        self,
        X: "Union[npt.ArrayLike, List[npt.ArrayLike], Tuple[npt.ArrayLike, ...]]",  # noqa: E501
        y: "npt.ArrayLike",
        sample_weight: "Optional[npt.ArrayLike]" = None,
        groups: "Optional[npt.ArrayLike]" = None,
    ) -> "np.ndarray":
        import rpy2.robjects as ro
        import rpy2.robjects.vectors as rov
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()
        if isinstance(X, (list, tuple)):
            X_ = [np.asarray(xi) for xi in X]
        else:
            X_ = [np.asarray(X)]
        y_ = np.asarray(y)

        BGLR = importr("BGLR")

        if self.component_names is not None:
            names: "List[Union[str, int]]" = self.component_names
        else:
            names = list(range(len(self.models)))

        if all(isinstance(i, int) for i in self.component_names):
            ETA: "Union[List[ro.ListVector], Dict[str, ro.ListVector]]" = []
            for data, mtype in zip(X_, self.models):
                if mtype == "RKHS":
                    ETA.append(ro.ListVector({"K": data, "model": mtype}))
                else:
                    ETA.append(ro.ListVector({"X": data, "model": mtype}))

        else:
            ETA = {}
            for data, name, mtype in zip(X_, names, self.models):
                if mtype == "RKHS":
                    ETA[name] = ro.ListVector({"K": data, "model": mtype})
                else:
                    ETA[name] = ro.ListVector({"X": data, "model": mtype})

        results = BGLR.BGLR(
            y=y_,
            response_type=self.response_type,
            ETA=ro.ListVector(ETA) if isinstance(ETA, dict) else ETA,
            nIter=self.niter,
            burnIn=self.burnin,
            thin=self.thin,
            saveAt=self.saveat,
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
        if isinstance(X, (list, tuple)):
            X_ = [np.asarray(xi) for xi in X]
        else:
            X_ = [np.asarray(X)]

        yhat = self.intercept_
        for name, Xi in zip(self.component_names, X_):
            if name not in self.component_prediction:
                continue

            component = self.eta_[name]
            yhat += component.predict(Xi)

        return yhat
