
from copy import copy
from baikal import make_step, Step
import numpy as np
import json
import xgboost as xgb

# from scikeras import KerasClassifier  # KerasRegressor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict
    from typing import Optional, Union, Any
    BaseTypes = Union[None, bool, str, int, float]
    Params = Dict[str, BaseTypes]

from sklearn import kernel_approximation
from sklearn import pipeline
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import ensemble
from sklearn import svm
from sklearn import linear_model
import ngboost

from selectml.sk import preprocessor
from selectml.sk import distance
from selectml.sk import feature_selection
from selectml.tf import wrapper as tfmodels


class XGBStep(Step):
    def __repr__(self) -> str:
        cp = copy(self)
        cp.__class__ = cp.__class__.__bases__[1]
        return repr(cp)

    def get_params(self, deep: bool = True) -> "Dict[str, Any]":  # noqa
        # pylint: disable=attribute-defined-outside-init
        """Get parameters."""
        # Based on: https://stackoverflow.com/questions/59248211
        # The basic flow in `get_params` is:
        # 0. Return parameters in subclass first, by using inspect.
        # 1. Return parameters in `XGBModel` (the base class).
        # 2. Return whatever in `**kwargs`.
        # 3. Merge them.

        cp1 = copy(self)
        cp1.__class__ = cp1.__class__.__bases__[1]
        params = cp1.get_params(deep)

        cp2 = copy(self)
        cp2.__class__ = cp2.__class__.__bases__[1].__bases__[0]
        params.update(cp2.get_params(deep))

        # if kwargs is a dict, update params accordingly
        if hasattr(self, "kwargs") and isinstance(self.kwargs, dict):
            params.update(self.kwargs)
        if isinstance(params['random_state'], np.random.RandomState):
            params['random_state'] = params['random_state'].randint(
                np.iinfo(np.int32).max)

        def parse_parameter(
            value: "Any"
        ) -> "Optional[Union[int, float, str]]":
            for t in (int, float, str):
                try:
                    ret = t(value)
                    return ret
                except ValueError:
                    continue
            return None

        # Get internal parameter values
        try:
            config = json.loads(self.get_booster().save_config())
            stack = [config]
            internal = {}
            while stack:
                obj = stack.pop()
                for k, v in obj.items():
                    if k.endswith('_param'):
                        for p_k, p_v in v.items():
                            internal[p_k] = p_v
                    elif isinstance(v, dict):
                        stack.append(v)

            for k, v in internal.items():
                if k in params and params[k] is None:
                    params[k] = parse_parameter(v)
        except ValueError:
            pass
        return params


class XGBRegressor(XGBStep, xgb.XGBRegressor):

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)


class XGBClassifier(XGBStep, xgb.XGBClassifier):

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)


class XGBRanker(XGBStep, xgb.XGBRanker):

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name, **kwargs)


KNeighborsRegressor = make_step(
    neighbors.KNeighborsRegressor,
    class_name="KNeighborsRegressor"
)


KNeighborsClassifier = make_step(
    neighbors.KNeighborsClassifier,
    class_name="KNeighborsClassifier"
)


RandomForestRegressor = make_step(
    ensemble.RandomForestRegressor,
    class_name="RandomForestRegressor"
)


RandomForestClassifier = make_step(
    ensemble.RandomForestClassifier,
    class_name="RandomForestClassifier"
)


ExtraTreesRegressor = make_step(
    ensemble.ExtraTreesRegressor,
    class_name="ExtraTreesRegressor"
)

ExtraTreesClassifier = make_step(
    ensemble.ExtraTreesClassifier,
    class_name="ExtraTreesClassifier"
)


NGBRegressor = make_step(
    ngboost.NGBRegressor,
    class_name="NGBRegressor"
)

NGBClassifier = make_step(
    ngboost.NGBClassifier,
    class_name="NGBClassifier"
)


LinearSVR = make_step(
    svm.LinearSVR,
    class_name="LinearSVR"
)

LinearSVC = make_step(
    svm.LinearSVC,
    class_name="LinearSVC"
)


SGDRegressor = make_step(
    linear_model.SGDRegressor,
    class_name="SGDRegressor"
)

SGDClassifier = make_step(
    linear_model.SGDClassifier,
    class_name="SGDClassifier"
)

LassoLars = make_step(
    linear_model.LassoLars,
    class_name="LassoLars"
)

Lars = make_step(
    linear_model.Lars,
    class_name="Lars"
)

ConvMLPClassifier = make_step(
    tfmodels.ConvMLPClassifier,
    class_name="ConvMLPClassifier"
)

ConvMLPRegressor = make_step(
    tfmodels.ConvMLPRegressor,
    class_name="ConvMLPRegressor"
)

ConvMLPRanker = make_step(
    tfmodels.ConvMLPRanker,
    class_name="ConvMLPRanker"
)

Nystroem = make_step(
    kernel_approximation.Nystroem,
    class_name="Nystroem"
)

QuantileTransformer = make_step(
    preprocessing.QuantileTransformer,
    class_name="QuantileTransformer"
)

OneHotEncoder = make_step(
    preprocessing.OneHotEncoder,
    class_name="OneHotEncoder"
)

StandardScaler = make_step(
    preprocessing.StandardScaler,
    class_name="StandardScaler"
)

RobustScaler = make_step(
    preprocessing.RobustScaler,
    class_name="RobustScaler"
)

PowerTransformer = make_step(
    preprocessing.PowerTransformer,
    class_name="PowerTransformer"
)

PolynomialFeatures = make_step(
    preprocessing.PolynomialFeatures,
    class_name="PolynomialFeatures"
)

Pipeline = make_step(pipeline.Pipeline, class_name="Pipeline")

FactorAnalysis = make_step(
    decomposition.FactorAnalysis,
    class_name="FactorAnalysis"
)

TruncatedSVD = make_step(decomposition.TruncatedSVD, class_name="TruncatedSVD")

Unity = make_step(preprocessor.Unity, class_name="Unity")

MAFScaler = make_step(preprocessor.MAFScaler, class_name="MAFScaler")

NOIAAdditiveScaler = make_step(
    preprocessor.NOIAAdditiveScaler,
    class_name="NOIAAdditiveScaler"
)

OrdinalTransformer = make_step(
    preprocessor.OrdinalTransformer,
    class_name="OrdinalTransformer"
)

Make2D = make_step(
    preprocessor.Make2D,
    class_name="Make2D"
)

Make1D = make_step(
    preprocessor.Make1D,
    class_name="Make1D"
)

VanRadenSimilarity = make_step(
    distance.VanRadenSimilarity,
    class_name="VanRadenSimilarity"
)

ManhattanDistance = make_step(
    distance.ManhattanDistance,
    class_name="ManhattanDistance"
)

EuclideanDistance = make_step(
    distance.EuclideanDistance,
    class_name="EuclideanDistance"
)

MultiSURF = make_step(feature_selection.MultiSURF, class_name="MultiSURF")

GEMMASelector = make_step(
    feature_selection.GEMMASelector,
    class_name="GEMMASelector"
)

MAFSelector = make_step(
    feature_selection.MAFSelector,
    class_name="MAFSelector"
)
