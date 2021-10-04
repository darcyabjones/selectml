#!/usr/bin/env python3

import enum
from typing import Type

from ..sk.optimise.base import OptimiseModel
from ..sk.optimise.models import (
    XGBModel,
    KNNModel,
    RFModel,
    ExtraTreesModel,
    NGBModel,
    SVRModel,
    ElasticNetDistModel,
    ElasticNetModel,
    ConvModel,
    MLPModel,
    ConvMLPModel
)


class ModelOptimiser(enum.Enum):

    xgb = 1
    knn = 2
    rf = 3
    extratrees = 4
    ngb = 5
    svr = 6
    elasticnetdist = 7
    elasticnet = 8
    conv = 9
    mlp = 10
    convmlp = 11

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_string(cls, s: str) -> "ModelOptimiser":
        try:
            return cls[s]
        except KeyError:
            raise ValueError(f"{s} is not a valid result type to parse.")

    def get_model(self) -> Type[OptimiseModel]:
        return NAME_TO_MODEL[self]


NAME_TO_MODEL = {
    ModelOptimiser.xgb: XGBModel,
    ModelOptimiser.knn: KNNModel,
    ModelOptimiser.rf: RFModel,
    ModelOptimiser.extratrees: ExtraTreesModel,
    ModelOptimiser.ngb: NGBModel,
    ModelOptimiser.svr: SVRModel,
    ModelOptimiser.elasticnetdist: ElasticNetDistModel,
    ModelOptimiser.elasticnet: ElasticNetModel,
    ModelOptimiser.conv: ConvModel,
    ModelOptimiser.mlp: MLPModel,
    ModelOptimiser.convmlp: ConvMLPModel
}
