#!/usr/bin/env python3

import enum
from typing import Type

from ..optimiser.runners import (
    BaseRunner,
    BGLRSKRunner,
    BGLRRunner,
    XGBRunner,
    KNNRunner,
    NGBRunner,
    RFRunner,
    SVMRunner,
    SGDRunner,
    LarsRunner,
    TFRunner,
)

from ..optimiser.stats import (
    OptimiseStats,
    RegressionStats,
    RankingStats,
    ClassificationStats,
)


class ModelOptimiser(enum.Enum):

    xgb = 1
    knn = 2
    rf = 3
    ngb = 4
    svm = 5
    sgd = 6
    lars = 7
    bglr_sk = 8
    bglr = 9
    tf = 10

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_string(cls, s: str) -> "ModelOptimiser":
        try:
            return cls[s]
        except KeyError:
            raise ValueError(f"{s} is not a valid result type to parse.")

    def get_model(self) -> Type[BaseRunner]:
        return NAME_TO_MODEL[self]


NAME_TO_MODEL = {
    ModelOptimiser.xgb: XGBRunner,
    ModelOptimiser.knn: KNNRunner,
    ModelOptimiser.rf: RFRunner,
    ModelOptimiser.ngb: NGBRunner,
    ModelOptimiser.svm: SVMRunner,
    ModelOptimiser.sgd: SGDRunner,
    ModelOptimiser.lars: LarsRunner,
    ModelOptimiser.bglr_sk: BGLRSKRunner,
    ModelOptimiser.bglr: BGLRRunner,
    ModelOptimiser.tf: TFRunner,
}


class Stats(enum.Enum):

    regression = 1
    ranking = 2
    ordinal = 3
    classification = 4

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_string(cls, s: str) -> "Stats":
        try:
            return cls[s]
        except KeyError:
            raise ValueError(f"{s} is not a valid result type to parse.")

    def get_stats(self) -> OptimiseStats:
        return NAME_TO_STATS[self]


NAME_TO_STATS = {
    Stats.regression: RegressionStats(),
    Stats.ranking: RankingStats(),
    Stats.ordinal: RankingStats(),
    Stats.classification: ClassificationStats(),
}
