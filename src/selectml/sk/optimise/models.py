#!/usr/bin/env python3

from math import log2
from typing import (
    Union,
    List,
    Dict,
    Any,
    Tuple,
    Optional
)

from collections import namedtuple

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy.typing as npt

import optuna
import numpy as np
from .base import SKModel, TFBaseModel

BaseTypes = Union[None, bool, str, int, float]

TFTuple = namedtuple(
    "TFTuple",
    ["target", "grouping", "marker", "model"]
)


class XGBModel(SKModel):

    use_weights: bool = True

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        params = self.sample_preprocessing_params(
            trial,
            target_options=["passthrough"],
            marker_options=["maf"],
            dist_options=["drop", "vanraden"],
            nonlinear_options=["drop"],
            feature_selection_options=["passthrough", "relief"],
            grouping_options=["onehot"],
            grouping_interaction_options=["drop"],
        )

        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "booster": trial.suggest_categorical(
                "booster",
                ["gbtree", "gblinear", "dart"]
            ),
            "gamma": trial.suggest_float("gamma", 0, 100),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.1, 1),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.1,
                1
            ),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel",
                0.1,
                1
            ),
            "colsample_bynode": trial.suggest_float(
                "colsample_bynode",
                0.1,
                1
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                0,
                50
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                0,
                50
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                10
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                1e-4,
                0.5,
                log=True
            ),
        })
        return params

    def model(self, params: Dict[str, Any]):
        import xgboost as xgb
        from sklearn.pipeline import Pipeline

        _, preprocessor = self.sample_preprocessing_model(params)
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=params["n_estimators"],
                booster=params["booster"],
                gamma=params["gamma"],
                min_child_weight=params["min_child_weight"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                colsample_bylevel=params["colsample_bylevel"],
                colsample_bynode=params["colsample_bynode"],
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
                random_state=self.seed,
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                n_jobs=1,
                verbosity=0,
            ))
        ])

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
            {
                "train_means": False,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "dist_preprocessor": "drop",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "n_estimators": 500,
                "booster": "gbtree",
                "gamma": 10,
                "min_child_weight": 1,
                "subsample": 1,
                "colsample_bytree": 1,
                "colsample_bylevel": 1,
                "colsample_bynode": 1,
                "reg_alpha": 1,
                "reg_lambda": 1,
                "max_depth": 4,
                "learning_rate": 1e-3,
            },
            {
                "train_means": False,
                "marker_preprocessor": "maf",
                "target_options": "passthrough",
                "feature_selector": "passthrough",
                "dist_preprocessor": "drop",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "n_estimators": 500,
                "booster": "gbtree",
                "gamma": 10,
                "min_child_weight": 1,
                "subsample": 1,
                "colsample_bytree": 1,
                "colsample_bylevel": 1,
                "colsample_bynode": 1,
                "reg_alpha": 1,
                "reg_lambda": 1,
                "max_depth": 9,
                "learning_rate": 1e-3,
            }
        ]


class KNNModel(SKModel):

    use_weights: bool = False

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        params = self.sample_preprocessing_params(
            trial,
            target_options=["passthrough"],
            marker_options=["maf"],
            dist_options=["vanraden"],
            nonlinear_options=["drop", "rbf", "laplacian", "poly"],
            feature_selection_options=["passthrough", "rf", "relief"],
            grouping_options=["onehot"],
            grouping_interaction_options=["drop", "poly"],
        )

        params.update({
            "n_neighbors": trial.suggest_int("n_neighbors", 2, 100),
            "weights": trial.suggest_categorical(
                "weights",
                ["distance", "uniform"]
            ),
            "leaf_size": trial.suggest_int("leaf_size", 10, 80),
            "algorithm": trial.suggest_categorical(
                "algorithm",
                ["kd_tree", "ball_tree"]
            ),
            "p": trial.suggest_categorical("p", [1, 2]),
        })
        return params

    def model(self, params: Dict[str, Any]):
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import Pipeline

        _, preprocessor = self.sample_preprocessing_model(params)
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", KNeighborsRegressor(
                n_neighbors=params["n_neighbors"],
                weights=params["weights"],
                leaf_size=params["leaf_size"],
                algorithm=params["algorithm"],
                p=params["p"],
                n_jobs=1
            ))
        ])

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "drop",
                "dist_preprocessor": "vanraden",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "n_neighbors": 10,
                "weights": "distance",
                "leaf_size": 10,
                "algorithm": "kd_tree",
                "p": 1,
            },
            {
                "train_means": False,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "drop",
                "dist_preprocessor": "vanraden",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "n_neighbors": 10,
                "weights": "distance",
                "leaf_size": 50,
                "algorithm": "kd_tree",
                "p": 1,
            },
        ]


class RFModel(SKModel):

    use_weights: bool = True

    def sample_params(self, trial: "optuna.Trial"):

        params = self.sample_preprocessing_params(
            trial,
            target_options=["passthrough"],
            marker_options=["maf"],
            dist_options=["drop", "vanraden"],
            nonlinear_options=["drop"],
            feature_selection_options=["passthrough", "relief"],
            grouping_options=["onehot"],
            grouping_interaction_options=["drop"],
        )

        assert isinstance(params["nparams"], int)
        params["max_features"] = trial.suggest_int(
            "max_features",
            max([1, round(log2(params["nparams"]))]),
            params["nparams"]
        )

        params.update({
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_impurity_decrease": trial.suggest_float(
                "min_impurity_decrease",
                0,
                1
            ),
        })

        if params["bootstrap"]:
            params["oob_score"] = trial.suggest_categorical(
                "oob_score",
                [True, False]
            )
        else:
            params["oob_score"] = False

        return params

    def model(self, params: Dict[str, Any]):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline

        _, preprocessor = self.sample_preprocessing_model(params)

        # Prevents key error when selecting from "best params"
        if "oob_score" in params:
            oob_score = params["oob_score"]
        else:
            oob_score = False

        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                criterion="mae",
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                max_features=params["max_features"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                min_impurity_decrease=params["min_impurity_decrease"],
                bootstrap=params["bootstrap"],
                oob_score=oob_score,
                n_jobs=1,
                random_state=self.seed
            ))
        ])

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
            {
                "train_means": False,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "dist_preprocessor": "drop",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "max_depth": 3,
                "n_estimators": 500,
                "max_features": 50,
                "min_samples_split": 2,
                "min_samples_leaf": 10,
                "min_impurity_decrease": 0.1,
                "bootstrap": False,
            },
        ]


class ExtraTreesModel(SKModel):

    use_weights: bool = True

    def sample_params(self, trial: "optuna.Trial"):
        params = self.sample_preprocessing_params(
            trial,
            target_options=["passthrough"],
            marker_options=["maf"],
            dist_options=["drop", "vanraden"],
            nonlinear_options=["drop"],
            feature_selection_options=["passthrough", "relief"],
            grouping_options=["onehot"],
            grouping_interaction_options=["drop"],
        )

        assert isinstance(params["nparams"], int)
        params["max_features"] = trial.suggest_int(
            "max_features",
            max([1, round(log2(params["nparams"]))]),
            params["nparams"]
        )

        params.update({
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "max_samples": trial.suggest_int("max_samples", 1, 127),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_impurity_decrease": trial.suggest_float(
                "min_impurity_decrease",
                0,
                1
            ),
        })

        if params["bootstrap"]:
            params["oob_score"] = trial.suggest_categorical(
                "oob_score",
                [True, False]
            )
        else:
            params["oob_score"] = False

        return params

    def model(self, params: Dict[str, Any]):
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.pipeline import Pipeline

        _, preprocessor = self.sample_preprocessing_model(params)

        if "oob_score" in params:
            oob_score = params["oob_score"]
        else:
            oob_score = False

        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", ExtraTreesRegressor(
                criterion="mae",
                max_depth=params["max_depth"],
                n_estimators=params["n_estimators"],
                max_features=params["max_features"],
                max_samples=params["max_samples"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                min_impurity_decrease=params["min_impurity_decrease"],
                bootstrap=params["bootstrap"],
                oob_score=oob_score,
                n_jobs=1,
                random_state=self.seed
            ))
        ])

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
            {
                "train_means": False,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "dist_preprocessor": "drop",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "max_depth": 3,
                "n_estimators": 500,
                "max_samples": 50,
                "max_features": 50,
                "min_samples_split": 2,
                "min_samples_leaf": 10,
                "min_impurity_decrease": 0.1,
                "bootstrap": False,
            },
        ]


class NGBModel(SKModel):

    use_weights: bool = True

    def sample_params(self, trial: "optuna.Trial"):
        params = self.sample_preprocessing_params(
            trial,
            target_options=["passthrough"],
            marker_options=["maf"],
            dist_options=["drop", "vanraden"],
            nonlinear_options=["drop"],
            feature_selection_options=["passthrough", "relief"],
            grouping_options=["onehot"],
            grouping_interaction_options=["drop"],
        )

        assert isinstance(params["nparams"], int)
        params["max_features"] = trial.suggest_int(
            "max_features",
            max([1, round(log2(params["nparams"]))]),
            params["nparams"]
        )

        params.update({
            "Dist": trial.suggest_categorical(
                "Dist",
                ["normal", "exponential"]
            ),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "min_impurity_decrease": trial.suggest_float(
                "min_impurity_decrease",
                0,
                1
            ),
            "col_sample": trial.suggest_float("col_sample", 0.1, 0.5),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                1e-4,
                0.1,
                log=True
            ),
            "natural_gradient": trial.suggest_categorical(
                "natural_gradient",
                [True, False]
            ),
        })

        return params

    def model(self, params: Dict[str, Any]):
        from ngboost.distns.normal import Normal
        from ngboost.distns.cauchy import Cauchy
        from ngboost.distns.exponential import Exponential
        from ngboost.distns.lognormal import LogNormal
        from ngboost import NGBRegressor
        from ngboost.learners import DecisionTreeRegressor

        from sklearn.pipeline import Pipeline

        _, preprocessor = self.sample_preprocessing_model(params)

        dist = {
            "normal": Normal,
            "cauchy": Cauchy,
            "exponential": Exponential,
            "lognormal": LogNormal,
        }[params["Dist"]]

        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", NGBRegressor(
                Dist=dist,
                n_estimators=params["n_estimators"],
                Base=DecisionTreeRegressor(
                    criterion='friedman_mse',
                    max_depth=params["max_depth"],
                    max_features=params["max_features"],
                    min_samples_split=params["min_samples_split"],
                    min_samples_leaf=params["min_samples_leaf"],
                    min_impurity_decrease=params["min_impurity_decrease"]
                ),
                verbose=False,
                col_sample=params["col_sample"],
                learning_rate=params["learning_rate"],
                natural_gradient=params["natural_gradient"],
                random_state=self.seed
            ))
        ])

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
            {
                "train_means": False,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "dist_preprocessor": "drop",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "Dist": "normal",
                "n_estimators": 500,
                "max_depth": 3,
                "max_features": 50,
                "min_samples_split": 2,
                "min_samples_leaf": 10,
                "min_impurity_decrease": 0.1,
                "col_sample": 0.5,
                "learning_rate": 0.1,
                "natural_gradient": True,
            },
        ]


class SVRModel(SKModel):

    use_weights: bool = True

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, BaseTypes]:

        params = self.sample_preprocessing_params(
            trial,
            target_options=["stdnorm", "quantile"],
            marker_options=["maf"],
            dist_options=["vanraden"],
            nonlinear_options=["drop", "rbf", "laplacian", "poly"],
            feature_selection_options=["drop", "relief"],
            grouping_options=["onehot"],
            grouping_interaction_options=["drop", "poly"],
        )

        params.update({
            "loss": trial.suggest_categorical(
                "loss",
                ['epsilon_insensitive', 'squared_epsilon_insensitive']
            ),
            "epsilon": trial.suggest_float("epsilon", 1e-10, 100, log=True),
            "C": trial.suggest_float("C", 1e-10, 10),
            "intercept_scaling": trial.suggest_float(
                "intercept_scaling",
                1e-10,
                5
            ),
            "dual": trial.suggest_categorical(
                "dual",
                [True, False]
            )
        })
        return params

    def model(self, params: Dict[str, Any]):
        from sklearn.svm import LinearSVR
        from sklearn.pipeline import Pipeline
        from sklearn.compose import TransformedTargetRegressor

        target_trans, preprocessor = self.sample_preprocessing_model(params)
        return TransformedTargetRegressor(
            regressor=Pipeline([
                ("preprocessor", preprocessor),
                ("model", LinearSVR(
                    random_state=self.seed,
                    fit_intercept=True,
                    max_iter=100000,
                    dual=params["dual"],
                    C=params["C"],
                    epsilon=params["epsilon"],
                    loss=params["loss"],
                    intercept_scaling=params["intercept_scaling"],
                ))
            ]),
            transformer=target_trans
        )

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "drop",
                "dist_preprocessor": "vanraden",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "loss": "epsilon_insensitive",
                "epsilon": 0,
                "C": 1,
                "dual": True,
                "intercept_scaling": 0.1,
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "drop",
                "dist_preprocessor": "vanraden",
                "nonlinear_preprocessor": "poly",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "poly_gamma": 1,
                "grouping_preprocessor": "onehot",
                "loss": "epsilon_insensitive",
                "epsilon": 0,
                "C": 1,
                "dual": True,
                "intercept_scaling": 0.1,
            },
        ]


class ElasticNetDistModel(SKModel):

    use_weights: bool = True

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, BaseTypes]:
        params = self.sample_preprocessing_params(
            trial,
            target_options=["passthrough", "stdnorm", "quantile"],
            marker_options=["maf"],
            dist_options=["vanraden"],
            nonlinear_options=["drop", "rbf", "laplacian", "poly"],
            feature_selection_options=["drop"],
            grouping_options=["onehot"],
            grouping_interaction_options=["drop", "poly"],
        )

        params.update({
            "alpha": trial.suggest_float("alpha", 0, 50),
            "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
        })
        return params

    def model(self, params: Dict[str, Any]):
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.compose import TransformedTargetRegressor

        target_trans, preprocessor = self.sample_preprocessing_model(params)
        return TransformedTargetRegressor(
            regressor=Pipeline([
                ("preprocessor", preprocessor),
                ("model", ElasticNet(
                    random_state=self.seed,
                    fit_intercept=True,
                    max_iter=100000,
                    selection="random",
                    alpha=params["alpha"],
                    l1_ratio=params["l1_ratio"],
                ))
            ]),
            transformer=target_trans
        )

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "drop",
                "dist_preprocessor": "vanraden",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "alpha": 10,
                "l1_ratio": 0.5,
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "drop",
                "dist_preprocessor": "vanraden",
                "nonlinear_preprocessor": "poly",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "poly_gamma": 1,
                "alpha": 10,
                "l1_ratio": 0.5,
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "drop",
                "dist_preprocessor": "vanraden",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "alpha": 10,
                "l1_ratio": 0.0,
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "drop",
                "dist_preprocessor": "vanraden",
                "nonlinear_preprocessor": "poly",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "poly_gamma": 1,
                "alpha": 10,
                "l1_ratio": 0.0,
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "drop",
                "dist_preprocessor": "vanraden",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "alpha": 10,
                "l1_ratio": 1.0,
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "drop",
                "dist_preprocessor": "vanraden",
                "nonlinear_preprocessor": "poly",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "poly_gamma": 1,
                "alpha": 10,
                "l1_ratio": 1.0,
            },
        ]


class LassoLarsDistModel(SKModel):

    use_weights: bool = False

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, BaseTypes]:
        params = self.sample_preprocessing_params(
            trial,
            target_options=["passthrough", "stdnorm", "quantile"],
            marker_options=["maf"],
            dist_options=["vanraden"],
            nonlinear_options=["drop", "rbf", "laplacian", "poly"],
            feature_selection_options=["drop"],
            grouping_options=["onehot"],
            grouping_interaction_options=["drop", "poly"],
        )

        params.update({
            "alpha": trial.suggest_float("alpha", 0, 50),
        })
        return params

    def model(self, params: Dict[str, Any]):
        from sklearn.linear_model import LassoLars
        from sklearn.pipeline import Pipeline
        from sklearn.compose import TransformedTargetRegressor

        target_trans, preprocessor = self.sample_preprocessing_model(params)
        return TransformedTargetRegressor(
            regressor=Pipeline([
                ("preprocessor", preprocessor),
                ("model", LassoLars(
                    alpha=params["alpha"],
                    fit_intercept=True,
                    max_iter=100000,
                    random_state=self.seed,
                ))
            ]),
            transformer=target_trans
        )

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "drop",
                "dist_preprocessor": "vanraden",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "alpha": 10,
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "drop",
                "dist_preprocessor": "vanraden",
                "nonlinear_preprocessor": "poly",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "dist_poly_gamma": 1,
                "alpha": 10,
            },
        ]


class LassoLarsModel(SKModel):

    use_weights: bool = False

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, BaseTypes]:
        params = self.sample_preprocessing_params(
            trial,
            target_options=["passthrough", "stdnorm", "quantile"],
            marker_options=["maf"],
            dist_options=["drop", "vanraden"],
            nonlinear_options=["drop", "rbf", "laplacian", "poly"],
            feature_selection_options=["passthrough", "relief"],
            grouping_options=["onehot"],
            grouping_interaction_options=["drop", "poly"],
        )

        params.update({
            "alpha": trial.suggest_float("alpha", 0, 50),
        })
        return params

    def model(self, params: Dict[str, Any]):
        from sklearn.linear_model import LassoLars
        from sklearn.pipeline import Pipeline
        from sklearn.compose import TransformedTargetRegressor

        target_trans, preprocessor = self.sample_preprocessing_model(params)
        return TransformedTargetRegressor(
            regressor=Pipeline([
                ("preprocessor", preprocessor),
                ("model", LassoLars(
                    alpha=params["alpha"],
                    fit_intercept=True,
                    max_iter=500000,
                    random_state=self.seed,
                ))
            ]),
            transformer=target_trans
        )

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "dist_preprocessor": "drop",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "alpha": 10,
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "dist_preprocessor": "drop",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "poly",
                "alpha": 10,
            },
        ]


class ElasticNetModel(SKModel):

    use_weights: bool = True

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, BaseTypes]:
        params = self.sample_preprocessing_params(
            trial,
            target_options=["passthrough", "stdnorm", "quantile"],
            marker_options=["maf"],
            dist_options=["drop", "vanraden"],
            nonlinear_options=["drop", "rbf", "laplacian", "poly"],
            feature_selection_options=["passthrough", "relief"],
            grouping_options=["onehot"],
            grouping_interaction_options=["drop", "poly"],
        )

        params.update({
            "alpha": trial.suggest_float("alpha", 0, 50),
            "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
        })
        return params

    def model(self, params: Dict[str, Any]):
        from sklearn.linear_model import ElasticNet
        from sklearn.pipeline import Pipeline
        from sklearn.compose import TransformedTargetRegressor

        target_trans, preprocessor = self.sample_preprocessing_model(params)
        return TransformedTargetRegressor(
            regressor=Pipeline([
                ("preprocessor", preprocessor),
                ("model", ElasticNet(
                    random_state=self.seed,
                    fit_intercept=True,
                    max_iter=100000,
                    selection="random",
                    alpha=params["alpha"],
                    l1_ratio=params["l1_ratio"],
                ))
            ]),
            transformer=target_trans
        )

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "dist_preprocessor": "drop",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "alpha": 0.0,
                "l1_ratio": 0.5,
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "dist_preprocessor": "drop",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "alpha": 10,
                "l1_ratio": 0.5,
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "dist_preprocessor": "drop",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "alpha": 10,
                "l1_ratio": 0.0,
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "passthrough",
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "dist_preprocessor": "drop",
                "nonlinear_preprocessor": "drop",
                "grouping_preprocessor": "onehot",
                "interactions": "drop",
                "alpha": 10,
                "l1_ratio": 1.0,
            },
        ]


class TFWrapper(object):

    def __init__(
        self,
        model=None,
        target_trans=None,
        group_trans=None,
        marker_trans=None,
    ):
        self.model = model
        self.target_trans = target_trans
        self.group_trans = group_trans
        self.marker_trans = marker_trans
        return

    def get_validation_samples(
        self,
        X_markers,
        X_grouping,
        y,
        individuals,
        sample_weight,
        proportion=0.2
    ):
        rng = np.random.default_rng()

        uindivs = np.unique(individuals)
        n = min([1, np.floor(len(uindivs) * proportion)])
        val_samples = set(rng.choice(uindivs, size=n, replace=False))

        val = np.array([(i in val_samples) for i in individuals])

        val_tup = (
            X_markers[val],
            X_grouping[val],
            y[val],
            individuals[val],
            sample_weight[val],
        )

        train_tup = (
            X_markers[~val],
            X_grouping[~val],
            y[~val],
            individuals[~val],
            sample_weight[~val],
        )
        return train_tup, val_tup

    def fit(self, X, y, sample_weight=None, individuals=None, **kwargs):
        import tensorflow as tf

        X_markers_, X_grouping_ = X
        X_markers = np.array(X_markers_)

        if X_grouping_ is not None:
            X_grouping = np.array(X_grouping_)
        else:
            X_grouping = None

        del X_markers_, X_grouping_

        y_ = np.array(y)
        if len(y_.shape) == 1:
            y_ = np.expand_dims(y_, -1)

        if self.target_trans is not None:
            y_ = self.target_trans.fit_transform(
                y_,
                individuals=individuals
            )

        if (X_grouping is not None) and (self.group_trans is not None):
            X_grouping = self.group_trans.fit_transform(X_grouping)

        if self.marker_trans is not None:
            X_markers = self.marker_trans.fit_transform(
                X_markers,
                y=y,
                individuals=individuals
            )

        channel = self._np_to_channels(X_markers, X_grouping, y, sample_weight)

        early_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0.01,
            patience=20
        )

        self.model.compile(optimizer="adam", loss="mae", metrics=["mae"])

        _ = self.model.fit(
            channel.shuffle(1024).batch(64),
            epochs=500,
            verbose=0,
            callbacks=[early_callback]
        )
        return self

    def predict(self, X: "Tuple[npt.ArrayLike, Optional[npt.ArrayLike]]"):
        X_markers_, X_grouping_ = X
        X_markers = np.array(X_markers_)
        X_grouping = np.array(X_grouping_)
        del X_markers_, X_grouping_

        if (X_grouping is not None) and (self.group_trans is not None):
            X_grouping = self.group_trans.transform(X_grouping)

        if self.marker_trans is not None:
            X_markers = self.marker_trans.transform(
                X_markers,
            )

        channel = self._np_to_channels(X_markers, X_grouping)

        preds = self.model.predict(channel.batch(32))
        if len(preds.shape) == 1:
            preds = np.expand_dims(preds, -1)

        if self.target_trans is not None:
            preds = self.target_trans.inverse_transform(
                preds,
            )

        return preds

    def _np_to_channels(
        self,
        X_markers,
        X_groups=None,
        y=None,
        sample_weights=None
    ):
        from scipy.sparse import issparse
        from tensorflow.data import Dataset

        if issparse(X_markers):
            X_markers = X_markers.toarray()

        if X_groups is None:
            datasets = [X_markers]
        else:
            if issparse(X_groups):
                X_groups = X_groups.toarray()
            datasets = [(X_markers, X_groups)]

        if y is not None:
            datasets.append(y)

            if sample_weights is not None:
                datasets.append(sample_weights)

        return Dataset.from_tensor_slices(tuple(datasets))


class ConvModel(TFBaseModel):

    use_weights: bool = True

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, BaseTypes]:
        params = self.sample_preprocessing_params(
            trial,
            target_options=["stdnorm", "quantile"],
            marker_options=["maf"],
            feature_selection_options=["passthrough", "relief"],
        )

        params["conv_nlayers"] = trial.suggest_int("conv_nlayers", 1, 3)
        params["conv_filters"] = trial.suggest_int("conv_filters", 1, 4)
        params["conv_kernel_size"] = trial.suggest_int(
            "conv_kernel_size",
            2,
            5
        )
        params["conv_strides"] = trial.suggest_int(
            "conv_strides",
            2,
            params["conv_kernel_size"]
        )
        params["conv_l1"] = trial.suggest_float("conv_l1", 1e-50, 50, log=True)
        params["conv_l2"] = trial.suggest_float("conv_l2", 1e-50, 50, log=True)
        params["conv_activation"] = trial.suggest_categorical(
            "conv_activation",
            ["linear", "relu"]
        )
        params["conv_use_bn"] = trial.suggest_categorical(
            "conv_use_bn",
            [True, False]
        )
        params["dropout_rate"] = trial.suggest_float("dropout_rate", 0, 1)

        params["marker_l1"] = trial.suggest_float(
            "marker_l1",
            1e-50,
            50,
            log=True
        )
        params["marker_l2"] = trial.suggest_float(
            "marker_l2",
            1e-50,
            50,
            log=True
        )

        if len(self.grouping_columns) > 0:
            params["group_l1"] = trial.suggest_float(
                "group_l1",
                1e-50,
                50,
                log=True
            )
            params["group_l2"] = trial.suggest_float(
                "group_l2",
                1e-50,
                50,
                log=True
            )

        return params

    def model(self, params: Dict[str, Any]):
        from selectml.tf.layers import AddChannel, Flatten1D, ConvLinkage
        from selectml.tf.models import SSModel
        from tensorflow.keras.layers import Dropout, Dense
        from tensorflow.keras.regularizers import L1L2
        from tensorflow.keras.models import Sequential

        (
            target_trans,
            grouping_trans,
            marker_trans
        ) = self.sample_preprocessing_model(params)

        marker_model = Sequential([
            AddChannel(),
            ConvLinkage(
                nlayers=params["conv_nlayers"],
                filters=params["conv_filters"],
                strides=params["conv_strides"],
                kernel_size=params["conv_kernel_size"],
                activation=params["conv_activation"],
                activation_first=params["conv_activation"],
                activation_last=params["conv_activation"],
                kernel_regularizer=L1L2(
                    params["conv_l1"],
                    params["conv_l2"],
                ),
                use_bn=params["conv_use_bn"]
            ),
            Flatten1D(),
            Dropout(params["dropout_rate"]),
            Dense(1, use_bias=True, kernel_regularizer=L1L2(
                params["marker_l1"],
                params["marker_l2"])
            ),
        ])

        if len(self.grouping_columns) > 0:
            group_model = Dense(1, use_bias=True, kernel_regularizer=L1L2(
                params["group_l1"],
                params["group_l2"])
            )
        else:
            group_model = None

        model = SSModel(
            marker_embedder=marker_model,
            env_embedder=group_model,
            combiner="add",
        )

        return TFWrapper(
            model=model,
            target_trans=target_trans,
            group_trans=grouping_trans,
            marker_trans=marker_trans,
        )

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
        ]


class MLPModel(TFBaseModel):

    use_weights: bool = True

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, BaseTypes]:
        params = self.sample_preprocessing_params(
            trial,
            target_options=["stdnorm", "quantile"],
            marker_options=["maf"],
            feature_selection_options=["passthrough", "relief"],
        )

        params["embed_nunits"] = trial.suggest_int(
            "embed_nunits",
            5,
            100,
            step=5
        )

        params["marker_embed_nlayers"] = trial.suggest_int(
            "marker_embed_nlayers", 1, 4
        )

        if len(self.grouping_columns) > 0:
            params["env_embed_nlayers"] = trial.suggest_int(
                "env_embed_nlayers", 1, 4
            )

            pass

        params["postembed_nlayers"] = trial.suggest_int(
            "postembed_nlayers", 0, 2
        )

        if params["postembed_nlayers"] > 0:
            params["postembed_nunits"] = trial.suggest_int(
                "postembed_nunits",
                5,
                100,
                step=5
            )

        params["dropout_rate"] = trial.suggest_float("dropout_rate", 0, 1)

        params["marker_l1"] = trial.suggest_float(
            "marker_l1",
            1e-50,
            50,
            log=True
        )
        params["marker_l2"] = trial.suggest_float(
            "marker_l2",
            1e-50,
            50,
            log=True
        )

        if len(self.grouping_columns) > 0:
            params["group_l1"] = trial.suggest_float(
                "group_l1",
                1e-50,
                50,
                log=True
            )
            params["group_l2"] = trial.suggest_float(
                "group_l2",
                1e-50,
                50,
                log=True
            )

        return params

    def model(self, params: Dict[str, Any]):
        from selectml.tf.layers import AddChannel, Flatten1D, ConvLinkage
        from selectml.tf.models import SSModel
        from tensorflow.keras.layers import Dropout, Dense
        from tensorflow.keras.regularizers import L1L2
        from tensorflow.keras.models import Sequential

        (
            target_trans,
            grouping_trans,
            marker_trans
        ) = self.sample_preprocessing_model(params)

        marker_model = Sequential([
            AddChannel(),
            ConvLinkage(
                nlayers=params["conv_nlayers"],
                filters=params["conv_filters"],
                strides=params["conv_strides"],
                kernel_size=params["conv_kernel_size"],
                activation=params["conv_activation"],
                activation_first=params["conv_activation"],
                activation_last=params["conv_activation"],
                kernel_regularizer=L1L2(
                    params["conv_l1"],
                    params["conv_l2"],
                ),
                use_bn=params["conv_use_bn"]
            ),
            Flatten1D(),
            Dropout(params["dropout_rate"]),
            Dense(1, use_bias=True, kernel_regularizer=L1L2(
                params["marker_l1"],
                params["marker_l2"])
            ),
        ])

        if len(self.grouping_columns) > 0:
            group_model = Dense(1, use_bias=True, kernel_regularizer=L1L2(
                params["group_l1"],
                params["group_l2"])
            )
        else:
            group_model = None

        model = SSModel(
            marker_embedder=marker_model,
            env_embedder=group_model,
            combiner="add",
        )

        return TFWrapper(
            model=model,
            target_trans=target_trans,
            group_trans=grouping_trans,
            marker_trans=marker_trans,
        )

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
        ]
