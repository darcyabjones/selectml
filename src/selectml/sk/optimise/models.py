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
from .base import SKModel, TFBaseModel, BGLRBaseModel

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
                n_jobs=-1,
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
            feature_selection_options=["drop", "passthrough", "rf", "relief"],
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
                n_jobs=-1
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
                n_jobs=-1,
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
            "max_samples": trial.suggest_int("max_samples", 50, 1000),
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
                n_jobs=-1,
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
                "max_samples": 100,
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
                1e-6,
                1.0,
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
            "epsilon": trial.suggest_float("epsilon", 0, 5),
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
                    max_iter=10 * len(self.marker_columns),
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
            target_options=["stdnorm", "quantile"],
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
                    max_iter=10 * len(self.marker_columns),
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
                "alpha": 1,
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
                "alpha": 1,
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
                "alpha": 1,
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
                "alpha": 1,
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
                "alpha": 1,
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
                "alpha": 1,
                "l1_ratio": 1.0,
            },
        ]


class LassoLarsDistModel(SKModel):

    use_weights: bool = False

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, BaseTypes]:
        params = self.sample_preprocessing_params(
            trial,
            target_options=["stdnorm", "quantile"],
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
                    max_iter=10 * len(self.marker_columns),
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
                "alpha": 1,
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
                "alpha": 1,
            },
        ]


class LassoLarsModel(SKModel):

    use_weights: bool = False

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, BaseTypes]:
        params = self.sample_preprocessing_params(
            trial,
            target_options=["stdnorm", "quantile"],
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
                    max_iter=10 * len(self.marker_columns),
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
                "alpha": 1,
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
                "alpha": 1,
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
            },
        ]


class ElasticNetModel(SKModel):

    use_weights: bool = True

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, BaseTypes]:
        params = self.sample_preprocessing_params(
            trial,
            target_options=["stdnorm", "quantile"],
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
                    max_iter=10 * len(self.marker_columns),
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
                "alpha": 1.0,
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
                "alpha": 10.0,
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
                "alpha": 1.0,
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
                "alpha": 1.0,
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
        proportion=0.1
    ):
        rng = np.random.default_rng()

        uindivs = np.unique(individuals)
        n = min([1, np.floor(len(uindivs) * proportion)])
        val_samples = set(rng.choice(uindivs, size=n, replace=False))

        val = np.array([(i in val_samples) for i in individuals])

        val_tup = (
            X_markers[val],
            X_grouping[val] if X_grouping is not None else None,
            y[val],
            individuals[val],
            sample_weight[val] if sample_weight is not None else None,
        )

        train_tup = (
            X_markers[~val],
            X_grouping[~val] if X_grouping is not None else None,
            y[~val],
            individuals[~val],
            sample_weight[~val] if sample_weight is not None else None,
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

        train_ch, val_ch = self._np_to_channels_validation(
            X_markers,
            X_grouping,
            y,
            individuals,
            sample_weight,
            proportion=0.1
        )

        early_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.1,
            patience=20,
            restore_best_weights=True,
        )

        self.model.compile(optimizer="adam", loss="mae", metrics=["mae"])

        _ = self.model.fit(
            train_ch.shuffle(1024).batch(64),
            epochs=500,
            verbose=0,
            validation_data=val_ch.batch(64),
            callbacks=[early_callback]
        )
        return self

    def predict(self, X: "Tuple[npt.ArrayLike, Optional[npt.ArrayLike]]"):
        X_markers_, X_grouping_ = X
        X_markers = np.array(X_markers_)
        if X_grouping_ is not None:
            X_grouping: Optional[np.ndarray] = np.array(X_grouping_)
        else:
            X_grouping = None
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

    def _np_to_channels_validation(
        self,
        X_markers,
        X_grouping,
        y,
        individuals,
        sample_weights,
        proportion=0.1,
    ):
        train_tup, val_tup = self.get_validation_samples(
            X_markers,
            X_grouping,
            y,
            individuals,
            sample_weights
        )

        (mar, gro, y_, indiv, wei) = train_tup
        train_ch = self._np_to_channels(mar, gro, y_, wei)

        (mar, gro, y_, indiv, wei) = val_tup
        val_ch = self._np_to_channels(mar, gro, y_, wei)
        return train_ch, val_ch

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

        nmarkers = len(self.marker_columns)
        if "feature_selection_nfeatures" in params:
            assert isinstance(params["feature_selection_nfeatures"], int)
            nmarkers = params["feature_selection_nfeatures"]

        if nmarkers < 30:
            max_conv_levels = 0
        elif nmarkers < 100:
            max_conv_levels = 1
        elif nmarkers < 200:
            max_conv_levels = 2
        else:
            max_conv_levels = 3

        params["conv_nlayers"] = trial.suggest_int(
            "conv_nlayers",
            0,
            max_conv_levels
        )
        assert isinstance(params["conv_nlayers"], int)
        if params["conv_nlayers"] > 0:
            params["conv_filters"] = trial.suggest_int("conv_filters", 1, 4)
            params["conv_kernel_size"] = trial.suggest_int(
                "conv_kernel_size",
                2,
                5
            )

            assert isinstance(params["conv_kernel_size"], int)
            max_stride = params["conv_kernel_size"]
            while (
                (nmarkers / (max_stride ** params["conv_nlayers"]))
                < (2 * params["conv_kernel_size"])
            ):
                # Avoid zero-division error in whie
                if max_stride <= 1:
                    max_stride = 1
                    break
                max_stride -= 1

            params["conv_strides"] = trial.suggest_int(
                "conv_strides",
                2,
                max_stride
            )
            params["conv_l1"] = trial.suggest_float(
                "conv_l1", 1e-50, 50, log=True)
            params["conv_l2"] = trial.suggest_float(
                "conv_l2", 1e-50, 50, log=True)
            params["conv_activation"] = trial.suggest_categorical(
                "conv_activation",
                ["linear", "relu"]
            )
            params["conv_use_bn"] = trial.suggest_categorical(
                "conv_use_bn",
                [True, False]
            )

        params["dropout_rate"] = trial.suggest_float("dropout_rate", 0, 1)

        params["marker_adaptaptive_l1"] = trial.suggest_categorical(
            "marker_adaptaptive_l1", [True, False]
        )

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

        params["use_rank_loss"] = trial.suggest_categorical(
            "use_rank_loss",
            [True, False]
        )
        if params["use_rank_loss"]:
            params["rank_loss"] = trial.suggest_float(
                "rank_loss",
                1e-20, 5, log=True
            )

        return params

    def model(self, params: Dict[str, Any]):
        from selectml.tf.layers import AddChannel, Flatten1D, ConvLinkage
        from selectml.tf.models import SSModel
        from selectml.tf.regularizers import AdaptiveL1L2Regularizer
        from tensorflow.keras.layers import Dropout, Dense
        from tensorflow.keras.regularizers import L1L2
        from tensorflow.keras.models import Sequential

        import tensorflow as tf
        tf.random.set_seed(self.seed)
        (
            target_trans,
            grouping_trans,
            marker_trans
        ) = self.sample_preprocessing_model(params)

        marker_model_list = []
        if params["conv_nlayers"] > 0:
            marker_model_list.extend([
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
            ])

        if params["marker_adaptaptive_l1"]:
            reg = AdaptiveL1L2Regularizer(
                l1=params["marker_l1"],
                l2=params["marker_l2"],
                adapt=True
            )
        else:
            reg = L1L2(
                params["marker_l1"],
                params["marker_l2"]
            )

        marker_model_list.extend([
            Dropout(params["dropout_rate"]),
            Dense(1, use_bias=True, kernel_regularizer=reg),
        ])

        marker_model = Sequential(marker_model_list)

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
            rank_loss=params.get("rank_loss", None),
        )

        return TFWrapper(
            model=model,
            target_trans=target_trans,
            group_trans=grouping_trans,
            marker_trans=marker_trans,
        )

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
            {
                "train_means": True,
                "weight": "none",
                "target_options": "stdnorm",
                "min_maf": 0.05,
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "conv_nlayers": 2,
                "conv_filters": 1,
                "conv_kernel_size": 3,
                "conv_strides": 2,
                "conv_l1": 1e-50,
                "conv_l2": 1e-50,
                "conv_activation": "linear",
                "conv_use_bn": False,
                "dropout_rate": 0.5,
                "marker_adaptaptive_l1": False,
                "marker_l1": 1e-50,
                "marker_l2": 1e-50,
                "group_l1": 1e-50,
                "group_l2": 1e-50,
                "use_rank_loss": False,
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "stdnorm",
                "min_maf": 0.05,
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "conv_nlayers": 0,
                "dropout_rate": 0.5,
                "marker_adaptaptive_l1": False,
                "marker_l1": 1e-50,
                "marker_l2": 1e-50,
                "group_l1": 1e-50,
                "group_l2": 1e-50,
                "use_rank_loss": False,
            }
        ]


class MLPModel(TFBaseModel):

    use_weights: bool = True

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, BaseTypes]:  # noqa
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

        assert isinstance(params["marker_embed_nlayers"], int)
        for i in range(params["marker_embed_nlayers"]):
            params[f"marker_embed{i}_activation"] = trial.suggest_categorical(
                f"marker_embed{i}_activation",
                ["linear", "relu"]
            )

            if params[f"marker_embed{i}_activation"] != "linear":
                params[f"marker_embed{i}_res"] = trial.suggest_categorical(
                    f"marker_embed{i}_res",
                    [True, False]
                )

        params["marker_embed_adaptive_l1"] = trial.suggest_categorical(
            "marker_embed_adaptive_l1",
            [True, False]
        )

        assert isinstance(params["marker_embed_adaptive_l1"], bool)
        if params["marker_embed_adaptive_l1"]:
            params["marker_embed_adaptive_l1_rate"] = trial.suggest_float(
                "marker_embed_adaptive_l1_rate",
                1e-10,
                50
            )

            params["marker_embed_adaptive_l2_rate"] = trial.suggest_float(
                "marker_embed_adaptive_l2_rate",
                1e-50,
                50
            )

        params["marker_embed0_dropout_rate"] = trial.suggest_float(
            "marker_embed0_dropout_rate",
            0, 1
        )

        params["marker_embed0_l1"] = trial.suggest_float(
            "marker_embed0_l1",
            1e-50,
            50,
            log=True
        )
        params["marker_embed0_l2"] = trial.suggest_float(
            "marker_embed0_l2",
            1e-50,
            50,
            log=True
        )

        assert isinstance(params["marker_embed_nlayers"], int)
        if params["marker_embed_nlayers"] > 1:
            params["marker_embed1_l1"] = trial.suggest_float(
                "marker_embed1_l1",
                1e-50,
                50,
                log=True
            )
            params["marker_embed1_l2"] = trial.suggest_float(
                "marker_embed0_l2",
                1e-50,
                50,
                log=True
            )

            params["marker_embed1_dropout_rate"] = trial.suggest_float(
                "marker_embed1_dropout_rate",
                0, 1
            )

        if len(self.grouping_columns) > 0:
            params["env_embed_nlayers"] = trial.suggest_int(
                "env_embed_nlayers", 1, 4
            )

            assert isinstance(params["env_embed_nlayers"], int)
            for i in range(params["env_embed_nlayers"]):
                params[f"env_embed{i}_activation"] = trial.suggest_categorical(
                    f"env_embed{i}_activation",
                    ["linear", "relu"]
                )
                if params[f"env_embed{i}_activation"] != "linear":
                    params[f"env_embed{i}_res"] = trial.suggest_categorical(
                        f"env_embed{i}_res",
                        [True, False]
                    )

            params["env_embed0_l2"] = trial.suggest_float(
                "env_embed0_l2",
                1e-50,
                50,
                log=True
            )

            params["env_embed0_dropout_rate"] = trial.suggest_float(
                "env_embed0_dropout_rate",
                0,
                1
            )

            assert isinstance(params["env_embed_nlayers"], int)
            if params["env_embed_nlayers"] > 1:
                params["env_embed1_l2"] = trial.suggest_float(
                    "env_embed1_l2",
                    1e-50,
                    50,
                    log=True
                )

                params["env_embed1_dropout_rate"] = trial.suggest_float(
                    "env_embed1_dropout_rate",
                    0,
                    1
                )

            params["combiner"] = trial.suggest_categorical(
                "combiner",
                ["add", "concat"]
            )

        params["postembed_nlayers"] = trial.suggest_int(
            "postembed_nlayers", 0, 2
        )

        assert isinstance(params["postembed_nlayers"], int)
        for i in range(params["postembed_nlayers"]):
            params[f"postembed{i}_activation"] = trial.suggest_categorical(
                f"postembed{i}_activation",
                ["linear", "relu"]
            )
            if params[f"postembed{i}_activation"] != "linear":
                params[f"postembed{i}_res"] = trial.suggest_categorical(
                    f"postembed{i}_res",
                    [True, False]
                )

        assert isinstance(params["postembed_nlayers"], int)
        if params["postembed_nlayers"] > 0:
            assert isinstance(params["embed_nunits"], int)
            params["postembed_nunits"] = trial.suggest_int(
                "postembed_nunits",
                5,
                params["embed_nunits"],
                step=5
            )

        params["postembed_dropout_rate"] = trial.suggest_float(
            "postembed_dropout_rate",
            0, 1
        )

        params["postembed_l1"] = trial.suggest_float(
            "postembed_l1",
            1e-50,
            50,
            log=True
        )

        params["postembed_l2"] = trial.suggest_float(
            "postembed_l2",
            1e-50,
            50,
            log=True
        )

        params["marker_embed_regularizer"] = trial.suggest_categorical(
            "marker_embed_regularizer",
            ["none", "semihard", "hard", "relief"]
        )

        if params["marker_embed_regularizer"] == "semihard":
            params["marker_semihard_loss"] = trial.suggest_float(
                "marker_semihard_loss_rate",
                1e-20, 5, log=True
            )
        elif params["marker_embed_regularizer"] == "hard":
            params["marker_hard_loss"] = trial.suggest_float(
                "marker_hard_loss",
                1e-20, 5, log=True
            )
        elif params["marker_embed_regularizer"] == "relief":
            params["marker_relief_loss"] = trial.suggest_float(
                "marker_relief_loss",
                1e-20, 5, log=True
            )

        params["use_rank_loss"] = trial.suggest_categorical(
            "use_rank_loss",
            [True, False]
        )
        if params["use_rank_loss"]:
            params["rank_loss"] = trial.suggest_float(
                "rank_loss",
                1e-20, 5, log=True
            )

        if len(self.grouping_columns) > 0:
            params["env_embed_regularizer"] = trial.suggest_categorical(
                "env_embed_regularizer",
                ["none", "semihard", "hard"]
            )
            if params["env_embed_regularizer"] == "hard":
                params["env_hard_loss"] = trial.suggest_float(
                    "env_hard_loss",
                    1e-20, 5, log=True
                )
            elif params["env_embed_regularizer"] == "semihard":
                params["env_semihard_loss"] = trial.suggest_float(
                    "env_semihard_loss_rate",
                    1e-20, 5, log=True
                )

        return params

    def model(self, params: Dict[str, Any]):
        from selectml.tf.layers import (
            LocalLasso,
            AddChannel,
            Flatten1D,
            ParallelResidualUnit,
            ResidualUnit
        )
        from selectml.tf.models import SSModel
        from selectml.tf.regularizers import AdaptiveL1L2Regularizer
        from tensorflow.keras.layers import (
            Dropout,
            Dense,
            BatchNormalization,
            ReLU
        )
        from tensorflow.keras.regularizers import L2, L1L2
        from tensorflow.keras.models import Sequential
        import tensorflow as tf
        tf.random.set_seed(self.seed)

        (
            target_trans,
            grouping_trans,
            marker_trans
        ) = self.sample_preprocessing_model(params)

        marker_model = []

        if params["marker_embed_adaptive_l1"]:
            reg = AdaptiveL1L2Regularizer(
                l1=params["marker_embed_adaptive_l1_rate"],
                l2=params["marker_embed_adaptive_l2_rate"],
                adapt=True,
            )
            marker_model.extend([
                AddChannel(),
                LocalLasso(kernel_regularizer=reg),
                Flatten1D()
            ])

        for i in range(params["marker_embed_nlayers"]):
            j = 0 if (i == 0) else 1
            marker_model.append(
                Dropout(params[f"marker_embed{j}_dropout_rate"])
            )

            reg = L1L2(
                params[f"marker_embed{j}_l1"],
                params[f"marker_embed{j}_l2"],
            )

            if params.get(f"marker_embed{i}_res", False):
                type_ = ParallelResidualUnit if (j == 0) else ResidualUnit
                marker_model.append(type_(
                    params["embed_nunits"],
                    params[f"marker_embed{j}_dropout_rate"],
                    activation=params[f"marker_embed{i}_activation"],
                    use_bias=True,
                    nonlinear_kernel_regularizer=reg,
                    gain_kernel_regularizer=reg,
                    linear_kernel_regularizer=reg,
                ))
            else:
                marker_model.append(Dense(
                    params["embed_nunits"],
                    activation="linear",
                    kernel_regularizer=reg,
                    use_bias=True
                ))
                marker_model.append(BatchNormalization())

                # The current trend seems to be to do activation after
                # batch/layer norm. This might change in future.
                if params[f"marker_embed{i}_activation"] == "relu":
                    marker_model.append(ReLU())

        if len(self.grouping_columns) > 0:
            env_model: Optional[List] = []
            assert env_model is not None

            for i in range(params["env_embed_nlayers"]):
                j = 0 if (i == 0) else 1
                env_model.append(
                    Dropout(params[f"env_embed{j}_dropout_rate"])
                )

                reg = L2(
                    params[f"env_embed{j}_l2"],
                )

                if params.get(f"env_embed{i}_res", False):
                    type_ = ParallelResidualUnit if (j == 0) else ResidualUnit
                    env_model.append(type_(
                        params["embed_nunits"],
                        params[f"env_embed{j}_dropout_rate"],
                        activation=params[f"env_embed{i}_activation"],
                        use_bias=True,
                        nonlinear_kernel_regularizer=reg,
                        gain_kernel_regularizer=reg,
                        linear_kernel_regularizer=reg,
                    ))
                else:
                    env_model.append(Dense(
                        params["embed_nunits"],
                        activation="linear",
                        kernel_regularizer=reg,
                        use_bias=True
                    ))
                    env_model.append(BatchNormalization())
                    if params[f"env_embed{i}_activation"] == "relu":
                        env_model.append(ReLU())
        else:
            env_model = None

        postembed_model = []

        reg = L1L2(
            params["postembed_l1"],
            params["postembed_l2"],
        )

        for i in range(params["postembed_nlayers"]):
            j = 0 if (i == 0) else 1
            postembed_model.append(
                Dropout(params["postembed_dropout_rate"])
            )

            if params.get(f"postembed{i}_res", False):
                type_ = ParallelResidualUnit if (j == 0) else ResidualUnit
                postembed_model.append(type_(
                    params["postembed_nunits"],
                    params["postembed_dropout_rate"],
                    activation=params[f"postembed{i}_activation"],
                    use_bias=True,
                    nonlinear_kernel_regularizer=reg,
                    gain_kernel_regularizer=reg,
                    linear_kernel_regularizer=reg,
                ))
            else:
                postembed_model.append(Dense(
                    params["postembed_nunits"],
                    activation="linear",
                    kernel_regularizer=reg,
                    use_bias=True
                ))
                postembed_model.append(BatchNormalization())

                # The current trend seems to be to do activation after
                # batch/layer norm. This might change in future.
                if params[f"postembed{i}_activation"] == "relu":
                    postembed_model.append(ReLU())

        postembed_model.extend([
            Dropout(params["postembed_dropout_rate"]),
            Dense(
                1,
                activation="linear",
                kernel_regularizer=reg,
                use_bias=True
            )
        ])

        model = SSModel(
            marker_embedder=Sequential(marker_model),
            env_embedder=None if env_model is None else Sequential(env_model),
            post_embedder=Sequential(postembed_model),
            combiner=params.get("combiner", "add"),
            marker_semihard_loss=params.get("marker_semihard_loss", None),
            marker_hard_loss=params.get("marker_hard_loss", None),
            marker_relief_loss=params.get("marker_relief_loss", None),
            rank_loss=params.get("rank_loss", None),
            env_semihard_loss=params.get("env_semihard_loss", None),
            env_hard_loss=params.get("env_hard_loss", None),
        )

        return TFWrapper(
            model=model,
            target_trans=target_trans,
            group_trans=grouping_trans,
            marker_trans=marker_trans,
        )

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
            {
                "train_means": True,
                "weight": "none",
                "target_options": "stdnorm",
                "min_maf": 0.05,
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "embed_nunits": 10,
                "marker_embed_nlayers": 1,
                "marker_embed0_activation": "linear",
                "marker_embed_adaptive_l1": False,
                "marker_embed0_dropout_rate": 0.5,
                "marker_embed0_l1": 1e-50,
                "marker_embed0_l2": 1e-50,
                "env_embed_nlayers": 1,
                "env_embed0_activation": "linear",
                "env_embed0_dropout_rate": 0.0,
                "env_embed0_l2": 1e-50,
                "combiner": "add",
                "postembed_nlayers": 0,
                "postembed_dropout_rate": 0.2,
                "postembed_l1": 1e-50,
                "postembed_l2": 1e-50,
                "use_rank_loss": False,
                "marker_embed_regularizer": "none",
                "env_embed_regularizer": "none",
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "stdnorm",
                "min_maf": 0.05,
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "embed_nunits": 10,
                "marker_embed_nlayers": 1,
                "marker_embed0_activation": "linear",
                "marker_embed_adaptive_l1": False,
                "marker_embed0_dropout_rate": 0.5,
                "marker_embed0_l1": 1e-50,
                "marker_embed0_l2": 1e-50,
                "env_embed_nlayers": 1,
                "env_embed0_activation": "linear",
                "env_embed0_dropout_rate": 0.0,
                "env_embed0_l2": 1e-50,
                "combiner": "add",
                "postembed_nlayers": 1,
                "postembed_nunits": 10,
                "postembed_dropout_rate": 0.2,
                "postembed_l1": 1e-50,
                "postembed_l2": 1e-50,
                "use_rank_loss": False,
                "marker_embed_regularizer": "none",
                "env_embed_regularizer": "none",
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "stdnorm",
                "min_maf": 0.05,
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "embed_nunits": 10,
                "marker_embed_nlayers": 1,
                "marker_embed0_activation": "linear",
                "marker_embed_adaptive_l1": True,
                "marker_embed_adaptive_l1_rate": 1e-3,
                "marker_embed_adaptive_l2_rate": 1e-9,
                "marker_embed0_dropout_rate": 0.5,
                "marker_embed0_l1": 1e-50,
                "marker_embed0_l2": 1e-50,
                "env_embed_nlayers": 1,
                "env_embed0_activation": "linear",
                "env_embed0_dropout_rate": 0.0,
                "env_embed0_l2": 1e-50,
                "combiner": "add",
                "postembed_nlayers": 0,
                "postembed_dropout_rate": 0.2,
                "postembed_l1": 1e-50,
                "postembed_l2": 1e-50,
                "use_rank_loss": False,
                "marker_embed_regularizer": "semihard",
                "marker_semihard_loss": 0.5,
                "env_embed_regularizer": "none",
            },
        ]


class ConvMLPModel(TFBaseModel):

    use_weights: bool = True

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, BaseTypes]:  # noqa
        params = self.sample_preprocessing_params(
            trial,
            target_options=["stdnorm", "quantile"],
            marker_options=["maf"],
            feature_selection_options=["passthrough", "relief"],
        )

        nmarkers = len(self.marker_columns)
        if "feature_selection_nfeatures" in params:
            assert isinstance(params["feature_selection_nfeatures"], int)
            nmarkers = params["feature_selection_nfeatures"]

        if nmarkers < 30:
            max_conv_levels = 0
        elif nmarkers < 100:
            max_conv_levels = 1
        elif nmarkers < 200:
            max_conv_levels = 2
        else:
            max_conv_levels = 3

        params["conv_nlayers"] = trial.suggest_int(
            "conv_nlayers",
            0,
            max_conv_levels
        )

        assert isinstance(params["conv_nlayers"], int)
        if params["conv_nlayers"] > 0:
            params["conv_filters"] = trial.suggest_int("conv_filters", 1, 4)
            params["conv_kernel_size"] = trial.suggest_int(
                "conv_kernel_size",
                2,
                5
            )

            assert isinstance(params["conv_kernel_size"], int)
            max_stride = params["conv_kernel_size"]
            while (
                (nmarkers / (max_stride ** params["conv_nlayers"]))
                < (2 * params["conv_kernel_size"])
            ):
                # Avoid zero-division error in whie
                if max_stride <= 1:
                    max_stride = 1
                    break
                max_stride -= 1

            params["conv_strides"] = trial.suggest_int(
                "conv_strides",
                2,
                max_stride
            )
            params["conv_l1"] = trial.suggest_float(
                "conv_l1", 1e-50, 50, log=True)
            params["conv_l2"] = trial.suggest_float(
                "conv_l2", 1e-50, 50, log=True)
            params["conv_activation"] = trial.suggest_categorical(
                "conv_activation",
                ["linear", "relu"]
            )
            params["conv_use_bn"] = trial.suggest_categorical(
                "conv_use_bn",
                [True, False]
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

        assert isinstance(params["marker_embed_nlayers"], int)
        for i in range(params["marker_embed_nlayers"]):
            params[f"marker_embed{i}_activation"] = trial.suggest_categorical(
                f"marker_embed{i}_activation",
                ["linear", "relu"]
            )

            if params[f"marker_embed{i}_activation"] != "linear":
                params[f"marker_embed{i}_res"] = trial.suggest_categorical(
                    f"marker_embed{i}_res",
                    [True, False]
                )

        params["marker_embed_adaptive_l1"] = trial.suggest_categorical(
            "marker_embed_adaptive_l1",
            [True, False]
        )

        params["marker_embed_adaptive_l1_rate"] = trial.suggest_float(
            "marker_embed_adaptive_l1_rate",
            1e-10,
            50
        )

        params["marker_embed_adaptive_l2_rate"] = trial.suggest_float(
            "marker_embed_adaptive_l2_rate",
            1e-50,
            50
        )

        params["marker_embed0_dropout_rate"] = trial.suggest_float(
            "marker_embed0_dropout_rate",
            0, 1
        )

        params["marker_embed0_l1"] = trial.suggest_float(
            "marker_embed0_l1",
            1e-50,
            50,
            log=True
        )
        params["marker_embed0_l2"] = trial.suggest_float(
            "marker_embed0_l2",
            1e-50,
            50,
            log=True
        )

        assert isinstance(params["marker_embed_nlayers"], int)
        if params["marker_embed_nlayers"] > 1:
            params["marker_embed1_l1"] = trial.suggest_float(
                "marker_embed1_l1",
                1e-50,
                50,
                log=True
            )
            params["marker_embed1_l2"] = trial.suggest_float(
                "marker_embed0_l2",
                1e-50,
                50,
                log=True
            )

            params["marker_embed1_dropout_rate"] = trial.suggest_float(
                "marker_embed1_dropout_rate",
                0, 1
            )

        if len(self.grouping_columns) > 0:
            params["env_embed_nlayers"] = trial.suggest_int(
                "env_embed_nlayers", 1, 4
            )

            assert isinstance(params["env_embed_nlayers"], int)
            for i in range(params["env_embed_nlayers"]):
                params[f"env_embed{i}_activation"] = trial.suggest_categorical(
                    f"env_embed{i}_activation",
                    ["linear", "relu"]
                )
                if params[f"env_embed{i}_activation"] != "linear":
                    params[f"env_embed{i}_res"] = trial.suggest_categorical(
                        f"env_embed{i}_res",
                        [True, False]
                    )

            params["env_embed0_l2"] = trial.suggest_float(
                "env_embed0_l2",
                1e-50,
                50,
                log=True
            )

            params["env_embed0_dropout_rate"] = trial.suggest_float(
                "env_embed1_dropout_rate",
                0,
                1
            )

            assert isinstance(params["env_embed_nlayers"], int)
            if params["env_embed_nlayers"] > 1:
                params["env_embed1_l2"] = trial.suggest_float(
                    "env_embed1_l2",
                    1e-50,
                    50,
                    log=True
                )

                params["env_embed1_dropout_rate"] = trial.suggest_float(
                    "env_embed1_dropout_rate",
                    0,
                    1
                )

            params["combiner"] = trial.suggest_categorical(
                "combiner",
                ["add", "concat"]
            )

        params["postembed_nlayers"] = trial.suggest_int(
            "postembed_nlayers", 0, 2
        )

        assert isinstance(params["postembed_nlayers"], int)
        for i in range(params["postembed_nlayers"]):
            params[f"postembed{i}_activation"] = trial.suggest_categorical(
                f"postembed{i}_activation",
                ["linear", "relu"]
            )
            if params[f"postembed{i}_activation"] != "linear":
                params[f"postembed{i}_res"] = trial.suggest_categorical(
                    f"postembed{i}_res",
                    [True, False]
                )

        assert isinstance(params["postembed_nlayers"], int)
        if params["postembed_nlayers"] > 0:
            assert isinstance(params["embed_nunits"], int)
            params["postembed_nunits"] = trial.suggest_int(
                "postembed_nunits",
                5,
                params["embed_nunits"],
                step=5
            )

        params["postembed_dropout_rate"] = trial.suggest_float(
            "postembed_dropout_rate",
            0, 1
        )

        params["postembed_l1"] = trial.suggest_float(
            "postembed_l1",
            1e-50,
            50,
            log=True
        )

        params["postembed_l2"] = trial.suggest_float(
            "postembed_l2",
            1e-50,
            50,
            log=True
        )

        params["marker_embed_regularizer"] = trial.suggest_categorical(
            "marker_embed_regularizer",
            ["none", "semihard", "hard", "relief"]
        )

        if params["marker_embed_regularizer"] == "semihard":
            params["marker_semihard_loss"] = trial.suggest_float(
                "marker_semihard_loss_rate",
                1e-20, 5, log=True
            )
        elif params["marker_embed_regularizer"] == "hard":
            params["marker_hard_loss"] = trial.suggest_float(
                "marker_hard_loss",
                1e-20, 5, log=True
            )
        elif params["marker_embed_regularizer"] == "relief":
            params["marker_relief_loss"] = trial.suggest_float(
                "marker_relief_loss",
                1e-20, 5, log=True
            )

        params["use_rank_loss"] = trial.suggest_categorical(
            "use_rank_loss",
            [True, False]
        )
        if params["use_rank_loss"]:
            params["rank_loss"] = trial.suggest_float(
                "rank_loss",
                1e-20, 5, log=True
            )

        if len(self.grouping_columns) > 0:
            params["env_embed_regularizer"] = trial.suggest_categorical(
                "env_embed_regularizer",
                ["none", "semihard", "hard"]
            )
            if params["env_embed_regularizer"] == "hard":
                params["env_hard_loss"] = trial.suggest_float(
                    "env_hard_loss",
                    1e-20, 5, log=True
                )
            elif params["env_embed_regularizer"] == "semihard":
                params["env_semihard_loss"] = trial.suggest_float(
                    "env_semihard_loss_rate",
                    1e-20, 5, log=True
                )

        return params

    def model(self, params: Dict[str, Any]):  # noqa
        from selectml.tf.layers import (
            LocalLasso,
            AddChannel,
            Flatten1D,
            ParallelResidualUnit,
            ResidualUnit,
            ConvLinkage
        )
        from selectml.tf.models import SSModel
        from selectml.tf.regularizers import AdaptiveL1L2Regularizer
        from tensorflow.keras.layers import (
            Dropout,
            Dense,
            BatchNormalization,
            ReLU
        )
        from tensorflow import tf
        from tensorflow.keras.regularizers import L2, L1L2
        from tensorflow.keras.models import Sequential
        tf.random.set_seed(self.seed)

        (
            target_trans,
            grouping_trans,
            marker_trans
        ) = self.sample_preprocessing_model(params)

        marker_model = []

        if params["conv_nlayers"] > 0:
            marker_model.extend([
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
            ])

        if params["marker_embed_adaptive_l1"]:
            reg = AdaptiveL1L2Regularizer(
                l1=params["marker_embed_adaptive_l1_rate"],
                l2=params["marker_embed_adaptive_l2_rate"],
                adapt=True,
            )
            marker_model.extend([
                AddChannel(),
                LocalLasso(kernel_regularizer=reg),
                Flatten1D()
            ])

        for i in range(params["marker_embed_nlayers"]):
            j = 0 if (i == 0) else 1
            marker_model.append(
                Dropout(params[f"marker_embed{j}_dropout_rate"])
            )

            reg = L1L2(
                params[f"marker_embed{j}_l1"],
                params[f"marker_embed{j}_l2"],
            )

            if params.get(f"marker_embed{i}_res", False):
                type_ = ParallelResidualUnit if (j == 0) else ResidualUnit
                marker_model.append(type_(
                    params["embed_nunits"],
                    params[f"marker_embed{j}_dropout_rate"],
                    activation=params[f"marker_embed{i}_activation"],
                    use_bias=True,
                    nonlinear_kernel_regularizer=reg,
                    gain_kernel_regularizer=reg,
                    linear_kernel_regularizer=reg,
                ))
            else:
                marker_model.append(Dense(
                    params["embed_nunits"],
                    activation="linear",
                    kernel_regularizer=reg,
                    use_bias=True
                ))
                marker_model.append(BatchNormalization())

                # The current trend seems to be to do activation after
                # batch/layer norm. This might change in future.
                if params[f"marker_embed{i}_activation"] == "relu":
                    marker_model.append(ReLU())

        if len(self.grouping_columns) > 0:
            env_model: Optional[List] = []
            assert env_model is not None

            for i in range(params["env_embed_nlayers"]):
                j = 0 if (i == 0) else 1
                env_model.append(
                    Dropout(params[f"env_embed{j}_dropout_rate"])
                )

                reg = L2(
                    params[f"env_embed{j}_l2"],
                )

                if params.get(f"env_embed{i}_res", False):
                    type_ = ParallelResidualUnit if (j == 0) else ResidualUnit
                    env_model.append(type_(
                        params["embed_nunits"],
                        params[f"env_embed{j}_dropout_rate"],
                        activation=params[f"env_embed{i}_activation"],
                        use_bias=True,
                        nonlinear_kernel_regularizer=reg,
                        gain_kernel_regularizer=reg,
                        linear_kernel_regularizer=reg,
                    ))
                else:
                    env_model.append(Dense(
                        params["embed_nunits"],
                        activation="linear",
                        kernel_regularizer=reg,
                        use_bias=True
                    ))
                    env_model.append(BatchNormalization())
                    if params[f"env_embed{i}_activation"] == "relu":
                        env_model.append(ReLU())
        else:
            env_model = None

        postembed_model = []

        reg = L1L2(
            params["postembed_l1"],
            params["postembed_l2"],
        )

        for i in range(params["postembed_nlayers"]):
            j = 0 if (i == 0) else 1
            postembed_model.append(
                Dropout(params["postembed_dropout_rate"])
            )

            if params.get(f"postembed{i}_res", False):
                type_ = ParallelResidualUnit if (j == 0) else ResidualUnit
                postembed_model.append(type_(
                    params["postembed_nunits"],
                    params["postembed_dropout_rate"],
                    activation=params[f"postembed{i}_activation"],
                    use_bias=True,
                    nonlinear_kernel_regularizer=reg,
                    gain_kernel_regularizer=reg,
                    linear_kernel_regularizer=reg,
                ))
            else:
                postembed_model.append(Dense(
                    params["postembed_nunits"],
                    activation="linear",
                    kernel_regularizer=reg,
                    use_bias=True
                ))
                postembed_model.append(BatchNormalization())

                # The current trend seems to be to do activation after
                # batch/layer norm. This might change in future.
                if params[f"postembed{i}_activation"] == "relu":
                    postembed_model.append(ReLU())

        postembed_model.extend([
            Dropout(params["postembed_dropout_rate"]),
            Dense(
                1,
                activation="linear",
                kernel_regularizer=reg,
                use_bias=True
            )
        ])

        model = SSModel(
            marker_embedder=Sequential(marker_model),
            env_embedder=None if env_model is None else Sequential(env_model),
            post_embedder=Sequential(postembed_model),
            combiner=params.get("combiner", "add"),
            marker_semihard_loss=params.get("marker_semihard_loss", None),
            marker_hard_loss=params.get("marker_hard_loss", None),
            marker_relief_loss=params.get("marker_relief_loss", None),
            rank_loss=params.get("rank_loss", None),
            env_semihard_loss=params.get("env_semihard_loss", None),
            env_hard_loss=params.get("env_hard_loss", None),
        )

        return TFWrapper(
            model=model,
            target_trans=target_trans,
            group_trans=grouping_trans,
            marker_trans=marker_trans,
        )

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return [
            {
                "train_means": True,
                "weight": "none",
                "target_options": "stdnorm",
                "min_maf": 0.05,
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "conv_nlayers": 2,
                "conv_filters": 1,
                "conv_kernel_size": 3,
                "conv_strides": 2,
                "conv_l1": 1e-50,
                "conv_l2": 1e-50,
                "conv_activation": "linear",
                "conv_use_bn": False,
                "embed_nunits": 10,
                "marker_embed_nlayers": 1,
                "marker_embed0_activation": "linear",
                "marker_embed_adaptive_l1": False,
                "marker_embed0_dropout_rate": 0.5,
                "marker_embed0_l1": 1e-50,
                "marker_embed0_l2": 1e-50,
                "env_embed_nlayers": 1,
                "env_embed0_activation": "linear",
                "env_embed0_dropout_rate": 0.0,
                "env_embed0_l2": 1e-50,
                "combiner": "add",
                "postembed_nlayers": 0,
                "postembed_dropout_rate": 0.2,
                "postembed_l1": 1e-50,
                "postembed_l2": 1e-50,
                "use_rank_loss": False,
                "marker_embed_regularizer": "none",
                "env_embed_regularizer": "none",
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "stdnorm",
                "min_maf": 0.05,
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "conv_nlayers": 2,
                "conv_filters": 1,
                "conv_kernel_size": 3,
                "conv_strides": 2,
                "conv_l1": 1e-50,
                "conv_l2": 1e-50,
                "conv_activation": "linear",
                "conv_use_bn": False,
                "embed_nunits": 10,
                "marker_embed_nlayers": 1,
                "marker_embed0_activation": "linear",
                "marker_embed_adaptive_l1": False,
                "marker_embed0_dropout_rate": 0.5,
                "marker_embed0_l1": 1e-50,
                "marker_embed0_l2": 1e-50,
                "env_embed_nlayers": 1,
                "env_embed0_activation": "linear",
                "env_embed0_dropout_rate": 0.0,
                "env_embed0_l2": 1e-50,
                "combiner": "add",
                "postembed_nlayers": 1,
                "postembed_nunits": 10,
                "postembed_dropout_rate": 0.2,
                "postembed_l1": 1e-50,
                "postembed_l2": 1e-50,
                "use_rank_loss": False,
                "marker_embed_regularizer": "none",
                "env_embed_regularizer": "none",
            },
            {
                "train_means": True,
                "weight": "none",
                "target_options": "stdnorm",
                "min_maf": 0.05,
                "marker_preprocessor": "maf",
                "feature_selector": "passthrough",
                "conv_nlayers": 2,
                "conv_filters": 1,
                "conv_kernel_size": 3,
                "conv_strides": 2,
                "conv_l1": 1e-50,
                "conv_l2": 1e-50,
                "conv_activation": "linear",
                "conv_use_bn": False,
                "embed_nunits": 10,
                "marker_embed_nlayers": 1,
                "marker_embed0_activation": "linear",
                "marker_embed_adaptive_l1": True,
                "marker_embed_adaptive_l1_rate": 1e-3,
                "marker_embed_adaptive_l2_rate": 1e-9,
                "marker_embed0_dropout_rate": 0.5,
                "marker_embed0_l1": 1e-50,
                "marker_embed0_l2": 1e-50,
                "env_embed_nlayers": 1,
                "env_embed0_activation": "linear",
                "env_embed0_dropout_rate": 0.0,
                "env_embed0_l2": 1e-50,
                "combiner": "add",
                "postembed_nlayers": 0,
                "postembed_dropout_rate": 0.2,
                "postembed_l1": 1e-50,
                "postembed_l2": 1e-50,
                "use_rank_loss": False,
                "marker_embed_regularizer": "semihard",
                "marker_semihard_loss": 0.5,
                "env_embed_regularizer": "none",
            },
        ]


class BGLRWrapper(object):

    def __init__(
        self,
        fs_model,
        add_trans,
        marker_model="BRR",
        grouping_model="FIXED",
        interaction_model="RKHS",
        target_trans=None,
        group_trans=None,
        dom_trans=None,
        epi_trans=None,
    ):
        self.fs_model = fs_model
        self.marker_model = marker_model
        self.grouping_model = grouping_model
        self.interaction_model = interaction_model
        self.target_trans = target_trans
        self.group_trans = group_trans
        self.add_trans = add_trans
        self.dom_trans = dom_trans
        self.epi_trans = epi_trans
        return

    def fit(self, X, y, sample_weight=None, individuals=None, **kwargs):
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
        elif len(y_.shape) > 2:
            raise ValueError("We don't currently support multi-target")

        self.markers = X_markers
        self.individuals = individuals
        self.groups = X_grouping
        self.y = y_
        return

    def join_train_test(
        self,
        X: "Tuple[npt.ArrayLike, Optional[npt.ArrayLike]]"
    ) -> "Tuple[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]], np.ndarray]":  # noqa
        assert hasattr(self, "y")

        X_markers_, X_grouping_ = X
        X_markers = np.array(X_markers_)
        n_predict = X_markers.shape[0]
        X_markers = np.concatenate([self.markers, X_markers])

        if self.individuals is not None:
            m = len(self.individuals) + 1
            individuals = np.concatenate([
                self.individuals,
                np.arange(m, m + n_predict)
            ])
        else:
            individuals = None

        if X_grouping_ is not None:
            X_grouping: Optional[np.ndarray] = np.array(X_grouping_)
            X_grouping = np.concatenate([self.groups, X_grouping])
        else:
            assert self.groups is None
            X_grouping = None
        del X_markers_, X_grouping_

        # Artificially create a censored dataset for the test samples.
        y = np.concatenate([
            self.y,
            np.expand_dims(np.full(n_predict, np.nan), -1)
        ])
        return (X_markers, X_grouping, individuals), y

    def run_bglr(  # noqa
        self,
        X: "Tuple[npt.ArrayLike, Optional[npt.ArrayLike], Optional[np.ArrayLike]]",  # noqa
        y: "npt.ArrayLike"
    ) -> "np.ndarray":
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()

        X_markers_, X_grouping_, X_individuals_ = X
        X_markers = np.array(X_markers_)

        if self.fs_model is not None:
            X_markers = self.fs_model.fit_transform(X_markers, y)

        if X_grouping_ is not None:
            X_grouping: Optional[np.ndarray] = np.array(X_grouping_)
        else:
            assert self.groups is None
            X_grouping = None

        if X_individuals_ is None:
            individuals = np.range(X.shape[0])
        else:
            individuals = np.array(X_individuals_)

        del X_markers_, X_grouping_, X_individuals_

        if self.target_trans is not None:
            y_ = self.target_trans.fit_transform(
                y,
            )
        else:
            y_ = y

        if (X_grouping is not None) and (self.group_trans is not None):
            X_grouping = self.group_trans.fit_transform(X_grouping)

        BGLR = importr("BGLR")
        ETA = []

        X_markers = self.add_trans.fit_transform(
            X_markers,
            y=y_,
        )

        if self.marker_model == "RKHS":
            ETA.append(
                ro.ListVector({"K": X_markers, "model": self.marker_model})
            )
        else:
            ETA.append(
                ro.ListVector({"X": X_markers, "model": self.marker_model})
            )

        if self.dom_trans is not None:
            X_dom = self.dom_trans.fit_transform(
                X_markers,
                y=y_,
            )
            ETA.append(
                ro.ListVector({"K": X_dom, "model": "RKHS"})
            )

        if self.epi_trans is not None:
            X_epi = self.epi_trans.fit_transform(
                X_markers,
                y=y_,
            )
            ETA.append(
                ro.ListVector({"K": X_epi, "model": "RKHS"})
            )

        if X_grouping is not None:
            ETA.append(ro.ListVector({
                "X": X_grouping,
                "model": self.grouping_model
            }))

        if (X_grouping is not None) and (self.interaction_model != "none"):
            X_trans = (
                (X_markers - X_markers.mean(axis=0)) / X_markers.std(axis=0)
            )
            X_trans = X_trans.dot(X_trans.T) / X_trans.shape[1]
            ETA.append(ro.ListVector({
                "K": X_grouping.dot(X_grouping.T) * X_trans,
                "model": self.interaction_model
            }))

        results = BGLR.BGLR(
            y=y_,
            ETA=ETA,
            nIter=10000,
            burnIn=1000,
            response_type="continuous",
            verbose=False
        )

        preds = BGLR.predict_BGLR(results)

        if len(preds.shape) == 1:
            preds = np.expand_dims(preds, -1)

        if self.target_trans is not None:
            preds = self.target_trans.inverse_transform(
                preds,
            )

        numpy2ri.deactivate()
        return preds

    def predict_all(self, X: "Tuple[npt.ArrayLike, Optional[npt.ArrayLike]]"):
        (X_markers, X_grouping, individuals), y = self.join_train_test(X)
        preds = self.run_bglr((X_markers, X_grouping, individuals), y)
        return preds

    def predict(self, X: "Tuple[npt.ArrayLike, Optional[npt.ArrayLike]]"):
        X_markers_, X_grouping_ = X
        n_predict = np.array(X_markers_).shape[0]

        (X_markers, X_grouping, individuals), y = self.join_train_test(X)
        preds = self.run_bglr((X_markers, X_grouping, individuals), y)
        return preds[-n_predict:]


class BGLRModel(BGLRBaseModel):

    use_weights: bool = False

    def sample_params(self, trial: "optuna.Trial") -> Dict[str, Any]:
        params = self.sample_preprocessing_params(
            trial,
            target_options=["stdnorm", "quantile"],
            marker_options=["maf", "noia_add"],
            feature_selection_options=["passthrough", "relief"],
            grouping_options=["onehot"],
        )

        if params["marker_preprocessor"] == "noia_add":
            params["marker_model"] = "RKHS"
        else:
            params["marker_model"] = trial.suggest_categorical(
                "marker_model",
                ["BRR", "BayesA", "BayesB", "BayesC", "BL", "RKHS"]
            )

        if len(self.grouping_columns) > 0:
            params.update({
                "grouping_model": trial.suggest_categorical(
                    "grouping_model",
                    ["FIXED", "BRR", "BayesA", "BayesB", "BayesC", "BL"]
                ),
                "interaction_model": trial.suggest_categorical(
                    "interaction_model",
                    ["none", "RHKS"]
                )
            })
        return params

    def model(self, params: Dict[str, Any]):
        (
            target_trans,
            feature_selector,
            gmarkers,
            dmarkers,
            emarkers,
            grouping
        ) = self.sample_preprocessing_model(params)

        return BGLRWrapper(
            fs_model=feature_selector,
            add_trans=gmarkers,
            grouping_model=params["grouping_model"],
            interaction_model=params["interaction_model"],
            target_trans=target_trans,
            group_trans=grouping,
            dom_trans=dmarkers,
            epi_trans=emarkers
        )

    def starting_points(self) -> List[Dict[str, BaseTypes]]:
        return []
