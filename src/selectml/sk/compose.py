#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator  # TransformerMixin

from sklearn.utils.metaestimators import if_delegate_has_method

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union
    from typing import Tuple
    import numpy.typing as npt


class Aggregated(BaseEstimator):
    """Reduce measurements by a grouping factor before applying the
    estimator or transformer.

    This is useful if you want to train or test on means, or if you only want
    part of your preprocessing pipeline to be trained on the means.

    Examples:

    To do MAF scaling but not have differences in replicate numbers bias that
    scaler.

    >>> from selectml.sk.compose import Aggregated
    >>> from selectml.sk.preprocessor import MAFScaler
    >>> from selectml.data import basic
    >>> x, y, indivs = basic()
    >>> (Aggregated(MAFScaler(), agg_train=True, agg_test=True)
    ...  .fit_transform(x, y, indivs))
    array([[-0.4,  1.6,  0.6, -0.4,  0.2, -0.2,  1.2, -1. ,  0.2, -0.4],
           [ 0.6, -0.4, -1.4,  0.6, -0.8,  0.8, -0.8,  1. ,  1.2, -0.4],
           [ 0.6, -0.4,  0.6, -1.4,  1.2,  0.8,  0.2,  1. , -0.8,  0.6],
           [-1.4, -0.4, -0.4,  0.6,  0.2, -0.2, -0.8, -1. ,  0.2, -0.4],
           [ 0.6, -0.4,  0.6,  0.6, -0.8, -1.2,  0.2,  0. , -0.8,  0.6]])
    >>> (Aggregated(MAFScaler(), agg_train=False, agg_test=True)
    ...  .fit_transform(x, y, indivs))
    array([[-0.4,  1.6,  0.6, -0.4,  0.2, -0.2,  1.2, -1. ,  0.2, -0.4],
           [ 0.6, -0.4, -1.4,  0.6, -0.8,  0.8, -0.8,  1. ,  1.2, -0.4],
           [ 0.6, -0.4,  0.6, -1.4,  1.2,  0.8,  0.2,  1. , -0.8,  0.6],
           [-1.4, -0.4, -0.4,  0.6,  0.2, -0.2, -0.8, -1. ,  0.2, -0.4],
           [ 0.6, -0.4,  0.6,  0.6, -0.8, -1.2,  0.2,  0. , -0.8,  0.6]])

    In this case the two are the same, because each individual has the same
    number of replicates, but the MarkerMAFScaler is trained on two different
    datasets, one is aggregated one isn't.

    Setting agg_test to False runs the raw dataset through transform and
    predict. Use in combination with agg_train to get different behaviour
    during train and test or between different steps of the pipeline.
    >>> (Aggregated(MAFScaler(), agg_train=True, agg_test=False)
    ...  .fit_transform(x, y, indivs))
    array([[-0.4,  1.6,  0.6, -0.4,  0.2, -0.2,  1.2, -1. ,  0.2, -0.4],
           [-0.4,  1.6,  0.6, -0.4,  0.2, -0.2,  1.2, -1. ,  0.2, -0.4],
           [-0.4,  1.6,  0.6, -0.4,  0.2, -0.2,  1.2, -1. ,  0.2, -0.4],
           [-0.4,  1.6,  0.6, -0.4,  0.2, -0.2,  1.2, -1. ,  0.2, -0.4],
           [-0.4,  1.6,  0.6, -0.4,  0.2, -0.2,  1.2, -1. ,  0.2, -0.4],
           [ 0.6, -0.4, -1.4,  0.6, -0.8,  0.8, -0.8,  1. ,  1.2, -0.4],
           [ 0.6, -0.4, -1.4,  0.6, -0.8,  0.8, -0.8,  1. ,  1.2, -0.4],
           [ 0.6, -0.4, -1.4,  0.6, -0.8,  0.8, -0.8,  1. ,  1.2, -0.4],
           [ 0.6, -0.4, -1.4,  0.6, -0.8,  0.8, -0.8,  1. ,  1.2, -0.4],
           [ 0.6, -0.4, -1.4,  0.6, -0.8,  0.8, -0.8,  1. ,  1.2, -0.4],
           [ 0.6, -0.4,  0.6, -1.4,  1.2,  0.8,  0.2,  1. , -0.8,  0.6],
           [ 0.6, -0.4,  0.6, -1.4,  1.2,  0.8,  0.2,  1. , -0.8,  0.6],
           [ 0.6, -0.4,  0.6, -1.4,  1.2,  0.8,  0.2,  1. , -0.8,  0.6],
           [ 0.6, -0.4,  0.6, -1.4,  1.2,  0.8,  0.2,  1. , -0.8,  0.6],
           [ 0.6, -0.4,  0.6, -1.4,  1.2,  0.8,  0.2,  1. , -0.8,  0.6],
           [-1.4, -0.4, -0.4,  0.6,  0.2, -0.2, -0.8, -1. ,  0.2, -0.4],
           [-1.4, -0.4, -0.4,  0.6,  0.2, -0.2, -0.8, -1. ,  0.2, -0.4],
           [-1.4, -0.4, -0.4,  0.6,  0.2, -0.2, -0.8, -1. ,  0.2, -0.4],
           [-1.4, -0.4, -0.4,  0.6,  0.2, -0.2, -0.8, -1. ,  0.2, -0.4],
           [-1.4, -0.4, -0.4,  0.6,  0.2, -0.2, -0.8, -1. ,  0.2, -0.4],
           [ 0.6, -0.4,  0.6,  0.6, -0.8, -1.2,  0.2,  0. , -0.8,  0.6],
           [ 0.6, -0.4,  0.6,  0.6, -0.8, -1.2,  0.2,  0. , -0.8,  0.6],
           [ 0.6, -0.4,  0.6,  0.6, -0.8, -1.2,  0.2,  0. , -0.8,  0.6],
           [ 0.6, -0.4,  0.6,  0.6, -0.8, -1.2,  0.2,  0. , -0.8,  0.6],
           [ 0.6, -0.4,  0.6,  0.6, -0.8, -1.2,  0.2,  0. , -0.8,  0.6]])
    """

    def __init__(
        self,
        inner: "Union[str, BaseEstimator]",
        *,
        agg_train: bool = True,
        agg_test: bool = True,
        xaggregator: str = "mode",
        yaggregator: str = "mean",
        copy: bool = True,
        **kwargs
    ):
        self.agg_train = agg_train
        self.agg_test = agg_test
        self.xaggregator = xaggregator
        self.yaggregator = yaggregator

        if inner == "passthrough":
            from selectml.sk.preprocessor import Unity
            self.inner = Unity()
        elif isinstance(inner, str):
            raise ValueError("inner must be either 'passthrough' or an object")
        else:
            self.inner = inner

        self.copy = copy

        if hasattr(inner, "set_params"):
            self.inner.set_params(**kwargs)
        return

    def _reset(self):
        if hasattr(self.inner, "_reset"):
            self.inner._reset()
        return

    def _aggregate(
        self,
        X: "npt.ArrayLike",
        y: "Union[None, npt.ArrayLike]",
        individuals: "Union[None, npt.ArrayLike]"
    ) -> "Tuple[np.ndarray, Union[None, np.ndarray]]":
        if individuals is None:
            raise ValueError(
                "Aggregated must be provided with individuals")

        if self.xaggregator not in ("mean", "median", "min", "max", "mode",
                                    "first", "last", "random"):
            raise ValueError("Invalid aggregation option selected.")

        if self.yaggregator not in ("mean", "median", "min", "max", "mode",
                                    "first", "random"):
            raise ValueError("Invalid aggregation option selected.")

        indivs = np.unique(np.array(individuals))

        if self.xaggregator in ("mean", "median", "min",
                                "max", "first", "last"):
            X_trans = getattr(
                pd.DataFrame(X).groupby(individuals),
                self.xaggregator
            )().loc[indivs, :].values
        elif self.xaggregator == "mode":
            X_trans = (
                pd.DataFrame(X)
                .groupby(individuals)
                .apply(lambda d: d.mode(axis="rows").head(1))
                .loc[indivs, :].values
            )
        elif self.xaggregator == "random":
            from numpy.random import default_rng
            rng = default_rng()
            X_trans = (
                pd.DataFrame(X).groupby(individuals)
                .apply(lambda x: (
                    x.apply(
                        lambda xi: rng.choice(xi, 1, replace=False)[0]
                    )
                )).loc[indivs, :].values
            )

        if y is None:
            y_trans: "Union[None, np.ndarray]" = None
        elif self.yaggregator in ("mean", "median", "min",
                                  "max", "first", "last"):
            y_trans = getattr(
                pd.DataFrame(y).groupby(individuals),
                self.yaggregator
            )().loc[indivs].values
        elif self.yaggregator == "mode":
            y_trans = (
                pd.DataFrame(y)
                .groupby(individuals)
                .apply(lambda d: d.mode(axis="columns", dropna=True).head(1))
                .loc[indivs].values
            )
        elif self.yaggregator == "random":
            from numpy.random import default_rng
            rng = default_rng()
            y_trans = (
                pd.DataFrame(y).groupby(individuals)
                .apply(lambda x: (
                    x.apply(
                        lambda xi: rng.choice(xi, 1, replace=False)[0]
                    )
                )).loc[indivs].values
            )
        return X_trans, y_trans

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Union[None, npt.ArrayLike]" = None,
        individuals: "Union[None, npt.ArrayLike]" = None,
        **fit_params
    ) -> "Aggregated":
        self._reset()
        if self.inner in ("passthrough", "drop"):
            return self

        if y is None:
            y_: "Union[None, np.ndarray]" = None
        else:
            y_ = np.array(y)

        if individuals is None:
            raise ValueError("Aggregated must be provided with individuals")
        else:
            individuals_: np.ndarray = np.array(individuals)

        X_ = np.array(X)
        if self.agg_train:
            X_, y_ = self._aggregate(X_, y_, individuals_)
        self.inner.fit(X_, y_, **fit_params)

        return self

    @if_delegate_has_method("inner")
    def transform(
        self,
        X: "npt.ArrayLike",
        individuals: "Union[None, npt.ArrayLike]" = None,
        agg: "Union[None, bool]" = None
    ) -> np.ndarray:

        agg = agg if isinstance(agg, bool) else self.agg_test
        if agg:
            assert individuals is not None, (
                "Individuals must be provided if aggregating")
            X_, _ = self._aggregate(X, None, np.array(individuals))
        else:
            X_ = np.array(X)

        if self.inner == "passthrough":
            return X_

        return self.inner.transform(X_)

    @if_delegate_has_method("inner")
    def fit_transform(
        self,
        X: "npt.ArrayLike",
        y: "Union[None, npt.ArrayLike]" = None,
        individuals: "Union[None, npt.ArrayLike]" = None,
        agg: "Union[None, bool]" = None,
        **fit_params
    ) -> np.ndarray:
        try:
            return self.fit(
                X,
                y,
                individuals=individuals,
                **fit_params
            ).transform(
                X,
                individuals=individuals,
                agg=agg
            )
        except Exception as e:
            print(individuals)
            raise e

    @if_delegate_has_method("inner")
    def predict(
        self,
        X: "npt.ArrayLike",
        individuals: "Union[None, npt.ArrayLike]" = None,
        agg: "Union[None, bool]" = None
    ) -> np.ndarray:
        agg = agg if isinstance(agg, bool) else self.agg_test
        if agg:
            X, y = self._aggregate(X, None, individuals)
        return self.inner.predict(X)

    @if_delegate_has_method("inner")
    def fit_predict(
        self,
        X: "npt.ArrayLike",
        y: "Union[None, npt.ArrayLike]",
        individuals: "Union[None, npt.ArrayLike]",
        agg: "Union[None, bool]" = None,
        **fit_params
    ) -> np.ndarray:
        return (
            self
            .fit(X, y, individuals=individuals, **fit_params)
            .predict(X, individuals=individuals, agg=agg)
        )

    @if_delegate_has_method("inner")
    def predict_proba(
        self,
        X: "npt.ArrayLike",
        individuals: "Union[None, npt.ArrayLike]" = None,
        agg: "Union[None, bool]" = None
    ) -> np.ndarray:
        agg = agg if isinstance(agg, bool) else self.agg_test
        if agg:
            X, y = self._aggregate(X, None, individuals)
        return self.inner.predict_proba(X)

    @if_delegate_has_method("inner")
    def decision_function(
        self,
        X: "npt.ArrayLike",
        individuals: "Union[None, npt.ArrayLike]" = None,
        agg: "Union[None, bool]" = None
    ):
        agg = agg if isinstance(agg, bool) else self.agg_test
        if agg:
            X, y = self._aggregate(X, None, individuals)
        return self.inner.decision_function(X)

    @if_delegate_has_method("inner")
    def score_samples(
        self,
        X: "npt.ArrayLike",
        individuals: "Union[None, npt.ArrayLike]" = None,
        agg: "Union[None, bool]" = None
    ):
        agg = agg if isinstance(agg, bool) else self.agg_test
        if agg:
            X, y = self._aggregate(X, None, individuals)
        return self.inner.score_samples(X)

    @if_delegate_has_method("inner")
    def predict_log_proba(
        self,
        X: "npt.ArrayLike",
        individuals: "Union[None, npt.ArrayLike]" = None,
        agg: "Union[None, bool]" = None
    ) -> np.ndarray:
        agg = agg if isinstance(agg, bool) else self.agg_test
        if agg:
            X, y = self._aggregate(X, None, individuals)
        return self.inner.score_samples(X)

    @if_delegate_has_method("inner")
    def inverse_transform(
        self,
        X: "npt.ArrayLike",
        individuals: "Union[None, npt.ArrayLike]" = None,
        agg: "Union[None, bool]" = None
    ) -> np.ndarray:
        agg = agg if isinstance(agg, bool) else self.agg_test
        if agg:
            X_, _ = self._aggregate(X, None, individuals)
        else:
            X_ = np.array(X)
        return self.inner.inverse_transform(X_)

    @if_delegate_has_method("inner")
    def score(
        self,
        X: "npt.ArrayLike",
        y: "Union[None, npt.ArrayLike]" = None,
        sample_weight: "Union[None, npt.ArrayLike]" = None,
        individuals: "Union[None, npt.ArrayLike]" = None,
        agg: "Union[None, bool]" = None
    ):
        agg = agg if isinstance(agg, bool) else self.agg_test
        if agg:
            X, y = self._aggregate(X, y, individuals)
        return self.inner.score(X, y, sample_weight=sample_weight)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if k in ("xaggregator", "yaggregator",
                     "inner", "agg_test", "agg_train"):
                setattr(self, k, v)
            else:
                setattr(self.inner, k, v)
        return

    def get_params(self, deep=True):
        params = {
            k: getattr(self, k)
            for k
            in ("xaggregator", "yaggregator", "inner", "agg_test", "agg_train")
        }

        if hasattr(self.inner, "get_params"):
            params.update(self.inner.get_params(deep=deep))
        return params

    @property
    def classes_(self):
        return self.inner.classes_

    @property
    def n_features_in_(self):
        return self.inner.n_features_in_

    @property
    def n_features_(self):
        return self.inner.n_features_
