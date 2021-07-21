#!/usr/bin/env python3

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Union, Optional
    from typing import List
    import numpy.typing as npt


class Unity(TransformerMixin, BaseEstimator):
    """A dummy transformer that does nothing.

    This is convenient for optimisation pipelines, but generally won't be used
    interactively.
    """

    requires_y: bool = False

    def __init__(self):
        return

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        return

    def fit(self, X: "npt.ArrayLike", y: "Optional[npt.ArrayLike]" = None) -> "Unity":
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """

        return self

    def transform(self, X: "npt.ArrayLike") -> "npt.ArrayLike":
        """Scale features of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return X

    def inverse_transform(self, X: "npt.ArrayLike") -> "npt.ArrayLike":
        """Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return X


class MarkerMAFScaler(TransformerMixin, BaseEstimator):
    """Scale each marker by the minor allele frequency

    This is the same scaling applied to the Van Raden similarity computation
    before the dot-product is taken. It is somewhat analogous to mean-centering.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.preprocessor import MarkerMAFScaler
    >>> from selectml.data import basic
    >>> X, _, _ = basic()
    >>> X = np.unique(X, axis=0)
    >>> MarkerMAFScaler().fit_transform(X)
    array([[-1.4, -0.4, -0.4,  0.6,  0.2, -0.2, -0.8, -1. ,  0.2, -0.4],
           [-0.4,  1.6,  0.6, -0.4,  0.2, -0.2,  1.2, -1. ,  0.2, -0.4],
           [ 0.6, -0.4, -1.4,  0.6, -0.8,  0.8, -0.8,  1. ,  1.2, -0.4],
           [ 0.6, -0.4,  0.6, -1.4,  1.2,  0.8,  0.2,  1. , -0.8,  0.6],
           [ 0.6, -0.4,  0.6,  0.6, -0.8, -1.2,  0.2,  0. , -0.8,  0.6]])
    """

    requires_y: bool = False

    def __init__(self, ploidy: int = 2, *, copy: bool = True):
        self.ploidy = int(ploidy)
        self.copy = bool(copy)
        return

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'n_samples_seen_'):
            del self.n_samples_seen_
            del self.n_features_in_
            del self.n_features_
            del self.allele_counts_
            del self.P_

    def partial_fit(self, X: "npt.ArrayLike", y: "Optional[npt.ArrayLike]" = None) -> "MarkerMAFScaler":
        """Online computation of min and max on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """

        first_pass: bool = not hasattr(self, 'n_samples_seen_')

        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite="allow-nan",
            estimator=self
        )

        # np.isin(X[~np.isnan(X)], np.arange(self.ploidy + 1)).all()
        all_ok = True

        if not all_ok:
            raise ValueError(
                "Encountered a value less than 0 or greater "
                f"than {self.ploidy}."
            )

        # Maybe raise a warning if no 0s or 2s i.e. maybe smaller ploidy than
        # specified.

        if first_pass:
            self.n_samples_seen_: np.ndarray = (~np.isnan(X_)).sum(axis=0)
            self.allele_counts_: np.ndarray = (self.ploidy - X_).sum(axis=0)
        else:
            assert X_.shape[1] == self.n_features_in_, \
                "Must have same number of features"
            self.n_samples_seen_ += (~np.isnan(X_)).sum(axis=0)
            self.allele_counts_ += (self.ploidy - X_).sum(axis=0)

        # Frequency of alternate allele
        p_i: np.ndarray = self.allele_counts_ / (self.ploidy * self.n_samples_seen_)
        self.P_: np.ndarray = self.ploidy * (p_i - 0.5)

        self.n_features_in_: int = self.P_.shape[0]
        self.n_features_ = self.n_features_in_
        return self

    def fit(self, X: "npt.ArrayLike", y: "Optional[npt.ArrayLike]" = None) -> "MarkerMAFScaler":
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def transform(self, X: "npt.ArrayLike") -> np.ndarray:
        """Scale features of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """

        check_is_fitted(self, 'n_features_in_')

        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite="allow-nan",
            estimator=self,
        )

        if X_.shape[1] != self.n_features_in_:
            raise ValueError("Must have same number of features")

        X_ = X_ - (self.ploidy / 2) + self.P_
        return X_

    def inverse_transform(self, X: "npt.ArrayLike") -> np.ndarray:
        """Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self, 'n_features_in_')

        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite="allow-nan",
            estimator=self,
        )

        if X_.shape[1] != self.n_features_in_:
            raise ValueError("Must have same number of features")

        X_ = X_ + (self.ploidy / 2) + self.P_
        return X_


class PercentileRankTransformer(TransformerMixin, BaseEstimator):
    """Convert continuous variables into a quantised ranks.

    This is useful for transforming response variables.

    Examples:

    To quantise by the 25-, 50- and 75-th percentiles, do

    >>> from selectml.sk.preprocessor import PercentileRankTransformer
    >>> from selectml.data import basic
    >>> _, y, _ = basic()
    >>> PercentileRankTransformer([25, 50, 75]).fit_transform(y.reshape((-1, 1)))
    array([[3.],
           [3.],
           [3.],
           [3.],
           [3.],
           [0.],
           [0.],
           [0.],
           [0.],
           [0.],
           [2.],
           [2.],
           [2.],
           [2.],
           [3.],
           [1.],
           [2.],
           [0.],
           [1.],
           [1.],
           [1.],
           [1.],
           [1.],
           [2.],
           [0.]])
    """

    requires_y: bool = False

    def __init__(
        self,
        boundaries: "Sequence[int]",
        *,
        reverse: "Union[bool, Sequence[bool]]" = False,
        copy: bool = True
    ):
        self.boundaries = boundaries

        if not isinstance(reverse, bool):
            self.reverse: Union[bool, List[bool]] = list(reverse)
        else:
            self.reverse = reverse

        self.copy = copy
        return

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        if hasattr(self, "n_features_in_"):
            del self.n_features_in_
            del self.n_features_
            del self.percentiles_
        return self

    def fit(self, X: "npt.ArrayLike", y: "Optional[npt.ArrayLike]" = None) -> "PercentileRankTransformer":
        self._reset()

        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite="allow-nan",
            estimator=self
        )

        if len(X_.shape) == 1:
            X_ = X_.reshape(-1, 1)

        if isinstance(self.reverse, bool):
            reverse: List[bool] = [self.reverse for _ in range(X_.shape[1])]
        else:
            reverse = self.reverse

        if len(reverse) != X_.shape[1]:
            raise ValueError(
                "Number of reverse specifictions does "
                "not match number of columns."
            )

        boundaries = np.repeat(
            np.array(self.boundaries).reshape(-1, 1),
            X_.shape[1],
            axis=1
        )

        percentiles = np.zeros(boundaries.shape)

        for i, r in enumerate(reverse):
            if r:
                boundaries[:, i] = (100 - boundaries[:, i])
            p = np.percentile(X_[:, i], boundaries[:, i])
            percentiles[:, i] = p

        self.percentiles_ = percentiles
        self.n_features_in_ = X_.shape[1]
        self.n_features_ = self.n_features_in_
        return self

    def transform(self, X: "npt.ArrayLike") -> np.ndarray:

        check_is_fitted(self, 'n_features_in_')

        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite="allow-nan",
            estimator=self,
        )

        if len(X_.shape) == 1:
            X_transformed = True
            X_ = X_.reshape(-1, 1)
        else:
            X_transformed = False

        if isinstance(self.reverse, bool):
            reverse: List[bool] = [self.reverse for _ in range(X_.shape[1])]
        else:
            reverse = self.reverse

        if X_.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} columns but "
                f"received {X_.shape[1]}."
            )

        ranks = np.zeros(X_.shape, dtype=float)

        for i in range(X_.shape[1]):
            if reverse[i]:
                for j, th in enumerate(self.percentiles_[:, i], 1):
                    ranks[X_[:, i] < th, i] = j
            else:
                for j, th in enumerate(self.percentiles_[:, i], 1):
                    ranks[X_[:, i] > th, i] = j

        if X_transformed:
            ranks = ranks.reshape(-1)
        return ranks
