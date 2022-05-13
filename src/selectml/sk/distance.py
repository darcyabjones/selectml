#!/usr/bin/env python3

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    import numpy.typing as npt


class VanRadenSimilarity(TransformerMixin, BaseEstimator):
    """Get the Van Raden similarity matrix.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.distance import VanRadenSimilarity
    >>> from selectml.data import basic
    >>> X, _, _ = basic()
    >>> X = np.unique(X, axis=0)
    >>> VanRadenSimilarity().fit_transform(X)
    array([[ 0.53731343,  0.        ,  0.28358209,  0.07462687,  0.35820896],
           [ 0.        ,  0.28358209, -0.17910448,  0.05970149,  0.19402985],
           [ 0.28358209, -0.17910448,  0.92537313,  0.26865672,  0.40298507],
           [ 0.07462687,  0.05970149,  0.26865672,  0.95522388,  0.49253731],
           [ 0.35820896,  0.19402985,  0.40298507,  0.49253731,  1.        ]])
    """

    requires_y: bool = False

    def __init__(
        self,
        *,
        ploidy: int = 2,
        distance: bool = False,
        scale: bool = True,
        copy: bool = True
    ):
        self.distance = distance
        self.scale = scale
        self.ploidy = ploidy
        self.copy = copy
        return

    def _reset(self):
        if hasattr(self, 'n_samples_seen_'):
            del self.n_samples_seen_
            del self.allele_counts_
            del self.X_
            del self.denom_
            del self.P_
            del self.max_
            del self.min_
            del self.n_features_
            del self.n_features_in_
        return

    def partial_fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "VanRadenSimilarity":
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

        first_pass = not hasattr(self, 'n_samples_seen_')

        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite=True,
            estimator=self
        )

        all_ok = True
        # np.isin(X[~np.isnan(X)], np.arange(self.ploidy + 1)).all()

        if not all_ok:
            raise ValueError(
                "Encountered a value less than 0 or greater "
                f"than {self.ploidy}."
            )

        # Maybe raise a warning if no 0s or 2s i.e. maybe smaller ploidy than
        # specified.

        if first_pass:
            self.n_samples_seen_ = (~np.isnan(X)).sum(axis=0)
            self.allele_counts_ = (self.ploidy - X_).sum(axis=0)
            self.X_ = X_
            self.max_ = float("-inf")
            self.min_ = float("inf")
        else:
            assert X_.shape[1] == self.X_.shape[1], \
                "Must have same number of features"

            self.n_samples_seen_ += (~np.isnan(X_)).sum(axis=0)
            self.allele_counts_ += (self.ploidy - X_).sum(axis=0)
            self.X_ = np.concatenate([self.X_, X_])

        self.n_features_in_ = self.X_.shape[1]
        self.n_features_ = self.X_.shape[0]

        # Frequency of alternate allele
        p_i = self.allele_counts_ / (self.ploidy * self.n_samples_seen_)

        # Adding eps is just to avoid zero division
        self.denom_ = (
            (self.ploidy * np.sum(p_i * (1 - p_i))) + np.finfo(float).eps
        )
        self.P_ = self.ploidy * (p_i - 0.5)

        # This is just so we can get the unscaled max
        scale = self.scale
        self.scale = False
        results = self.transform(X_)
        self.scale = scale

        self.max_ = max([self.max_, np.max(np.abs(results))])
        self.min_ = min([self.min_, np.min(np.abs(results))])
        return self

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "VanRadenSimilarity":
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

        p = self.partial_fit(X, y)
        return p

    def transform(
        self,
        X: "npt.ArrayLike"
    ) -> np.ndarray:
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
            force_all_finite=True,
            estimator=self,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError("Must have same number of features")

        Xtrain = self.X_ - ((self.ploidy / 2) + self.P_)
        X_ = X_ - ((self.ploidy / 2) + self.P_)

        dists = X_.dot(Xtrain.T) / self.denom_

        # Min max scale
        if self.scale:
            diff = self.max_ - self.min_ + np.finfo(float).eps

            dists -= self.min_
            dists /= diff

        if self.distance:
            return -1 * dists
        else:
            return dists


class ManhattanDistance(TransformerMixin, BaseEstimator):
    """Get the Manhattan distance matrix.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.distance import ManhattanDistance
    >>> from selectml.data import basic
    >>> X, _, _ = basic()
    >>> X = np.unique(X, axis=0)
    >>> ManhattanDistance().fit_transform(X)
    array([[ 0.,  7.,  8., 12.,  9.],
           [ 7.,  0., 13., 11., 10.],
           [ 8., 13.,  0., 10.,  9.],
           [12., 11., 10.,  0.,  7.],
           [ 9., 10.,  9.,  7.,  0.]])
    """

    requires_y: bool = False

    def __init__(self, *, similarity: bool = False):
        self.similarity = similarity
        return

    def partial_fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "ManhattanDistance":
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

        first_pass = not hasattr(self, 'X_')

        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite=True,
            estimator=self,
        )

        if first_pass:
            self.X_ = X_
        else:
            assert X_.shape[1] == self.X_.shape[1], \
                "Must have same number of features"
            self.X_ = np.concatenate([self.X_, X_])

        self.n_features_in_ = self.X_.shape[1]
        self.n_features_ = self.X_.shape[0]
        return self

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "ManhattanDistance":
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

    def _reset(self):
        if hasattr(self, 'n_features_in_'):
            del self.n_features_in_
            del self.n_features_
            del self.X_
        return

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

        X_ = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite=True,
            estimator=self,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError("Must have same number of features")

        dist = np.apply_along_axis(
            lambda x: np.abs(self.X_ - x).sum(axis=1),
            1,
            X_
        )

        if self.similarity:
            return -1 * dist
        else:
            return dist


class EuclideanDistance(TransformerMixin, BaseEstimator):
    """ Get the Euclidean distance matrix.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.distance import EuclideanDistance
    >>> from selectml.data import basic
    >>> X, _, _ = basic()
    >>> X = np.unique(X, axis=0)
    >>> EuclideanDistance().fit_transform(X)
    array([[ 0., 11., 12., 18., 11.],
           [11.,  0., 21., 15., 12.],
           [12., 21.,  0., 18., 15.],
           [18., 15., 18.,  0., 13.],
           [11., 12., 15., 13.,  0.]])
    """

    requires_y: bool = False

    def __init__(self, *, similarity: bool = False):
        self.similarity = similarity
        return

    def partial_fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "EuclideanDistance":
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

        first_pass = not hasattr(self, 'X_')
        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite=True,
            estimator=self,
        )

        if first_pass:
            self.X_ = X_
        else:
            assert X_.shape[1] == self.X_.shape[1], \
                "Must have same number of features"
            self.X_ = np.concatenate([self.X_, X_])

        self.n_features_in_ = self.X_.shape[1]
        self.n_features_ = self.X_.shape[0]
        return self

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "EuclideanDistance":
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

    def _reset(self):
        if hasattr(self, 'n_features_in_'):
            del self.n_features_in_
            del self.n_features_
            del self.X_
        return

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
            force_all_finite=True,
            estimator=self,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError("Must have same number of features")

        dist = np.apply_along_axis(
            lambda x: ((self.X_ - x) ** 2).sum(axis=1),
            1,
            X_
        )

        if self.similarity:
            return -1 * dist
        else:
            return dist


class HammingDistance(TransformerMixin, BaseEstimator):
    """Get the Hamming distance matrix.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.distance import HammingDistance
    >>> from selectml.data import basic
    >>> X, _, _ = basic()
    >>> X = np.unique(X, axis=0)
    >>> HammingDistance().fit_transform(X)
    array([[0., 5., 6., 9., 8.],
           [5., 0., 9., 9., 9.],
           [6., 9., 0., 6., 6.],
           [9., 9., 6., 0., 4.],
           [8., 9., 6., 4., 0.]])
    """

    requires_y: bool = False

    def __init__(self, *, similarity: bool = False):
        self.similarity = similarity
        return

    def partial_fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "HammingDistance":
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

        first_pass = not hasattr(self, 'X_')
        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite=True,
            estimator=self,
        )

        if first_pass:
            self.X_ = X_
        else:
            assert X_.shape[1] == self.X_.shape[1], \
                "Must have same number of features"
            self.X_ = np.concatenate([self.X_, X_])

        self.n_features_in_ = self.X_.shape[1]
        self.n_features_ = self.X_.shape[0]
        return self

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "HammingDistance":
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

    def _reset(self):
        if hasattr(self, 'n_features_in_'):
            del self.n_features_in_
            del self.n_features_
            del self.X_
        return

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
            force_all_finite=True,
            estimator=self,
        )

        if X_.shape[1] != self.n_features_in_:
            raise ValueError("Must have same number of features")

        dist = np.apply_along_axis(
            lambda x: (self.X_ != x).sum(axis=1),
            1,
            X_
        )

        dist = dist.astype(X_.dtype)

        if self.similarity:
            return -1 * dist
        else:
            return dist


class NOIAAdditiveKernel(TransformerMixin, BaseEstimator):
    """Get the NOIA Additive distance kernel.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.distance import NOIAAdditiveKernel
    >>> from selectml.data import basic
    >>> X, _, _ = basic()
    >>> X = np.unique(X, axis=0)
    >>> NOIAAdditiveKernel().fit_transform(X)
    array([[ 0.78082192, -0.04109589, -0.00684932, -0.55479452, -0.17808219],
           [-0.04109589,  1.02054795, -0.65753425, -0.17808219, -0.14383562],
           [-0.00684932, -0.65753425,  1.26027397, -0.31506849, -0.28082192],
           [-0.55479452, -0.17808219, -0.31506849,  1.19178082, -0.14383562],
           [-0.17808219, -0.14383562, -0.28082192, -0.14383562,  0.74657534]])
    """

    requires_y: bool = False

    def __init__(
        self,
        AA: float = 2.,
        Aa: float = 1.,
        aa: float = 0.,
        copy: bool = True
    ):
        from .preprocessor import NOIAAdditiveScaler
        self.scaler = NOIAAdditiveScaler(AA=AA, Aa=Aa, aa=aa)
        self.copy = copy
        return

    def partial_fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "NOIAAdditiveKernel":
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

        first_pass = not hasattr(self, 'X_')

        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite=True,
            estimator=self,
        )

        if first_pass:
            self.X_ = X_
        else:
            assert X_.shape[1] == self.X_.shape[1], \
                "Must have same number of features"
            self.X_ = np.concatenate([self.X_, X_])

        self.scaler.partial_fit(X_)

        scaled = self.scaler.transform(self.X_)
        mat = scaled.dot(scaled.T)
        self.trace = mat.trace()

        self.n_features_in_ = self.X_.shape[1]
        self.n_features_ = self.X_.shape[0]
        return self

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "NOIAAdditiveKernel":
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

    def _reset(self):
        if hasattr(self, 'n_features_in_'):
            del self.n_features_in_
            del self.n_features_
            del self.X_
            del self.trace_
            self.scaler._reset()
        return

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

        X_ = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite=True,
            estimator=self,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError("Must have same number of features")

        X_trans = self.scaler.transform(X_)
        train_trans = self.scaler.transform(self.X_)
        dotprod = X_trans.dot(train_trans.T)

        return dotprod / (self.trace / self.X_.shape[0])


class NOIADominanceKernel(TransformerMixin, BaseEstimator):
    """Get the NOIA Dominance similarity kernel.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.distance import NOIADominanceKernel
    >>> from selectml.data import basic
    >>> X, _, _ = basic()
    >>> X = np.unique(X, axis=0)
    >>> NOIADominanceKernel().fit_transform(X)
    array([[ 1.19398514,  0.46848641, -0.49215089, -0.50930513, -0.66101553],
           [ 0.46848641,  1.6314182 , -0.38734268, -0.85218068, -0.86038124],
           [-0.49215089, -0.38734268,  0.51388238,  0.2586608 ,  0.10695039],
           [-0.50930513, -0.85218068,  0.2586608 ,  0.67454646,  0.42827855],
           [-0.66101553, -0.86038124,  0.10695039,  0.42827855,  0.98616783]])
    """

    requires_y: bool = False

    def __init__(
        self,
        AA: float = 2,
        Aa: float = 1,
        aa: float = 0,
        copy: bool = True
    ):
        from .preprocessor import NOIADominanceScaler
        self.scaler = NOIADominanceScaler(AA=AA, Aa=Aa, aa=aa)
        self.copy = copy
        return

    def partial_fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "NOIADominanceKernel":
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

        first_pass = not hasattr(self, 'X_')

        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite=True,
            estimator=self,
        )

        if first_pass:
            self.X_ = X_
        else:
            assert X_.shape[1] == self.X_.shape[1], \
                "Must have same number of features"
            self.X_ = np.concatenate([self.X_, X_])

        self.scaler.partial_fit(X_)

        scaled = self.scaler.transform(self.X_)
        mat = scaled.dot(scaled.T)
        self.trace = mat.trace()

        self.n_features_in_ = self.X_.shape[1]
        self.n_features_ = self.X_.shape[0]
        return self

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "NOIADominanceKernel":
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

    def _reset(self):
        if hasattr(self, 'n_features_in_'):
            del self.n_features_in_
            del self.n_features_
            del self.X_
            del self.trace_
            self.scaler._reset()
        return

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

        X_ = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite=True,
            estimator=self,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError("Must have same number of features")

        X_trans = self.scaler.transform(X_)
        train_trans = self.scaler.transform(self.X_)
        dotprod = X_trans.dot(train_trans.T)

        return dotprod / (self.trace / self.X_.shape[0])


class HadamardCovariance(TransformerMixin, BaseEstimator):
    """Get the Hadamard product of two kernels.

    Note that the two kernels must return square matrices for this to work.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.distance import (
    ...   HadamardCovariance,
    ...   NOIAAdditiveKernel
    ... )
    >>> from selectml.data import basic
    >>> X, _, _ = basic()
    >>> X = np.unique(X, axis=0)
    >>> k = NOIAAdditiveKernel()
    >>> HadamardCovariance(a=k, b=k, fit_b=False).fit_transform(X)
    array([[5.84299e-01, 1.61855e-03, 4.49599e-05, 2.94982e-01, 3.03929e-02],
           [1.61855e-03, 9.98156e-01, 4.14351e-01, 3.03929e-02, 1.98273e-02],
           [4.49599e-05, 4.14351e-01, 1.52216e+00, 9.51353e-02, 7.55777e-02],
           [2.94982e-01, 3.03929e-02, 9.51353e-02, 1.36120e+00, 1.98273e-02],
           [3.03929e-02, 1.98273e-02, 7.55777e-02, 1.98273e-02, 5.34169e-01]])
    """

    requires_y: bool = False

    def __init__(
        self,
        a,
        b,
        copy: bool = True,
        fit_a: bool = True,
        fit_b: bool = True,
    ):
        self.copy = copy
        self.a = a
        self.b = b
        self.fit_a = fit_a
        self.fit_b = fit_b
        return

    def partial_fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "HadamardCovariance":
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

        first_pass = not hasattr(self, 'X_')

        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite=True,
            estimator=self,
        )

        if first_pass:
            self.X_ = X_
        else:
            assert X_.shape[1] == self.X_.shape[1], \
                "Must have same number of features"
            self.X_ = np.concatenate([self.X_, X_])

        if self.fit_a:
            self.a.partial_fit(X_)

        if self.fit_b:
            self.b.partial_fit(X_)

        A = self.a.transform(self.X_)
        B = self.b.transform(self.X_)

        AB = A * B
        self.trace = AB.trace()

        self.n_features_in_ = self.X_.shape[1]
        self.n_features_ = self.X_.shape[0]
        return self

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "HadamardCovariance":
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

    def _reset(self):
        if hasattr(self, 'n_features_in_'):
            del self.n_features_in_
            del self.n_features_
            del self.X_
            del self.trace_
            self.a._reset()
            self.b._reset()
        return

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

        X_ = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite=True,
            estimator=self,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError("Must have same number of features")

        A = self.a.transform(self.X_)
        B = self.b.transform(self.X_)

        AB = A * B
        return AB / (self.trace / self.X_.shape[0])


class SquaredEuclideanDistance(TransformerMixin, BaseEstimator):
    """ Get the Euclidean distance matrix.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.distance import SquaredEuclideanDistance
    >>> from selectml.data import basic
    >>> X, _, _ = basic()
    >>> X = np.unique(X, axis=0)
    >>> SquaredEuclideanDistance().fit_transform(X)
    array([[1.    , 0.3328, 0.3011, 0.1652, 0.3328],
           [0.3328, 1.    , 0.1224, 0.2231, 0.3011],
           [0.3011, 0.1224, 1.    , 0.1652, 0.2231],
           [0.1652, 0.2231, 0.1652, 1.    , 0.2725],
           [0.3328, 0.3011, 0.2231, 0.2725, 1.    ]])
    """

    requires_y: bool = False

    def __init__(self, *, bandwidth: float = 1):
        self.bandwidth = bandwidth
        return

    def partial_fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "SquaredEuclideanDistance":
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

        first_pass = not hasattr(self, 'X_')
        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite=True,
            estimator=self,
        )

        if first_pass:
            self.X_ = X_
        else:
            assert X_.shape[1] == self.X_.shape[1], \
                "Must have same number of features"
            self.X_ = np.concatenate([self.X_, X_])

        self.n_features_in_ = self.X_.shape[1]
        self.n_features_ = self.X_.shape[0]
        return self

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "EuclideanDistance":
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

    def _reset(self):
        if hasattr(self, 'n_features_in_'):
            del self.n_features_in_
            del self.n_features_
            del self.X_
        return

    def transform(
        self,
        X: "npt.ArrayLike",
    ) -> np.ndarray:
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
            force_all_finite=True,
            estimator=self,
        )
        if X_.shape[1] != self.n_features_in_:
            raise ValueError("Must have same number of features")

        dist = np.apply_along_axis(
            lambda x: ((self.X_ - x) ** 2).sum(axis=1),
            1,
            X_
        )

        return np.exp((-self.bandwidth * dist) / X_.shape[1])
