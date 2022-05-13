#!/usr/bin/env python3

import pandas as pd
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
    from typing import Type
    import numpy.typing as npt


class Make2D(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        *,
        validate=False,
        accept_sparse=False,
    ):
        self.validate = validate
        self.accept_sparse = accept_sparse

    def _check_input(self, X, *, reset):
        if self.validate:
            return self._validate_data(
                X,
                accept_sparse=self.accept_sparse,
                reset=reset
            )
        return X

    def _make_2d(self, x):
        if len(x.shape) == 1:
            return x.reshape(-1, 1)
        if len(x.shape) == 2:
            return x
        else:
            raise ValueError("Cannot have more than a 2D or empty array.")

    def _make_1d(self, x):
        if len(x.shape) == 2:
            if x.shape[1] > 1:
                return x
            elif x.shape[1] == 1:
                return x.reshape(-1)
            else:
                raise ValueError("Cannot have a zero-dimensioned feature.")
        elif len(x.shape) == 1:
            return x
        else:
            raise ValueError("Cannot have more than a 2D or empty array.")

    def fit(self, X, y=None):
        """Fit transformer by checking X.
        If ``validate`` is ``True``, ``X`` will be checked.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        self : object
            FunctionTransformer class instance.
        """
        X = self._check_input(X, reset=True)
        return self

    def transform(self, X):
        """Transform X using the forward function.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.
        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        X = self._check_input(X, reset=False)
        return self._make_2d(X)

    def inverse_transform(self, X):
        """Transform X using the inverse function.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.
        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        if self.validate:
            X = check_array(X, accept_sparse=self.accept_sparse)
        return self._make_1d(X)

    def __sklearn_is_fitted__(self):
        """Return True since FunctionTransfomer is stateless."""
        return True

    def _more_tags(self):
        return {"no_validation": not self.validate, "stateless": True}


class Make1D(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        *,
        validate=False,
        accept_sparse=False,
    ):
        self.validate = validate
        self.accept_sparse = accept_sparse

    def _check_input(self, X, *, reset):
        if self.validate:
            return self._validate_data(
                X,
                accept_sparse=self.accept_sparse,
                reset=reset
            )
        return X

    def _make_2d(self, x):
        if len(x.shape) == 1:
            return x.reshape(-1, 1)
        if len(x.shape) == 2:
            return x
        else:
            raise ValueError("Cannot have more than a 2D or empty array.")

    def _make_1d(self, x):
        if len(x.shape) == 2:
            if x.shape[1] > 1:
                return x
            elif x.shape[1] == 1:
                return x.reshape(-1)
            else:
                raise ValueError("Cannot have a zero-dimensioned feature.")
        elif len(x.shape) == 1:
            return x
        else:
            raise ValueError("Cannot have more than a 2D or empty array.")

    def fit(self, X, y=None):
        """Fit transformer by checking X.
        If ``validate`` is ``True``, ``X`` will be checked.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        self : object
            FunctionTransformer class instance.
        """
        X = self._check_input(X, reset=True)
        return self

    def transform(self, X):
        """Transform X using the forward function.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.
        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        X = self._check_input(X, reset=False)
        return self._make_1d(X)

    def inverse_transform(self, X):
        """Transform X using the inverse function.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.
        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        if self.validate:
            X = check_array(X, accept_sparse=self.accept_sparse)
        return self._make_2d(X)

    def __sklearn_is_fitted__(self):
        """Return True since FunctionTransfomer is stateless."""
        return True

    def _more_tags(self):
        return {"no_validation": not self.validate, "stateless": True}


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

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None,
        **kwargs,
    ) -> "Unity":
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

    def transform(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None,
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
        return np.array(X)

    def inverse_transform(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None,
    ) -> np.ndarray:
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
        return np.array(X)


class MAFScaler(TransformerMixin, BaseEstimator):
    """Scale each marker by the minor allele frequency

    This is the same scaling applied to the Van Raden similarity computation
    before the dot-product is taken.
    It is somewhat analogous to mean-centering.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.preprocessor import MAFScaler
    >>> from selectml.data import basic
    >>> X, _, _ = basic()
    >>> X = np.unique(X, axis=0)
    >>> MAFScaler().fit_transform(X)
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

    def partial_fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "MAFScaler":
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
        p_i = self.allele_counts_ / (self.ploidy * self.n_samples_seen_)
        self.P_: np.ndarray = self.ploidy * (p_i - 0.5)

        self.n_features_in_: int = self.P_.shape[0]
        self.n_features_ = self.n_features_in_
        return self

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "MAFScaler":
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

    def transform(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None,
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
    >>> PercentileRankTransformer([25, 50, 75])\
         .fit_transform(y.reshape((-1, 1)))
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

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "PercentileRankTransformer":
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


class NOIAAdditiveScaler(TransformerMixin, BaseEstimator):
    """Scale each marker by the minor allele frequency

    This is the same scaling applied to the Van Raden similarity computation
    before the dot-product is taken.
    It is somewhat analogous to mean-centering.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.preprocessor import NOIAAdditiveScaler
    >>> from selectml.data import basic
    >>> X, _, _ = basic()
    >>> X = np.unique(X, axis=0)
    >>> sc = NOIAAdditiveScaler()
    >>> trans = sc.fit_transform(X)
    >>> trans
    array([[-1.4, -0.4, -0.4,  0.6,  0.2, -0.2, -0.8, -1. ,  0.2, -0.4],
           [-0.4,  1.6,  0.6, -0.4,  0.2, -0.2,  1.2, -1. ,  0.2, -0.4],
           [ 0.6, -0.4, -1.4,  0.6, -0.8,  0.8, -0.8,  1. ,  1.2, -0.4],
           [ 0.6, -0.4,  0.6, -1.4,  1.2,  0.8,  0.2,  1. , -0.8,  0.6],
           [ 0.6, -0.4,  0.6,  0.6, -0.8, -1.2,  0.2,  0. , -0.8,  0.6]])
    >>> assert (sc.inverse_transform(trans) == X).all()
    """

    requires_y: bool = False

    def __init__(
        self,
        AA: float = 2,
        Aa: float = 1,
        aa: float = 0,
        copy: bool = True
    ):
        self.AA = AA
        self.Aa = Aa
        self.aa = aa
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
            del self.Aa_counts_
            del self.aa_counts_

    def partial_fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "NOIAAdditiveScaler":
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
            self.n_samples_seen_: np.ndarray = (
                (~np.isnan(X_))
                .astype(int)
                .sum(axis=0)
            )
            self.Aa_counts_: np.ndarray = (
                (X_ == self.Aa)
                .astype(int)
                .sum(axis=0)
            )
            self.aa_counts_: np.ndarray = (
                (X_ == self.aa)
                .astype(int)
                .sum(axis=0)
            )
        else:
            assert X_.shape[1] == self.n_features_in_, \
                "Must have same number of features"
            self.n_samples_seen_ += (~np.isnan(X_)).astype(int).sum(axis=0)
            self.Aa_counts_ += (
                (X_ == self.Aa)
                .astype(int)
                .sum(axis=0)
            )
            self.aa_counts_ += (
                (X_ == self.aa)
                .astype(int)
                .sum(axis=0)
            )

        self.n_features_in_: int = self.aa_counts_.shape[0]
        self.n_features_ = self.n_features_in_
        return self

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "NOIAAdditiveScaler":
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

    def transform(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None,
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
            force_all_finite="allow-nan",
            estimator=self,
        )

        if X_.shape[1] != self.n_features_in_:
            raise ValueError("Must have same number of features")

        p_aa = self.aa_counts_ / self.n_samples_seen_
        p_Aa = self.Aa_counts_ / self.n_samples_seen_

        dom = -1 * (-p_Aa - 2 * p_aa) * (X_ == self.AA).astype(float)
        het = -1 * (1 - p_Aa - 2 * p_aa) * (X_ == self.Aa).astype(float)
        rec = -1 * (2 - p_Aa - 2 * p_aa) * (X_ == self.aa).astype(float)
        return dom + het + rec

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

        p_aa = self.aa_counts_ / self.n_samples_seen_
        p_Aa = self.Aa_counts_ / self.n_samples_seen_

        dom = self.AA * np.isclose(X_, -1 * (-p_Aa - 2 * p_aa)).astype(float)
        het = self.Aa * np.isclose(X_, -1 * (1 - p_Aa - 2 * p_aa)).astype(float)  # noqa
        rec = self.aa * np.isclose(X_, -1 * (2 - p_Aa - 2 * p_aa)).astype(float)  # noqa
        return dom + het + rec


class NOIADominanceScaler(TransformerMixin, BaseEstimator):
    """Scale each marker by the minor allele frequency

    This is the same scaling applied to the Van Raden similarity computation
    before the dot-product is taken.
    It is somewhat analogous to mean-centering.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.preprocessor import NOIADominanceScaler
    >>> from selectml.data import basic
    >>> X, _, _ = basic()
    >>> X = np.unique(X, axis=0)
    >>> sc = NOIADominanceScaler()
    >>> trans = sc.fit_transform(X)
    >>> trans
    array([[-0.375     ,  0.        ,  0.75      , -0.125     ,  0.57142857,
             0.57142857, -0.28571429, -0.2       ,  0.57142857,  0.        ],
           [ 0.75      ,  0.        , -0.125     ,  0.75      ,  0.57142857,
             0.57142857, -0.57142857, -0.2       ,  0.57142857,  0.        ],
           [-0.125     ,  0.        , -0.375     , -0.125     , -0.28571429,
            -0.28571429, -0.28571429, -0.2       , -0.57142857,  0.        ],
           [-0.125     ,  0.        , -0.125     , -0.375     , -0.57142857,
            -0.28571429,  0.57142857, -0.2       , -0.28571429,  0.        ],
           [-0.125     ,  0.        , -0.125     , -0.125     , -0.28571429,
            -0.57142857,  0.57142857,  0.8       , -0.28571429,  0.        ]])
    """

    requires_y: bool = False

    def __init__(
        self,
        AA: float = 2.,
        Aa: float = 1.,
        aa: float = 0.,
        copy: bool = True
    ):
        self.AA = AA
        self.Aa = Aa
        self.aa = aa
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
            del self.AA_counts_
            del self.Aa_counts_
            del self.aa_counts_

    def partial_fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "NOIADominanceScaler":
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
            self.n_samples_seen_: np.ndarray = (
                (~np.isnan(X_))
                .astype(int)
                .sum(axis=0)
            )
            self.AA_counts_: np.ndarray = (
                (X_ == self.AA)
                .astype(int)
                .sum(axis=0)
            )
            self.Aa_counts_: np.ndarray = (
                (X_ == self.Aa)
                .astype(int)
                .sum(axis=0)
            )
            self.aa_counts_: np.ndarray = (
                (X_ == self.aa)
                .astype(int)
                .sum(axis=0)
            )
        else:
            assert X_.shape[1] == self.n_features_in_, \
                "Must have same number of features"
            self.n_samples_seen_ += (~np.isnan(X_)).astype(int).sum(axis=0)
            self.AA_counts_ += (
                (X_ == self.AA)
                .astype(int)
                .sum(axis=0)
            )
            self.Aa_counts_ += (
                (X_ == self.Aa)
                .astype(int)
                .sum(axis=0)
            )
            self.aa_counts_ += (
                (X_ == self.aa)
                .astype(int)
                .sum(axis=0)
            )

        self.n_features_in_: int = self.aa_counts_.shape[0]
        self.n_features_ = self.n_features_in_
        return self

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "NOIADominanceScaler":
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

    def transform(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None,
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
            force_all_finite="allow-nan",
            estimator=self,
        )

        if X_.shape[1] != self.n_features_in_:
            raise ValueError("Must have same number of features")

        p_aa = self.aa_counts_ / self.n_samples_seen_
        p_Aa = self.Aa_counts_ / self.n_samples_seen_
        p_AA = self.AA_counts_ / self.n_samples_seen_

        dom_vals = -1 * (2 * p_Aa * p_aa) / (p_AA + p_aa - (p_AA - p_aa) ** 2)
        het_vals = (4 * p_AA * p_aa) / (p_AA + p_aa - (p_AA - p_aa) ** 2)
        rec_vals = -1 * (2 * p_AA * p_Aa) / (p_AA + p_aa - (p_AA - p_aa) ** 2)
        dom = dom_vals * np.isclose(X_, self.AA).astype(float)
        het = het_vals * np.isclose(X_, self.Aa).astype(float)
        rec = rec_vals * np.isclose(X_, self.aa).astype(float)
        return dom + het + rec


class OrdinalTransformer(TransformerMixin, BaseEstimator):
    """Convert a continuous variable into a multiclass classification.  # noqa
    This is used for ordinal regression situations.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.preprocessor import OrdinalTransformer
    >>> np.random.seed(5)
    >>> x = np.random.uniform(0, 3, size=(5, 2))
    >>> ot = OrdinalTransformer(boundaries=[np.arange(3 + 1), np.arange(3 + 1)])
    >>> xhat = ot.fit_transform(x)
    >>> xhat
    array([[0.66597951, 0.        , 0.        , 1.        , 1.        , 0.61219692],
           [0.62015747, 0.        , 0.        , 1.        , 1.        , 0.75583272],
           [1.        , 0.46523357, 0.        , 1.        , 0.83523159, 0.        ],
           [1.        , 1.        , 0.29772357, 1.        , 0.55525396, 0.        ],
           [0.8904015 , 0.        , 0.        , 0.56316369, 0.        , 0.        ]])
    >>> assert np.isclose(ot.inverse_transform(xhat), x).all(), (x, xhat)
    """

    requires_y: bool = False

    def __init__(
        self,
        *,
        boundaries: "Union[str, Sequence[npt.ArrayLike]]" = "auto",
        dtype: "Type" = float,
    ):
        self.boundaries = boundaries
        self.dtype = dtype
        return

    def _reset(self):
        return

    def _check_X(
        self,
        X: "npt.ArrayLike",
        force_all_finite: "Union[str, bool]" = True
    ):
        """
        Perform custom check_array:
        - convert list of strings to object dtype
        - check for missing values for object dtype data (check_array does
          not do that)
        - return list of features (arrays): this list of features is
          constructed feature by feature to preserve the data types
          of pandas DataFrame columns, as otherwise information is lost
          and cannot be used, eg for the `categories_` attribute.
        """
        if not (hasattr(X, "iloc") and getattr(X, "ndim", 0) == 2):
            # if not a dataframe, do normal check_array validation
            X_temp: np.ndarray = check_array(
                np.array(X),
                dtype=None,
                force_all_finite=force_all_finite
            )

            if (
                not hasattr(X, "dtype")
                and np.issubdtype(X_temp.dtype, np.str_)
            ):
                X_ = check_array(
                    np.array(X),
                    dtype=object,
                    force_all_finite=force_all_finite
                )
            else:
                X_ = X_temp
            needs_validation: "Union[str, bool]" = False
            n_samples, n_features = X_.shape
        else:
            # pandas dataframe, do validation later column by column, in order
            # to keep the dtype information to be used in the encoder.
            needs_validation = force_all_finite
            assert isinstance(X, pd.DataFrame)
            X_ = X
            n_samples, n_features = X_.shape

        X_columns = []

        for i in range(n_features):
            Xi = self._get_feature(X_, feature_idx=i)
            Xi = check_array(
                Xi,
                ensure_2d=False,
                dtype=None,
                force_all_finite=needs_validation
            )
            X_columns.append(Xi)

        return X_columns, n_samples, n_features

    def _get_feature(
        self,
        X: "Union[npt.ArrayLike, pd.DataFrame]",
        feature_idx: int
    ) -> "np.ndarray":
        if hasattr(X, "iloc"):
            assert isinstance(X, pd.DataFrame)
            # pandas dataframes
            return X.iloc[:, feature_idx].values
        else:
            # numpy arrays, sparse arrays
            return np.array(X)[:, feature_idx]

    @staticmethod
    def _find_range(Xi: "npt.ArrayLike") -> "np.ndarray":
        Xi_ = np.array(Xi)
        min_ = np.floor(np.min(Xi_, where=~np.isnan(Xi_), initial=np.inf))
        max_ = np.ceil(np.max(Xi_, where=~np.isnan(Xi_), initial=-np.inf)) + 0.1  # noqa
        return np.arange(min_, max_, 1)

    def _fit(
        self,
        X: "npt.ArrayLike",
        force_all_finite: "Union[bool, str]" = True
    ):
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        X_list, n_samples, n_features = self._check_X(
            X, force_all_finite=force_all_finite
        )
        self.n_features_in_ = n_features

        if self.boundaries != "auto":
            if len(self.boundaries) != n_features:
                raise ValueError(
                    "Shape mismatch: if boundaries is an array,"
                    " it has to be of shape (n_features,)."
                )

        self.categories_ = []

        for i in range(n_features):
            Xi = X_list[i]
            if self.boundaries == "auto":
                cats = self._find_range(Xi)
            else:
                cats = np.sort(np.array(self.boundaries[i], dtype=Xi.dtype))
                if Xi.dtype.kind in "OUS":
                    print(Xi.dtype.kind)
                    raise ValueError("Input type must be numeric")

                if np.any(np.isnan(cats)):
                    raise ValueError(
                        "Cannot support nan values in ordinal categories"
                    )
            self.categories_.append(cats)

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None
    ) -> "OrdinalTransformer":
        self._fit(X, force_all_finite="allow-nan")
        return self

    def _transform_column(
        self,
        Xi: "np.ndarray",
        cats: "np.ndarray"
    ) -> np.ndarray:
        diffs = np.diff(cats)
        preds: "np.ndarray" = (
            np.expand_dims(Xi, -1) -
            np.expand_dims(cats[:-1], 0)
        )
        whole = (preds >= diffs).astype(int)
        fractional = (
            ((preds // diffs) == 0).astype(int)
            * (preds % diffs) / diffs
        )
        return whole + fractional

    def transform(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None,
    ) -> np.ndarray:
        check_is_fitted(self)

        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)
        X_list, n_samples, n_features = self._check_X(
            X, force_all_finite="allow-nan"
        )

        arrays = []
        for i in range(n_features):
            Xi = X_list[i]
            cats = self.categories_[i]
            trans = self._transform_column(Xi, cats)
            arrays.append(trans)

        return np.concatenate(arrays, axis=1)

    def inverse_transform(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None,
    ) -> np.ndarray:
        check_is_fitted(self)

        X_: np.ndarray = check_array(X, accept_sparse=False)

        n_samples, _ = X_.shape
        n_features = len(self.categories_)

        n_transformed_features = sum(len(c) - 1 for c in self.categories_)
        if X_.shape[1] != n_transformed_features:
            raise ValueError(
                "Shape of the passed X data is not correct. "
                f"Expected {n_transformed_features} columns got {X_.shape[1]}."
            )

        dt = np.find_common_type([c.dtype for c in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        def mappl(Xi, mapper, diffs):
            pos, = np.where(Xi == 1)
            if len(pos) == 0:
                pos = 0
            else:
                pos = pos.max() + 1

            extra = np.sum(Xi / diffs, where=~(Xi == 1))
            return mapper[pos] + extra

        j = 0
        for i in range(n_features):
            cats = self.categories_[i]
            n_categories = len(cats) - 1
            diffs = np.diff(cats)

            Xi = X_[:, j:(j + n_categories)]
            X_tr[:, i] = np.apply_along_axis(
                lambda xi: mappl(xi, cats, diffs),
                1,
                Xi
            )

            j += n_categories

        return X_tr
