#!/usr/bin/env python3

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    import numpy.typing as npt


class MAFSelector(SelectorMixin, BaseEstimator):

    """ Filter out markers with a low minor allele frequency.

    This calculates the minor allele frequency given a standard encoding and
    ploidy, then filters out loci that have a lower proportion than the
    threshold. Note that this implementation can only handle biallelic
    markers. Expects the 0, 1, 2 encoding, where 0 and 2 are homozygous
    and 1 is heterozygous. This function should generalise to other ploidies
    given that the correct `ploidy` is set.
    Proportions are calculated assuming that 0 is the minor allele, but
    filtering considers both as the potential minor allele

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.feature_selection import MAFSelector
    >>> from selectml.data import basic
    >>> X, _, _ = basic()
    >>> X = np.unique(X, axis=0)
    >>> X
    array([[0., 0., 1., 2., 1., 1., 0., 0., 1., 1.],
           [1., 2., 2., 1., 1., 1., 2., 0., 1., 1.],
           [2., 0., 0., 2., 0., 2., 0., 2., 2., 1.],
           [2., 0., 2., 0., 2., 2., 1., 2., 0., 2.],
           [2., 0., 2., 2., 0., 0., 1., 1., 0., 2.]])
    >>> ms = MAFSelector(threshold=0.3)
    >>> ms.fit_transform(X)
    array([[1., 1., 0., 0., 1.],
           [1., 1., 2., 0., 1.],
           [0., 2., 0., 2., 2.],
           [2., 2., 1., 2., 0.],
           [0., 0., 1., 1., 0.]])
    >>> ms.allele_counts_
    array([3., 8., 3., 3., 6., 4., 6., 5., 6., 3.])
    >>> ms.n_samples_seen_
    array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    >>> ms._get_support_mask()
    array([False, False, False, False,  True,  True,  True,  True,  True,
           False])
    """

    requires_y: bool = False

    def __init__(self, threshold: float = 0.0, ploidy: int = 2):
        self.threshold = threshold
        self.ploidy = ploidy
        return

    def _reset(self):
        """Reset internal data-dependent state of the selector, if necessary.
        __init__ parameters are not touched.
        """

        if hasattr(self, "n_samples_seen_"):
            del self.n_samples_seen_
            del self.n_features_in_
            del self.n_features_
            del self.allele_counts_

        return

    def partial_fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]",
        **kwargs
    ) -> "MAFSelector":
        """
        """

        first_pass: bool = not hasattr(self, "n_samples_seen_")

        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite="allow-nan",
            estimator=self
        )

        all_ok = True

        if not all_ok:
            raise ValueError(
                "Encountered a value less than 0 or greater "
                f"than {self.ploidy}."
            )

        if first_pass:
            self.n_samples_seen_: np.ndarray = (~np.isnan(X_)).sum(axis=0)
            self.allele_counts_: np.ndarray = (self.ploidy - X_).sum(axis=0)
        else:
            assert X_.shape[1] == self.n_features_in_, \
                "Must have same number of features"
            self.n_samples_seen_ = (
                self.n_samples_seen_ +
                (~np.isnan(X_)).sum(axis=0)
            )
            self.allele_counts_ = (
                self.allele_counts_ +
                (self.ploidy - X_).sum(axis=0)
            )

        self.n_features_in_: int = self.allele_counts_.shape[0]
        self.n_features_ = self.n_features_in_

        return self

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None,
        **kwargs
    ) -> "MAFSelector":
        """
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def _get_support_mask(self) -> "np.ndarray":
        check_is_fitted(self)
        p_i = self.allele_counts_ / (self.ploidy * self.n_samples_seen_)
        # We have to check both upper and lower because we aren't guaranteed
        # that variant 1 is the minor allele.
        mask = (p_i > self.threshold) & (p_i < (1 - self.threshold))
        return mask


class MultiSURF(SelectorMixin, BaseEstimator):

    """ Filter markers using the MultiSURF algorithm.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.feature_selection import MultiSURF
    >>> from selectml.data import basic
    >>> X, y, _ = basic()
    >>> y = np.expand_dims(y, -1)
    >>> X = X[::5]
    >>> y = y[::5]
    >>> ms = MultiSURF(n=5)
    >>> ms.fit_transform(X, y)
    array([[1., 1., 1., 0., 1.],
           [2., 0., 2., 2., 2.],
           [2., 2., 2., 2., 0.],
           [0., 1., 1., 0., 1.],
           [2., 0., 0., 1., 0.]])
    >>> ms.relevance_
    array([ 8.81620816e-39, -1.76324163e-38,  2.93873605e-39, -2.93873605e-39,
            1.17549442e-38,  1.17549442e-38, -8.81620816e-39,  1.17549442e-38,
            5.87747210e-39,  2.93873605e-39])
    >>> ms._get_support_mask()
    array([ True, False, False, False,  True,  True, False,  True,  True,
           False])
    """

    requires_y: bool = True

    def __init__(
        self,
        threshold: "Optional[float]" = None,
        n: "Optional[int]" = None,
        sd: "Optional[float]" = None,
        batchsize: int = 200,
        nepoch: int = 1,
        distance_metric: str = "euclidean",
        random_state: "Optional[int]" = None
    ):
        self.sd = sd

        if (threshold is not None) and (n is not None):
            raise ValueError(
                "Please select a value for threshold or n, "
                "but not both."
            )
        elif (threshold is None) and (n is None):
            threshold = 0.0

        self.threshold = threshold
        self.n = n
        self.distance_metric = "euclidean"

        # It doesn't work otherwise
        assert batchsize > 2
        self.batchsize = batchsize
        assert nepoch > 0
        self.nepoch = nepoch

        self.random_state = random_state
        return

    def _reset(self):
        if hasattr(self, "n_samples_seen_"):
            del self.n_samples_seen_
            del self.n_features_in_
            del self.n_features_
            del self.relevance_

        if hasattr(self, "sd_"):
            del self.sd_

    @staticmethod
    def relief_update(
        X: "npt.ArrayLike",
        mat: "npt.ArrayLike"
    ) -> "np.ndarray":
        X_ = np.array(X)
        mat_ = np.array(mat)

        where = np.where(mat_)
        X0 = X_[where[0]]
        X1 = X_[where[1]]

        Xdiffs = np.nansum(np.abs(X0 - X1), axis=0)
        nmatches = Xdiffs.sum()

        n = X_.shape[0]
        return Xdiffs / (n * nmatches + np.finfo(np.float32).min)

    def find_ydiffs(self, y: "npt.ArrayLike") -> "np.ndarray":
        y_ = np.array(y)

        if self.sd is not None:
            sd = self.sd
        elif hasattr(self, "sd_"):
            assert isinstance(self.sd_, float)
            sd = self.sd_
        else:
            sd = np.nanstd(y_)

        if len(y_.shape) == 1:
            y_ = np.expand_dims(y_, -1)
        elif len(y_.shape) == 2:
            assert y_.shape[1] == 1, \
                "Cannot currently handle multiple response"
        else:
            raise ValueError("y must have 1 or 2 dimensions.")

        return np.abs(y_ - y_.T) < sd

    def find_neighbors(self, X: "npt.ArrayLike") -> "np.ndarray":
        from scipy.spatial.distance import pdist, squareform

        X_ = np.array(X)

        Xdist = squareform(pdist(X_, metric=self.distance_metric))
        Xdist[np.diag_indices_from(Xdist)] = np.nan

        radii = np.nanmean(Xdist, axis=1) - (np.nanstd(Xdist, axis=1) / 2.)
        neighbors = (Xdist < radii) & (Xdist > 0.0)
        return neighbors

    def partial_fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]",
        **kwargs
    ) -> "MultiSURF":
        first_pass: bool = not hasattr(self, "n_samples_seen_")
        assert y is not None

        X_: np.ndarray = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite="allow-nan",
            estimator=self
        )

        y_: np.ndarray = check_array(
            y,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite="allow-nan",
            estimator=self
        )

        if first_pass:
            self.n_samples_seen_: int = X_.shape[0]
            self.relevance_: np.ndarray = np.zeros((X_.shape[1],), dtype=float)
            self.n_features_in_ = X_.shape[1]
            self.n_features_ = self.n_features_in_
        else:
            assert X_.shape[1] == self.n_features_in_, \
                "Must have same number of features"

        ydiffs = self.find_ydiffs(y_)
        neighbors = self.find_neighbors(X_)

        hits = neighbors & ydiffs
        misses = neighbors & (~ydiffs)

        hit_update = self.relief_update(X_, hits)
        miss_update = self.relief_update(X_, misses)

        update = (-1. * hit_update) + miss_update
        self.relevance_ += update
        return self

    def fit(
        self,
        X: "npt.ArrayLike",
        y: "Optional[npt.ArrayLike]" = None,
    ) -> "MultiSURF":
        self._reset()
        assert y is not None
        self.sd_ = np.nanstd(y)

        if np.shape(X)[0] <= self.batchsize:
            for epoch in range(self.nepoch):
                self.partial_fit(X, y)
            return self

        X_ = np.array(X)
        y_ = np.array(y)

        rng = np.random.default_rng(seed=self.random_state)
        indices = np.arange(np.shape(X_)[0])
        nsplits = np.ceil(np.shape(X_)[0] / self.batchsize)
        for epoch in range(self.nepoch):
            rng.shuffle(indices)

            chunks = np.array_split(indices, nsplits)

            for chunk in chunks:
                X_chunk = X_[chunk]
                y_chunk = y_[chunk]

                self.partial_fit(X_chunk, y_chunk)

        return self

    def _get_support_mask(self) -> "np.ndarray":
        if self.n is not None:
            sorted_ = np.argsort(self.relevance_)[-self.n:]
            out = np.full(self.relevance_.shape, False)
            out[sorted_] = True
            return out
        else:
            assert self.threshold is not None
            mask = self.relevance_ > self.threshold
            return mask
