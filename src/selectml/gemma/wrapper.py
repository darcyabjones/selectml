#!/usr/bin/env python3

from contextlib import contextmanager

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import (
    check_array,
)

import numpy as np
import pandas as pd
from os.path import join as pjoin

from typing import TYPE_CHECKING, cast
if TYPE_CHECKING:
    from typing import IO, Union, Optional, List, Any, Iterator
    from typing import Tuple, Sequence, Dict, Iterable
    import numpy.typing as npt


class GEMMA(object):

    def __init__(
        self,
        exe: str = "gemma",
        AA: "Union[int, float]" = 2,
        Aa: "Union[int, float]" = 1,
        aa: "Union[int, float]" = 0,
        r2: float = 1.0,
    ):
        self.exe = exe
        self.AA = AA
        self.Aa = Aa
        self.aa = aa
        self.r2 = r2
        return

    def genos_to_bimbam(
        self,
        X: "npt.ArrayLike",
        marker_columns: "Optional[List[Any]]" = None
    ) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = df.copy()

        if marker_columns is None:
            marker_columns = list(df.columns)

        df = df.loc[:, marker_columns].T.astype(float)
        df[np.isclose(df.values, float(self.AA))] = 1
        df[np.isclose(df.values, float(self.Aa))] = 0.5
        df[np.isclose(df.values, float(self.aa))] = 0
        df.index.name = "SNP_ID"
        df.reset_index(inplace=True, drop=False)
        df["MAJ"] = "X"
        df["MIN"] = "Y"
        df = df[["SNP_ID", "MAJ", "MIN"] + list(df.columns[1:-2])]
        return df

    @staticmethod
    def prep_groups(X: "npt.ArrayLike") -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            df: np.ndarray = X.values
        else:
            df = np.asarray(X)
        singular = np.all(df.sum(axis=1) == 1)

        if singular:
            df = df[:, 1:]

        # df = np.column_stack((df, np.ones(df.shape[0])))
        return df

    @staticmethod
    def _check_is_singular(X):
        is_singular = np.linalg.det(X.T.dot(X)) == 0.0
        if is_singular:
            raise ValueError(
                "Your marker array is singular, "
                "so it cannot be used with GEMMA."
            )

    @contextmanager
    def write_inputs(
        self,
        X: "npt.ArrayLike",
        y: "npt.ArrayLike",
        groups: "Optional[npt.ArrayLike]" = None,
        covariates: "Optional[npt.ArrayLike]" = None,
    ) -> "Iterator":
        from tempfile import NamedTemporaryFile
        X_file = NamedTemporaryFile(mode="w+")
        y_file = NamedTemporaryFile(mode="w+")

        if (covariates is not None) or (groups is not None):
            cov_file: "Optional[IO[str]]" = NamedTemporaryFile(mode="w+")
        else:
            cov_file = None

        try:
            #  self._check_is_singular(np.asarray(X))
            X_ = self.genos_to_bimbam(X)
            X_.to_csv(X_file, index=False, header=False, na_rep="NA")
            X_file.seek(0)

            y_ = np.asarray(y)
            if len(y_.shape) == 1:
                y_ = np.expand_dims(y_, -1)

            pd.DataFrame(y_).to_csv(
                y_file,
                index=False,
                header=False,
                na_rep="NA"
            )
            y_file.seek(0)

            cov: "List[np.ndarray]" = []

            if groups is not None:
                cov.append(self.prep_groups(groups))

            if covariates is not None:
                cov.append(np.asarray(covariates))

            if len(cov) > 0:
                """
                We add a column of ones to include an intercept term.
                """

                assert cov_file is not None
                cov_ = np.concatenate(
                    [np.ones((y_.shape[0], 1)), *cov],
                    axis=1
                )
                pd.DataFrame(cov_).to_csv(
                    cov_file,
                    index=False,
                    header=False,
                    na_rep="NA"
                )
                cov_file.seek(0)

            del X_
            del y_
            del cov

            yield X_file, y_file, cov_file
        finally:
            X_file.close()
            y_file.close()

            if cov_file is not None:
                cov_file.close()

        return

    def get_assocs(
        self,
        X: "npt.ArrayLike",
        y: "npt.ArrayLike",
        groups: "Optional[npt.ArrayLike]" = None,
        covariates: "Optional[npt.ArrayLike]" = None,
    ) -> pd.DataFrame:
        from tempfile import TemporaryDirectory
        import subprocess

        y_ = np.array(y)

        if len(y_.shape) == 1:
            y_ = np.expand_dims(y_, -1)

        with TemporaryDirectory() as tdir_name, \
                self.write_inputs(X, y_, groups, covariates) as (X_handle, y_handle, cov_handle):  # noqa: E501

            del y_
            cmd = [
                self.exe,
                "-g", X_handle.name,
                "-p", y_handle.name,
                "-gk",
                "-outdir", tdir_name
            ]

            if cov_handle is not None:
                cmd.extend(["-c", cov_handle.name])

            subprocess.run(cmd, check=True, capture_output=True)

            cmd = [
                self.exe,
                "-g", X_handle.name,
                "-p", y_handle.name,
                "-k", pjoin(tdir_name, "result.cXX.txt"),
                "-outdir", tdir_name,
                "-lmm", "1",
                "-miss", "1",
                "-r2", str(self.r2),
                "-notsnp",
                "-n"
            ]

            if cov_handle is not None:
                cmd.extend(["-c", cov_handle.name])

            subprocess.run(cmd, check=True, capture_output=True)
            df = pd.read_csv(pjoin(tdir_name, "result.assoc.txt"), sep="\t")
        return df


class GEMMASelector(SelectorMixin, BaseEstimator):

    """ Filter markers using the GEMMA GWAS p-values.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.feature_selection import GEMMASelector
    >>> from selectml.data import basic
    >>> X, y, _ = basic()
    >>> y = np.expand_dims(y, -1)
    >>> X = X[::5]
    >>> y = y[::5]
    >>> ms = GEMMASelector(n=5)
    >>> ms.fit_transform(X, y)  # doctest: +SKIP
    array([[1., 2., 1., 2., 0.],
           [2., 0., 0., 0., 2.],
           [2., 0., 2., 1., 2.],
           [0., 0., 1., 0., 0.],
           [2., 0., 0., 1., 1.]])
    >>> ms.pvalues_  # doctest: +SKIP
    array([0.2834333 , 0.00282126, 0.5162332 , 0.9332765 , 0.4073413 ,
           0.7417847 , 0.1985471 , 0.3603569 , 0.6030616 , 1.        ])
    >>> ms.coefs_  # doctest: +SKIP
    array([11.90039 , 19.44042 ,  7.874934,  1.058424,  7.998619,  4.122992,
           13.62786 , -8.704236,  6.779367,  0.      ])
    >>> ms._get_support_mask()  # doctest: +SKIP
    array([ True,  True, False, False,  True, False,  True,  True, False,
           False])
    """

    requires_y: bool = True

    def __init__(
        self,
        *,
        use_groups=False,
        use_covariates=False,
        threshold: "Optional[float]" = None,
        n: "Optional[int]" = None,
        exe: str = "gemma",
        AA: "Union[int, float]" = 2,
        Aa: "Union[int, float]" = 1,
        aa: "Union[int, float]" = 0,
        r2: float = 1.0,
    ):
        if (threshold is not None) and (n is not None):
            raise ValueError(
                "Please select a value for threshold or n, "
                "but not both."
            )
        elif (threshold is None) and (n is None):
            threshold = 0.1

        self.use_covariates = use_covariates
        self.use_groups = use_groups

        self.threshold = threshold
        self.n = n
        self.exe = exe
        self.AA = AA
        self.Aa = Aa
        self.aa = aa
        self.r2 = r2
        return

    def _reset(self):
        if hasattr(self, "coefs_se_"):
            del self.n_samples_seen_
            del self.n_features_in_
            del self.n_features_
            del self.pvalues_
            del self.coefs_
            del self.coefs_se_

    def partial_fit(
        self,
        X: "Union[npt.ArrayLike, List[npt.ArrayLike], Tuple[npt.ArrayLike, ...], Dict[str, npt.ArrayLike]]",  # noqa: E501
        y: "Optional[npt.ArrayLike]" = None,
        **kwargs
    ) -> "GEMMASelector":
        import pandas as pd
        from ..gemma import GEMMA
        first_pass: bool = not hasattr(self, "coefs_se_")
        assert y is not None

        X_ = self._check_X(X, reset=first_pass)

        y_: np.ndarray = np.asarray(check_array(
            y,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype="numeric",
            force_all_finite="allow-nan",
            ensure_2d=False,
            ensure_min_samples=1,
            estimator=self
        ))

        self._check_X_y(X_, y_)

        grp: "Optional[np.ndarray]" = None
        cov: "Optional[np.ndarray]" = None

        if isinstance(X_, np.ndarray):
            markers = X_
        elif isinstance(X_, dict):
            assert "markers" in X_
            markers = X_["markers"]
            grp = X_.get("groups", None)
            cov = X_.get("covariates", None)
        elif isinstance(X_, list):
            markers = X_.pop(0)
            if self.use_groups:
                assert len(X_) > 0
                grp = X_.pop(0)

            if self.use_covariates:
                assert len(X_) > 0
                cov = X_.pop(0)
        else:
            raise ValueError("This shouldn't be reachable")

        g = GEMMA(
            exe=self.exe,
            AA=self.AA,
            Aa=self.Aa,
            aa=self.aa,
            r2=self.r2,
        )

        results = g.get_assocs(markers, y_, grp, cov)
        results = results[["rs", "beta", "se", "p_wald"]]

        X_cols = set(np.arange(markers.shape[1]))
        X_cols = X_cols.difference(results["rs"])
        remaining = pd.DataFrame({
            "rs": list(X_cols),
            "beta": 0.0,
            "se": 0.0,
            "p_wald": 1.0
        })

        results = pd.concat([results, remaining], axis=0, ignore_index=True)
        del remaining
        del X_cols

        results.sort_values("rs", inplace=True)
        self.pvalues_ = results.loc[:, "p_wald"].values
        self.coefs_ = results.loc[:, "beta"].values
        self.coefs_se_ = results.loc[:, "se"].values
        return self

    def fit(
        self,
        X: "Union[npt.ArrayLike, List[npt.ArrayLike], Tuple[npt.ArrayLike, ...], Dict[str, npt.ArrayLike]]",  # noqa: E501
        y: "Optional[npt.ArrayLike]" = None,
        **kwargs
    ) -> "GEMMASelector":
        self._reset()
        assert y is not None
        res = self.partial_fit(X, y, **kwargs)
        return res

    def transform(self, X):
        """Reduce X to the selected features.
        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        # note: we use _safe_tags instead of _get_tags because this is a
        # public Mixin.
        X_ = self._check_X(X, reset=False)

        if isinstance(X_, np.ndarray):
            markers = X_
        elif isinstance(X_, dict):
            assert "markers" in X_
            markers = X_["markers"]
        elif isinstance(X_, list):
            markers = X_.pop(0)
        else:
            raise ValueError("This shouldn't be reachable")

        return self._transform(markers)

    def _get_support_mask(self) -> "np.ndarray":
        if self.n is not None:
            sorted_ = np.argsort(self.pvalues_)[:self.n]
            out = np.full(self.pvalues_.shape, False)
            out[sorted_] = True
            return out
        else:
            assert self.threshold is not None
            mask = self.pvalues_ < self.threshold
            return mask

    @staticmethod
    def _all_sample_first_dim(
        arrs: "Sequence[np.ndarray]"
    ) -> "Tuple[bool, Optional[int]]":
        s = set(it.shape[0] for it in arrs)
        is_single = len(s) <= 1

        if len(s) == 1:
            length: "Optional[int]" = s.pop()
        else:
            length = None
        return is_single, length

    def _all_sample_same_first_dim(self, arrs: "Sequence[np.ndarray]") -> bool:
        b, _ = self._all_sample_first_dim(arrs)
        return b

    def _check_X(  # noqa: C901
        self,
        X: "Union[npt.ArrayLike, List[npt.ArrayLike], Tuple[npt.ArrayLike, ...], Dict[str, npt.ArrayLike]]",  # noqa: E501
        reset: bool,
        estimator: "Optional[str]" = None,
    ) -> "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]":  # noqa: E501
        multi_input = (
            isinstance(X, (list, tuple, dict))
            and not any(isinstance(xi, (int, float)) for xi in X)
        )

        if multi_input and isinstance(X, tuple):
            X = list(X)

        def _check_X_array(arr: "npt.ArrayLike") -> "np.ndarray":
            return np.asarray(check_array(
                arr,
                accept_sparse=False,
                accept_large_sparse=False,
                force_all_finite=True,
                ensure_min_samples=1,
                ensure_min_features=1,
                allow_nd=True,
                ensure_2d=True,
                dtype=float,
                estimator=estimator,
                input_name="X",
            ))

        if multi_input and isinstance(X, list):
            X_: "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]" = [  # noqa: E501
                _check_X_array(xi)
                for xi
                in X
            ]

            assert isinstance(X_, list)
            if not self._all_sample_same_first_dim(X_):
                raise ValueError(
                    "The X-arrays have different numbers of samples."
                )

            X_dtype_: "Union[List[np.dtype], Dict[str, np.dtype], np.dtype]" = [  # noqa: E501
                xi.dtype for xi in X_
            ]
            n_features_in_: "Union[int, List[int], Dict[str, int]]" = [
                xi.shape[1] for xi in X_
            ]
            X_shape_: "Union[List[Tuple[int, ...]], Dict[str, Tuple[int, ...]], Tuple[int, ...]]" = [  # noqa: E501
                xi.shape for xi in X_
            ]
        elif multi_input and isinstance(X, dict):
            X_ = {k: _check_X_array(xi) for k, xi in X.items()}

            assert isinstance(X_, dict)
            if not self._all_sample_same_first_dim(list(X_.values())):
                raise ValueError(
                    "The X-arrays have different numbers of samples."
                )

            X_dtype_ = {k: xi.dtype for k, xi in X_.items()}
            n_features_in_ = {k: xi.shape[1] for k, xi in X_.items()}
            X_shape_ = {k: xi.shape for k, xi in X_.items()}
        else:
            assert isinstance(X, np.ndarray)
            X_ = _check_X_array(X)
            X_dtype_ = X_.dtype
            n_features_in_ = X_.shape[1]
            X_shape_ = X_.shape

        if reset:
            self.X_dtype_ = X_dtype_
            self.X_shape_ = X_shape_
            self.n_features_in_ = n_features_in_

        elif multi_input:
            isdict = isinstance(X_, dict)

            if isdict:
                assert isinstance(X_, dict)
                keys: "Iterable[Union[int, str]]" = X_.keys()
            else:
                assert isinstance(X_, list)
                keys = range(len(X_))

            for k in keys:
                if isdict:
                    assert isinstance(k, str)
                    assert isinstance(X_dtype_, dict)
                    assert isinstance(X_shape_, dict)
                    assert isinstance(self.X_dtype_, dict)
                    assert isinstance(self.X_shape_, dict)
                    xi_dtype_ = cast("Dict", X_dtype_)[k]
                    dtype_ = cast("Dict", self.X_dtype_)[k]
                    xi_shape_ = cast("Dict", X_shape_)[k]
                    shape_ = cast("Dict", self.X_shape_)[k]
                else:
                    assert isinstance(k, int)
                    assert isinstance(X_dtype_, list)
                    assert isinstance(X_shape_, list)
                    assert isinstance(self.X_dtype_, list)
                    assert isinstance(self.X_shape_, list)
                    xi_dtype_ = cast("List", X_dtype_)[k]
                    dtype_ = cast("List", self.X_dtype_)[k]
                    xi_shape_ = cast("List", X_shape_)[k]
                    shape_ = cast("List", self.X_shape_)[k]

                assert isinstance(xi_dtype_, np.dtype)
                assert isinstance(dtype_, np.dtype)
                if not np.can_cast(xi_dtype_, dtype_):
                    raise ValueError(
                        f"Got y input {k} with dtype {xi_dtype_},"
                        f" but this {self.name} expected {dtype_}"
                        f" and casting from {xi_dtype_} to {dtype_} "
                        "is not safe!"
                    )
                if len(shape_) != len(xi_shape_):
                    raise ValueError(
                        f"X input {k} has {len(xi_shape_)} dimensions, "
                        f"but this {self.name} is expecting "
                        f"{len(shape_)} dimensions in X {k}."
                    )
                if shape_[1:] != xi_shape_[1:]:
                    raise ValueError(
                        f"X has shape {xi_shape_[1:]}, but this "
                        f"{self.name} is expecting X of shape "
                        f"{shape_[1:]}"
                    )
        else:
            assert isinstance(X_dtype_, np.dtype)
            assert isinstance(self.X_dtype_, np.dtype), self.X_dtype_
            if not np.can_cast(X_dtype_, self.X_dtype_):
                raise ValueError(
                    f"Got X with dtype {X_dtype_},"
                    f" but this {self.name} expected {self.X_dtype_}"
                    f" and casting from {X_dtype_} to {self.X_dtype_} "
                    "is not safe!"
                )
            if len(self.X_shape_) != len(X_shape_):
                raise ValueError(
                    f"X has {len(X_shape_)} dimensions, but this "
                    f"{self.name} is expecting {len(self.X_shape_)} "
                    "dimensions in X."
                )

            assert isinstance(X_shape_, tuple)
            assert isinstance(self.X_shape_, tuple)
            if X_shape_[1:] != self.X_shape_[1:]:
                raise ValueError(
                    f"X has shape {X_shape_[1:]}, but this {self.name}"
                    f" is expecting X of shape {self.X_shape_[1:]}"
                )
        return X_

    def _check_X_y(
        self,
        X: "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]",  # noqa: E501
        y: "np.ndarray",  # noqa: E501
        accept_sparse=False,
        *,
        accept_large_sparse=True,
        dtype="numeric",
        order=None,
        copy=False,
        force_all_finite=True,
        ensure_2d=True,
        allow_nd=False,
        multi_output=False,
        ensure_min_samples=1,
        ensure_min_features=1,
        y_numeric=False,
        estimator=None,
    ) -> """Tuple[
            Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],  # noqa: E501
            np.ndarray,
    ]""":
        """
        A rewrite of
        https://github.com/scikit-learn/scikit-learn/blob/4685cf624582cbc9a35d646f239347e54db798dc/sklearn/utils/validation.py#L941"
        to allow for the multi-input bits
        """
        from sklearn.utils.validation import check_consistent_length

        arrays: "List[np.ndarray]" = []

        if isinstance(X, list):
            arrays.extend(X)
        elif isinstance(X, dict):
            arrays.extend(X.values())
        elif X is not None:
            arrays.append(X)

        if y is not None:
            arrays.append(y)

        check_consistent_length(*arrays)
        return X, y
