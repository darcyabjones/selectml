#!/usr/bin/env python3

import numpy as np
from scipy import sparse
from scipy.sparse import issparse

from sklearn.compose import ColumnTransformer

from joblib import Parallel
from sklearn.utils.fixes import np_version, parse_version, delayed
from sklearn.base import clone
from sklearn.utils.validation import check_array

_ERR_MSG_1DCOLUMN = ("1D data passed to a transformer that expects 2D data. "
                     "Try to specify the column selection as a list of one "
                     "item instead of a scalar.")


def _check_X(X):
    """Use check_array only on lists and other non-array-likes / sparse"""
    if hasattr(X, '__array__') or sparse.issparse(X):
        return X
    return check_array(X, force_all_finite='allow-nan', dtype=object)


def _fit_transform_one(transformer,
                       X,
                       y,
                       weight,
                       message_clsname='',
                       message=None,
                       **fit_params):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    if hasattr(transformer, 'fit_transform'):
        res = transformer.fit_transform(X, y, **fit_params)
    else:
        res = transformer.fit(X, y, **fit_params).transform(X)

    if weight is None:
        return res, transformer
    return res * weight, transformer


def _determine_key_type(key, accept_slice=True):  # noqa
    """Determine the data type of key.
    Parameters
    ----------
    key : scalar, slice or array-like
        The key from which we want to infer the data type.
    accept_slice : bool, default=True
        Whether or not to raise an error if the key is a slice.
    Returns
    -------
    dtype : {'int', 'str', 'bool', None}
        Returns the data type of key.
    """
    err_msg = ("No valid specification of the columns. Only a scalar, list or "
               "slice of all integers or all strings, or boolean mask is "
               "allowed")

    dtype_to_str = {int: 'int', str: 'str', bool: 'bool', np.bool_: 'bool'}
    array_dtype_to_str = {'i': 'int', 'u': 'int', 'b': 'bool', 'O': 'str',
                          'U': 'str', 'S': 'str'}

    if key is None:
        return None
    if isinstance(key, tuple(dtype_to_str.keys())):
        try:
            return dtype_to_str[type(key)]
        except KeyError:
            raise ValueError(err_msg)
    if isinstance(key, slice):
        if not accept_slice:
            raise TypeError(
                'Only array-like or scalar are supported. '
                'A Python slice was given.'
            )
        if key.start is None and key.stop is None:
            return None
        key_start_type = _determine_key_type(key.start)
        key_stop_type = _determine_key_type(key.stop)
        if key_start_type is not None and key_stop_type is not None:
            if key_start_type != key_stop_type:
                raise ValueError(err_msg)
        if key_start_type is not None:
            return key_start_type
        return key_stop_type
    if isinstance(key, (list, tuple)):
        unique_key = set(key)
        key_type = {_determine_key_type(elt) for elt in unique_key}
        if not key_type:
            return None
        if len(key_type) != 1:
            raise ValueError(err_msg)
        return key_type.pop()
    if hasattr(key, 'dtype'):
        try:
            return array_dtype_to_str[key.dtype.kind]
        except KeyError:
            raise ValueError(err_msg)
    raise ValueError(err_msg)


def _array_indexing(array, key, key_dtype, axis):
    """Index an array or scipy.sparse consistently across NumPy version."""
    if np_version < parse_version('1.12') or issparse(array):
        # FIXME: Remove the check for NumPy when using >= 1.12
        # check if we have an boolean array-likes to make the proper indexing
        if key_dtype == 'bool':
            key = np.asarray(key)
    if isinstance(key, tuple):
        key = list(key)
    return array[key] if axis == 0 else array[:, key]


def _pandas_indexing(X, key, key_dtype, axis):
    """Index a pandas dataframe or a series."""
    if hasattr(key, 'shape'):
        # Work-around for indexing with read-only key in pandas
        # FIXME: solved in pandas 0.25
        key = np.asarray(key)
        key = key if key.flags.writeable else key.copy()
    elif isinstance(key, tuple):
        key = list(key)
    # check whether we should index with loc or iloc
    indexer = X.iloc if key_dtype == 'int' else X.loc
    return indexer[:, key] if axis else indexer[key]


def _list_indexing(X, key, key_dtype):
    from itertools import compress
    """Index a Python list."""
    if np.isscalar(key) or isinstance(key, slice):
        # key is a slice or a scalar
        return X[key]
    if key_dtype == 'bool':
        # key is a boolean array-like
        return list(compress(X, key))
    # key is a integer array-like of key
    return [X[idx] for idx in key]


def _safe_indexing(X, indices, *, axis=0):
    """Return rows, items or columns of X using indices.
    .. warning::
        This utility is documented, but **private**. This means that
        backward compatibility might be broken without any deprecation
        cycle.
    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series
        Data from which to sample rows, items or columns. `list` are only
        supported when `axis=0`.
    indices : bool, int, str, slice, array-like
        - If `axis=0`, boolean and integer array-like, integer slice,
          and scalar integer are supported.
        - If `axis=1`:
            - to select a single column, `indices` can be of `int` type for
              all `X` types and `str` only for dataframe. The selected subset
              will be 1D, unless `X` is a sparse matrix in which case it will
              be 2D.
            - to select multiples columns, `indices` can be one of the
              following: `list`, `array`, `slice`. The type used in
              these containers can be one of the following: `int`, 'bool' and
              `str`. However, `str` is only supported when `X` is a dataframe.
              The selected subset will be 2D.
    axis : int, default=0
        The axis along which `X` will be subsampled. `axis=0` will select
        rows while `axis=1` will select columns.
    Returns
    -------
    subset
        Subset of X on axis 0 or 1.
    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if indices is None:
        return X

    if axis not in (0, 1):
        raise ValueError(
            "'axis' should be either 0 (to index rows) or 1 (to index "
            " column). Got {} instead.".format(axis)
        )

    indices_dtype = _determine_key_type(indices)

    if axis == 0 and indices_dtype == 'str':
        raise ValueError(
            "String indexing is not supported with 'axis=0'"
        )

    if axis == 1 and X.ndim != 2:
        raise ValueError(
            "'X' should be a 2D NumPy array, 2D sparse matrix or pandas "
            "dataframe when indexing the columns (i.e. 'axis=1'). "
            "Got {} instead with {} dimension(s).".format(type(X), X.ndim)
        )

    if axis == 1 and indices_dtype == 'str' and not hasattr(X, 'loc'):
        raise ValueError(
            "Specifying the columns using strings is only supported for "
            "pandas DataFrames"
        )

    if hasattr(X, "iloc"):
        return _pandas_indexing(X, indices, indices_dtype, axis=axis)
    elif hasattr(X, "shape"):
        return _array_indexing(X, indices, indices_dtype, axis=axis)
    else:
        return _list_indexing(X, indices, indices_dtype)


class MyColumnTransformer(ColumnTransformer):

    def _fit_transform(self, X, y, func, fitted=False, **fit_params):
        """
        Private function to fit and/or transform on demand.
        Return value (transformers and/or transformed X data) depends
        on the passed function.
        ``fitted=True`` ensures the fitted transformers are used.
        """
        transformers = list(
            self._iter(fitted=fitted, replace_strings=True)
        )

        fit_param_steps = self._check_fit_params(**fit_params)
        try:
            return Parallel(n_jobs=self.n_jobs)(
                delayed(func)(
                    transformer=clone(trans) if not fitted else trans,
                    X=_safe_indexing(X, column, axis=1),
                    y=y,
                    weight=weight,
                    message_clsname='ColumnTransformer',
                    message=self._log_message(name, idx, len(transformers)),
                    **fit_param_steps[name]
                )
                for idx, (name, trans, column, weight) in enumerate(
                        self._iter(fitted=fitted, replace_strings=True), 1))
        except ValueError as e:
            if "Expected 2D array, got 1D array instead" in str(e):
                raise ValueError(_ERR_MSG_1DCOLUMN) from e
            else:
                raise

    def fit(self, X, y=None, **fit_params):
        """Fit all transformers using X.
        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.
        y : array-like of shape (n_samples,...), default=None
            Targets for supervised learning.
        Returns
        -------
        self : ColumnTransformer
            This estimator
        """
        # we use fit_transform to make sure to set sparse_output_ (for which we
        # need the transformed data) to have consistent output type in predict
        self.fit_transform(X, y=y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.
        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.
        y : array-like of shape (n_samples,), default=None
            Targets for supervised learning.
        Returns
        -------
        X_t : {array-like, sparse matrix} of \
                shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.
        """
        # TODO: this should be `feature_names_in_` when we start having it
        if hasattr(X, "columns"):
            self._feature_names_in = np.asarray(X.columns)
        else:
            self._feature_names_in = None
        X = _check_X(X)
        # set n_features_in_ attribute
        self._check_n_features(X, reset=True)
        self._validate_transformers()
        self._validate_column_callables(X)
        self._validate_remainder(X)

        result = self._fit_transform(X, y, _fit_transform_one, **fit_params)

        if not result:
            self._update_fitted_transformers([])
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*result)

        # determine if concatenated output will be sparse or not
        if any(sparse.issparse(X) for X in Xs):
            nnz = sum(X.nnz if sparse.issparse(X) else X.size for X in Xs)
            total = sum(X.shape[0] * X.shape[1] if sparse.issparse(X)
                        else X.size for X in Xs)
            density = nnz / total
            self.sparse_output_ = density < self.sparse_threshold
        else:
            self.sparse_output_ = False

        self._update_fitted_transformers(transformers)
        self._validate_output(Xs)

        return self._hstack(list(Xs))

    def _check_fit_params(self, fitted=False, **fit_params):
        fit_params_steps = {
            name: {}
            for name, trans, column, weight
            in self._iter(fitted=fitted, replace_strings=True)
            if trans is not None
        }

        valid_steps = {n for n, _, _ in self.transformers}
        print(fit_params)
        for pname, pval in fit_params.items():
            if '__' not in pname:
                raise ValueError(
                    "MyColumnTransformer.fit does not accept the {} parameter."
                    "You can pass parameters to specific steps of your "
                    "pipeline using the stepname__parameter format, e.g. "
                    "`Pipeline.fit(X, y, logisticregression__sample_weight"
                    "=sample_weight)`.".format(pname))
            step, param = pname.split('__', 1)

            try:
                fit_params_steps[step][param] = pval
            except KeyError as e:
                if step not in valid_steps:
                    raise e
        return fit_params_steps
