#!/usr/bin/env python3

import warnings
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy.typing as npt


def pearsons_correlation(
    x: "npt.ArrayLike",
    y: "npt.ArrayLike"
) -> float:
    from scipy.stats import pearsonr
    from scipy.stats import (
        PearsonRNearConstantInputWarning,
        PearsonRConstantInputWarning
    )

    x_ = np.array(x)
    y_ = np.array(y)

    if len(x_.shape) == 1:
        x_ = x_.reshape(-1, 1)

    if len(y_.shape) == 1:
        y_ = y_.reshape(-1, 1)

    assert x_.shape == y_.shape

    out = np.zeros(x_.shape[1])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=PearsonRNearConstantInputWarning,
        )
        warnings.filterwarnings(
            "ignore",
            category=PearsonRConstantInputWarning,
        )
        for i in range(x_.shape[1]):
            mask = ~(np.isnan(y_[:, i]) | np.isnan(x_[:, i]))
            if mask.all():
                cor = np.nan
            elif (
                (y_[0, i] == y_[:, i]).all()
                or (x_[0, i] == x_[:, i]).all()
            ):
                cor = 0.0
            else:
                cor, _ = pearsonr(x_[mask, i], y_[mask, i])
            out[i] = cor

    if np.all(np.isnan(out)):
        return np.nan
    else:
        return np.nanmean(out)


def spearmans_correlation(
    x: "npt.ArrayLike",
    y: "npt.ArrayLike"
) -> float:
    from scipy.stats import spearmanr
    from scipy.stats import SpearmanRConstantInputWarning

    x_ = np.array(x)
    y_ = np.array(y)

    if len(x_.shape) == 1:
        x_ = x_.reshape(-1, 1)

    if len(y_.shape) == 1:
        y_ = y_.reshape(-1, 1)

    assert x_.shape == y_.shape

    out = np.zeros(x_.shape[1])

    # Note that spearmanr does accept multi-column
    # but it will then output a matrix of pairwise correlations..
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=SpearmanRConstantInputWarning
        )
        for i in range(x_.shape[1]):
            mask = ~(np.isnan(y_[:, i]) | np.isnan(x_[:, i]))
            if mask.all():
                cor = np.nan
            elif (
                (y_[0, i] == y_[:, i]).all()
                or (x_[0, i] == x_[:, i]).all()
            ):
                cor = 0.0
            else:
                cor, _ = spearmanr(x_[mask, i], y_[mask, i])
            out[i] = cor

    if np.all(np.isnan(out)):
        return np.nan
    else:
        return np.nanmean(out)


def tau_correlation(
    x: "npt.ArrayLike",
    y: "npt.ArrayLike",
    variant: str = "b"
) -> float:
    from scipy.stats import kendalltau

    x_ = np.array(x)
    y_ = np.array(y)

    if len(x_.shape) == 1:
        x_ = x_.reshape(-1, 1)

    if len(y_.shape) == 1:
        y_ = y_.reshape(-1, 1)

    assert x_.shape == y_.shape

    out = np.zeros(x_.shape[1])

    # Note that spearmanr does accept multi-column
    # but it will then output a matrix of pairwise correlations..
    for i in range(x_.shape[1]):
        mask = ~(np.isnan(y_[:, i]) | np.isnan(x_[:, i]))
        if mask.all():
            cor = np.nan
        elif (
            (y_[0, i] == y_[:, i]).all()
            or (x_[0, i] == x_[:, i]).all()
        ):
            cor = 0.0
        else:
            cor, _ = kendalltau(x_[mask, i], y_[mask, i])
        out[i] = cor

    if np.all(np.isnan(out)):
        return np.nan
    else:
        return np.nanmean(out)
