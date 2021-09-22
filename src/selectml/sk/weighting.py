#!/usr/bin/env python3

import pandas as pd
import numpy as np


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Union, Optional
    from typing import List, Tuple
    from typing import Callable
    import numpy.typing as npt

    """ Function interface for weighting schemes """
    WeightingInterface = Callable[
        [
            "npt.ArrayLike",
            "npt.ArrayLike",
            "Optional[npt.ArrayLike]",
            bool
        ],
        np.ndarray
    ]


def variance_weights(
    X: "npt.ArrayLike",
    y: "npt.ArrayLike",
    grouping: "Optional[npt.ArrayLike]" = None,
    means: bool = False
) -> np.ndarray:
    """ Compute weights by reciprocal variance of the y-values within
    groupings.

    This function ignores information in the X-values.
    Examples:

    >>> import numpy as np
    >>> from selectml.sk.weighting import variance_weights
    >>> from selectml.data import basic
    >>> X, y, indivs = basic()
    >>> variance_weights(X, y, indivs)
    array([0.44702982, 0.44702982, 0.44702982, 0.44702982, 0.44702982,
           0.11108895, 0.11108895, 0.11108895, 0.11108895, 0.11108895,
           0.57986478, 0.57986478, 0.57986478, 0.57986478, 0.57986478,
           0.11462054, 0.11462054, 0.11462054, 0.11462054, 0.11462054,
           0.3161391 , 0.3161391 , 0.3161391 , 0.3161391 , 0.3161391 ])
    """

    y_ = np.array(y)

    if grouping is None:
        grouping_ = np.arange(y_.shape[0])
    else:
        grouping_ = np.array(grouping)

    df = pd.DataFrame(y_)
    ycols = [f"y{i}" for i in range(len(df.columns))]
    df.columns = ycols

    df["groups"] = grouping_
    x = df.groupby("groups")[ycols].var().mean(axis=1)

    if not means:
        x = x.loc[df["groups"]]

    counts = df.groupby("groups")[ycols[0]].count()
    if not means:
        counts = counts.loc[df["groups"]]

    out = 1 / (x * counts)
    return out.values


def distance_weights(
    X: "npt.ArrayLike",
    y: "npt.ArrayLike",
    grouping: "Optional[npt.ArrayLike]" = None,
    means: bool = False
) -> np.ndarray:
    """ Compute weights based on Manhattan distance of the X-values.

    This function ignores information in the y-values.
    Examples:

    >>> import numpy as np
    >>> from selectml.sk.weighting import distance_weights
    >>> from selectml.data import basic
    >>> X, y, indivs = basic()
    >>> distance_weights(X, y, indivs)
    array([41., 41., 41., 41., 41., 40., 40., 40., 40., 40., 40., 40., 40.,
           40., 40., 36., 36., 36., 36., 36., 35., 35., 35., 35., 35.])
    """
    from scipy.spatial.distance import pdist, squareform

    X_ = np.array(X)

    if grouping is None:
        grouping_ = np.arange(X_.shape[0])
    else:
        grouping_ = np.array(grouping)

    x = pd.DataFrame({
        "index": np.arange(X_.shape[0]),
        "genotypes": np.apply_along_axis(
            lambda z: "".join(str(z_i) for z_i in z), 1, X)
    })
    firsts = pd.DataFrame(X_).groupby(x["genotypes"]).first()
    groups = (
        x
        .groupby("genotypes")["index"]
        .unique()
        .apply(pd.Series)
        .unstack()
        .reset_index(level=0, drop=True)
        .reset_index()
        .rename(columns={0: "index"})
    )

    dist = squareform(pdist(firsts.values, "cityblock"))
    np.fill_diagonal(dist, 0)

    corr = pd.DataFrame(dist, index=firsts.index.values)
    corr = pd.merge(
        groups,
        corr,
        left_on="genotypes",
        right_index=True
    ).drop(
        columns="genotypes"
    ).set_index("index", drop=True)

    corr = corr.loc[np.arange(X_.shape[0])].sum(axis=1)

    if not means:
        return corr.values

    # Pandas can't do multiindex selection, so just cat groups all together.
    corr = corr.groupby(grouping_).mean()
    return corr.values


def cluster_weights(
    X: "npt.ArrayLike",
    y: "npt.ArrayLike",
    grouping: "Optional[npt.ArrayLike]" = None,
    means: bool = False
) -> np.ndarray:
    """ Compute clusters on the X values based on Manhattan distance, then
    weight by cluster size.

    This function ignores information in the y-values.

    Examples:

    >>> import numpy as np
    >>> from selectml.sk.weighting import cluster_weights
    >>> from selectml.data import basic
    >>> X, y, indivs = basic()
    >>> cluster_weights(X, y, indivs)
    array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
           4., 4., 4., 4., 4., 4., 4., 4.])
    """

    from fastcluster import average
    from scipy.cluster.hierarchy import cut_tree
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist, squareform

    X_ = np.array(X)

    if grouping is None:
        grouping_ = np.arange(X_.shape[0])
    else:
        grouping_ = np.array(grouping)

    x = pd.DataFrame({
        "index": np.arange(X_.shape[0]),
        "genotypes": np.apply_along_axis(
            lambda z: "".join(str(z_i) for z_i in z), 1, X_)
    })
    firsts = pd.DataFrame(X_).groupby(x["genotypes"]).first()
    groups = (
        x
        .groupby("genotypes")["index"]
        .unique()
        .apply(pd.Series)
        .unstack()
        .reset_index(level=0, drop=True)
        .reset_index()
        .rename(columns={0: "index"})
    )

    dist = pdist(firsts.values, "cityblock")
    hier = average(dist)
    coph = squareform(cophenet(hier))

    height = np.percentile(coph[coph > 0], 0.5)
    clusters = pd.DataFrame({
        "genotypes": firsts.index.values,
        "clusters": cut_tree(hier, height=height)[:, 0]
    })
    clusters = (
        pd.merge(groups, clusters, left_on="genotypes", right_on="genotypes")
        .drop(columns="genotypes")
    )

    cluster_counts = (
        clusters.groupby("clusters").count()["index"]
        .apply(lambda x: (clusters.shape[0] - x) / x)
        .reset_index()
        .rename(columns={"index": "weight"})
    )

    clusters = pd.merge(
        clusters,
        cluster_counts,
        on="clusters"
    ).set_index("index")
    clusters = clusters.loc[np.arange(X_.shape[0]), "weight"]

    if not means:
        return clusters.values

    clusters = clusters.groupby(grouping_).mean()
    return clusters.values
