from importlib.resources import path

import numpy as np


def basic():
    from numpy.lib import recfunctions as rfn

    with path(__name__, "basic.tsv") as handler:
        arr = np.recfromcsv(handler, delimiter="\t")

    X = (
        rfn.structured_to_unstructured(arr[[f"m{i}" for i in range(10)]])
        .astype(float)
    )

    y = arr["y"].astype(float)
    indivs = arr["individual"]

    return X, y, indivs


def generate_sample_data(
    n,
    nmarkers,
    ngroups,
    ncovariates,
    covariate_std=None,
    seed=None
):

    rng = np.random.default_rng(seed)
    marker_effects = rng.normal(scale=1/nmarkers, size=(1, nmarkers))
    markers = rng.choice([0, 1, 2], size=(n, nmarkers))

    y = markers.dot(marker_effects.T)

    indivs = np.arange(n)

    out = dict(
        markers=markers,
        marker_effects=marker_effects,
        indivs=indivs,
        y=y,
    )

    if ngroups > 0:
        indivs = np.repeat(indivs, ngroups)
        markers = np.repeat(markers, ngroups, axis=0)
        y = markers.dot(marker_effects.T)

        group_effects = rng.normal(size=(1, ngroups))
        groups_ = np.tile(np.arange(ngroups), n)

        n = n * ngroups
        groups = np.zeros((n, ngroups))
        groups[np.arange(n), groups_] = 1
        del groups_

        y += groups.dot(group_effects.T)
        out.update(dict(
            markers=markers,
            marker_effects=marker_effects,
            indivs=indivs,
            groups=groups,
            group_effects=group_effects,
            y=y,
        ))

    if ncovariates > 0:
        if covariate_std is None:
            covariate_std = 1 / ncovariates

        cov_effects = rng.normal(scale=covariate_std, size=(1, ncovariates))
        covs = rng.normal(size=(n, ncovariates))

        y += covs.dot(cov_effects.T)

        out.update(dict(
            covs=covs,
            cov_effects=cov_effects,
            y=y,
        ))

    return out
