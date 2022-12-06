import sys
import argparse
import json
import contextlib
import warnings
import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # from selectml.optimiser.optimise import BaseTypes
    from typing import Optional, List, Tuple
    from typing import Callable
    TRACKER_TYPE = List[pd.Series]

from selectml.data import generate_sample_data

def cli(parser: argparse.ArgumentParser) -> None:

    parser.add_argument(
        "-n", "--ngenotypes",
        type=int,
        default=100,
        help="How many distinct genotypes to simulate."
    )

    parser.add_argument(
        "-m", "--nmarkers",
        type=int,
        default=100,
        help="How many markers"
    )

    parser.add_argument(
        "-g", "--ngroups",
        type=int,
        default=0,
        help=(
            "How many grouping factors to simulate."
        )
    )

    parser.add_argument(
        "-c", "--ncovariates",
        type=int,
        default=0,
        help=(
            "How many continuous covariates to simulate."
        )
    )

    parser.add_argument(
        "--covariate-stdev",
        type=float,
        default=None,
        help=(
            "Set the standard deviation of the covariates."
        )
    )

    parser.add_argument(
        "-p", "--ploidy",
        type=int,
        default=2,
        help=(
            "The ploidy to use. Haploid (1) will choose alleles [0, 1]. "
            "Diploid (2) will choose alleles as [0, 1, 2]."
        )
    )

    parser.add_argument(
        "-o", "--outprefix",
        type=str,
        default="test_data",
        help="The prefix to use to write simulated data."
    )

    parser.add_argument(
        "-j", "--out-json",
        type=argparse.FileType("w"),
        default=None,
        help="Write out the generated results as a JSON dict as well. This will contain the true effect sizes."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The random seed to use."
    )
    return

def runner(args: argparse.Namespace) -> None:
    sample_data = generate_sample_data(
        n=args.ngenotypes,
        nmarkers=args.nmarkers,
        ngroups=args.ngroups,
        ncovariates=args.ncovariates,
        covariate_std=args.covariate_stdev,
        ploidy=args.ploidy,
        seed=args.seed
    )

    if args.out_json is not None:
        json.dump(sample_data, args.out_json)

    markers = pd.DataFrame(sample_data["markers"])
    markers["name"] = sample_data["indivs"]
    markers = markers[["name"] + list(markers.columns[:-1])]

    colnames = ["name", "response"]
    responses = [
        pd.DataFrame(sample_data["indivs"].reshape((-1, 1)), columns=["name"]),
        pd.DataFrame(sample_data["y"], columns=["response"])
    ]

    if (sample_data.get("groups", None) is not None) and (args.ngroups > 0):
        groups = sample_data["groups"]
        responses.append(
            pd.DataFrame(groups, columns=[f"G{i}" for i in range(groups.shape[1])])
        )

    if (sample_data.get("covs", None) is not None) and (args.ncovariates > 0):
        covariates = sample_data["covs"]
        responses.append(
            pd.DataFrame(covariates, columns=[f"C{i}" for i in range(covariates.shape[1])])
        )

    response = pd.concat(responses, axis=1)

    markers.to_csv(
        f"{args.outprefix}-markers.tsv",
        sep="\t",
        index=False
    )

    response.to_csv(
        f"{args.outprefix}-experiment.tsv",
        sep="\t",
        index=False,
    )
    return
