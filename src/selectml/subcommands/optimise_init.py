#!/usr/bin/env python3

import sys
import argparse
import json
import contextlib
import warnings
import random

from os.path import basename, splitext

import numpy as np
import pandas as pd
import optuna

from rpy2.rinterface_lib.embedded import RRuntimeError

from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from ..optimiser.cv import gen_splits, CVData
from ..optimiser.runners import BaseRunner
from ..optimiser.stats import OptimiseStats
from .model_enum import ModelOptimiser, Stats
from ..study import create_study

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # from selectml.optimiser.optimise import BaseTypes
    from typing import Optional, List, Tuple
    from typing import Callable
    from typing import TextIO
    from typing import Literal
    TRACKER_TYPE = List[pd.Series]


def cli(parser: argparse.ArgumentParser) -> None:

    parser.add_argument(
        "database",
        type=str,
        help="The database URI."
    )

    parser.add_argument(
        "task",
        type=Stats.from_string,
        choices=list(Stats),
        help="The type of modelling task to optimise."
    )

    parser.add_argument(
        "model",
        type=ModelOptimiser.from_string,
        choices=list(ModelOptimiser),
        help="The model type to optimise."
    )

    parser.add_argument(
        "markers",
        type=str,
        help="The marker tsv file to parse as input."
    )

    parser.add_argument(
        "experiment",
        type=str,
        help="The experimental data tsv file."
    )

    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name the study this instead of the default based on parameters."
    )

    parser.add_argument(
        "-r", "--response-col",
        type=str,
        help="The column to use from experiment as the y value"
    )

    parser.add_argument(
        "-n", "--name-col",
        type=str,
        default="name",
        help="The column to for names to align experiment and marker tables."
    )

    parser.add_argument(
        "-g", "--group-cols",
        type=str,
        nargs="+",
        default=None,
        help=(
            "The column(s) in the experiment table to use for grouping "
            "factors (e.g. different environments) that should be included."
        )
    )

    parser.add_argument(
        "-c", "--covariate-cols",
        type=str,
        nargs="+",
        default=None,
        help=(
            "The column(s) in experiment to use as covariates."
        )
    )

    parser.add_argument(
        "--ntrials",
        type=int,
        default=200,
        help="The number of iterations to try for optimisation."
    )

    parser.add_argument(
        "--cv-k",
        type=int,
        default=5,
        help="The number of cross validation folds to use."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The random seed to use."
    )

    return


def prep_data(
    markers: TextIO,
    experiment: TextIO,
    name_col: str,
    group_cols: list[str],
    covariate_cols: list[str],
    response_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame]:
    markers_ = pd.read_csv(markers, sep="\t")
    exp = pd.read_csv(experiment, sep="\t")
    data = pd.merge(exp, markers_, on=name_col, how="inner")

    if isinstance(group_cols, list) and len(group_cols) == 0:
        group_cols_ = None
    else:
        group_cols_ = group_cols

    if group_cols_ is not None:
        groups: "Optional[np.ndarray]" = (
            data
            .loc[:, group_cols_]
            .astype(float)
        )
    else:
        groups = None

    if isinstance(covariate_cols, list) and len(covariate_cols) == 0:
        covariate_cols_ = None
    else:
        covariate_cols_ = group_cols

    if (covariate_cols_ is not None):
        covariates: "Optional[np.ndarray]" = (
            data
            .loc[:, covariate_cols_]
            .astype(float)
        )
    else:
        covariates = None

    markers_ = (
        data
        .loc[:, markers_.columns.difference([name_col])]
        .astype(float)
    )

    y = data.loc[:, [response_col]].astype(float)
    return data.loc[:, [name_col]], markers_, groups, covariates, y


def gen_study_name(
    model: str,
    task: str,
    markers: str,
    experiment: str,
    response_col: str,
    group_cols: list[str],
    covariate_cols: list[str],
):
    marker_bname, _ = splitext(basename(markers))
    experiment_bname, _ = splitext(basename(experiment))

    out = [marker_bname, experiment_bname, model, task, response_col]
    if len(group_cols) > 0:
        out.extend("_".join(group_cols))

    if len(covariate_cols) > 0:
        out.extend("_".join(covariate_cols))

    name = "_".join(out)

    # Max table name length is 128
    # Reserve 4 characters for -mar and -exp data tables
    # And 1 spare, just because
    if len(name) > (128 - 5):
        raise ValueError((
            f"The generated table name ({name}) given your inputs is too long. "
            "SQL won't be able to deal with it. "
            "Please provide an argument to --study-name, which should "
            f"be at most {128 - 5} characters long."
        ))

    return name


def runner(args: argparse.Namespace) -> None:
    model = str(args.model)
    task = str(args.task)

    if args.group_cols is None:
        group_cols = []
    else:
        group_cols = args.group_cols

    if args.covariate_cols is None:
        covariate_cols = []
    else:
        covariate_cols = args.covariate_cols

    if args.study_name is not None:
        study_name = args.study_name
    else:
        study_name = gen_study_name(
            model,
            task,
            args.markers,
            args.experimental,
            args.response_col,
            args.group_cols,
            args.covariate_cols,
        )

    if task in ("regression", "classification"):
        direction: Literal["minimize", "maximize"] = "minimize"
    else:
        direction = "maximize"

    create_study(
        args.storage_url,
        study_name,
        direction,
        load_if_exists=False,
    )

    with open(args.markers, "r") as mhandle, open(args.experiment, "r") as ehandle:
        indivs, markers, groups, covariates, y = prep_data(
            mhandle,
            ehandle,
            args.name_col,
            group_cols,
            covariate_cols,
            args.response_col
        )

    if args.seed is None:
        seed = random.randint(0, 2**16)
    else:
        seed = args.seed

    cv = pd.DataFrame({"cv": gen_splits(markers, args.cv_k, seed)})

    indivs.to_sql(f"{study_name}-ind", args.storage_url)
    markers.to_sql(f"{study_name}-mar", args.storage_url)

    y.to_sql(f"{study_name}-y", args.storage_url)
    cv.to_sql(f"{study_name}-cv", args.storage_url)

    if groups is not None:
        groups.to_sql(f"{study_name}-grp", args.storage_url)

    if covariates is not None:
        covariates.to_sql(f"{study_name}-cov", args.storage_url)

    params = pd.DataFrame(
        {"k": [args.cv_k], "model": [model], "task": [task], "ntrials": [args.ntrials]},
        index=["params"]
    )
    params.to_sql(f"{study_name}-par", args.storage_url)
    return
