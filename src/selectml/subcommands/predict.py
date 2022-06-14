#!/usr/bin/env python3

import sys
import argparse
import contextlib

import numpy as np
import pandas as pd

from .model_enum import ModelOptimiser, Stats
from ..optimiser.cv import Dataset

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # from selectml.optimiser.optimise import BaseTypes
    from typing import Optional
    from typing import List


def cli(parser: argparse.ArgumentParser) -> None:

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
        "params",
        type=argparse.FileType('r'),
        help="The parameters to model with."
    )

    parser.add_argument(
        "markers",
        type=argparse.FileType('r'),
        help="The marker tsv file to parse as input."
    )

    parser.add_argument(
        "experiment",
        type=argparse.FileType('r'),
        help="The experimental data tsv file."
    )

    parser.add_argument(
        "-t", "--train-col",
        type=str,
        help="The column to use from experiment to set the training values"
    )

    parser.add_argument(
        "-r", "--response-col",
        type=str,
        help="The column to use from experiment as the y value"
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
        "-n", "--name-col",
        type=str,
        default=None,
        help=(
            "The column in experiment that indicates which individual it is."
        )
    )

    parser.add_argument(
        "-o", "--outfile",
        type=argparse.FileType('w'),
        default=sys.stdout,
        help="Where to write the output to. Default: stdout"
    )

    parser.add_argument(
        "-s", "--stats",
        type=argparse.FileType('w'),
        default=None,
        help="Where to write the evaluation stats to."
    )

    parser.add_argument(
        "--outmodel",
        type=argparse.FileType('wb'),
        default=None,
        help="Where to write the model",
    )

    parser.add_argument(
        "--cpu",
        type=int,
        default=-1,
        help="The number CPUs to use."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The random seed to use."
    )

    return


class DummyContext(object):

    @contextlib.contextmanager
    def scope(self):
        try:
            yield None
        finally:
            pass
        return None


def check_tf_session(
    n_threads: "Optional[int]" = None,
):
    import tensorflow as tf
    if len(tf.config.list_physical_devices('GPU')) > 0:
        return tf.distribute.MultiWorkerMirroredStrategy()
    else:
        tf.config.threading.set_inter_op_parallelism_threads(n_threads)
        tf.config.threading.set_intra_op_parallelism_threads(n_threads)
        return tf.distribute.get_strategy()


def get_data(
    data: pd.DataFrame,
    group_cols: "Optional[List[str]]",
    covariate_cols: "Optional[List[str]]",
    name_col: str,
    response_col: str,
    marker_cols: "List[str]"
) -> "Dataset":

    if group_cols is not None:
        groups: "Optional[np.ndarray]" = (
            data
            .loc[:, group_cols]
            .values
            .astype(float)
        )
    else:
        groups = None

    if covariate_cols is not None:
        covariates: "Optional[np.ndarray]" = (
            data
            .loc[:, covariate_cols]
            .values
            .astype(float)
        )
    else:
        covariates = None

    markers = (
        data
        .loc[:, marker_cols]
        .values
        .astype(float)
    )

    y = data.loc[:, [response_col]].values.astype(float)
    ds = Dataset(
        markers,
        y,
        groups,
        covariates
    )
    return ds


def runner(args: argparse.Namespace) -> None:
    import json
    import pandas as pd

    model_cls = args.model.get_model()
    task = str(args.task)
    markers = pd.read_csv(args.markers, sep="\t")
    exp = pd.read_csv(args.experiment, sep="\t")
    data = pd.merge(exp, markers, on=args.name_col, how="inner")
    data_train = data.loc[data.loc[:, args.train_col] == "train", ]
    marker_cols = markers.columns.difference([args.name_col])

    ds = get_data(
        data_train,
        args.group_cols,
        args.covariate_cols,
        args.name_col,
        args.response_col,
        marker_cols,
    )

    if args.seed is not None:
        np.random.set_state(args.seed)

    if args.model == ModelOptimiser.tf:
        strategy = check_tf_session(args.cpu)
    else:
        strategy = DummyContext()

    params = json.load(args.params)
    with strategy.scope():
        optimiser = model_cls(
            task=task,
            ploidy=2,
            seed=args.seed
        )
        model = optimiser.model(params, ds)
        optimiser.fit(model, ds)

    del data_train
    del ds

    ds = get_data(
        data,
        args.group_cols,
        args.covariate_cols,
        args.name_col,
        args.response_col,
        marker_cols,
    )

    with strategy.scope():
        preds = optimiser.predict(model, ds)

    data["predictions"] = preds

    selection = data[[
        c for c in data.columns if c not in marker_cols
    ]].drop_duplicates()

    selection.to_csv(args.outfile, sep="\t", index=False)

    if args.stats is not None:
        stats = args.task.get_stats()
        gb_cols = [args.train_col]

        if "generation" in selection.columns:
            gb_cols.append("generation")

        if "population" in selection.population:
            gb_cols.append("populations")

        stat_results = (
            selection
            .groupby(gb_cols)
            [[args.response_col, "predictions"]]
            .apply(lambda x: pd.Series(stats(x.iloc[:, 0], x.iloc[:, 1])))
        )
        stat_results.to_csv(args.stats, sep="\t", index=True)

    if args.outmodel is not None:
        import joblib
        joblib.dump(model, args.outmodel)
    return
