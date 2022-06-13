#!/usr/bin/env python3

import sys
import argparse
import json
import contextlib
import warnings

import numpy as np
import pandas as pd
import optuna

from rpy2.rinterface_lib.embedded import RRuntimeError

from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from ..optimiser.cv import CVData
from ..optimiser.runners import BaseRunner
from ..optimiser.stats import OptimiseStats
from .model_enum import ModelOptimiser, Stats

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # from selectml.optimiser.optimise import BaseTypes
    from typing import Optional, List, Tuple
    from typing import Callable
    TRACKER_TYPE = List[pd.Series]


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
        "-o", "--outfile",
        type=argparse.FileType('w'),
        default=sys.stdout,
        help="Where to write the output to. Default: stdout"
    )

    parser.add_argument(
        "--full",
        type=argparse.FileType('w'),
        default=None,
        help="Write the full results of CVs"
    )

    parser.add_argument(
        "--continue",
        dest="continue_",
        type=argparse.FileType("rb"),
        default=None,
        help="Where to continue from."
    )

    parser.add_argument(
        "--pickle",
        type=argparse.FileType("wb"),
        default=None,
        help="Where to save the trials to."
    )

    parser.add_argument(
        "--importance",
        type=argparse.FileType('w'),
        default=None,
        help="Where to write the output to. Default: stdout"
    )

    parser.add_argument(
        "-b", "--best",
        type=argparse.FileType("w"),
        default=None,
        help="Write out the best parameters in JSON format"
    )

    parser.add_argument(
        "--ntrials",
        type=int,
        default=200,
        help="The number of iterations to try for optimisation."
    )

    parser.add_argument(
        "--cpu",
        type=int,
        default=1,
        help="The number CPUs to use."
    )

    parser.add_argument(
        "--ntasks",
        type=int,
        default=1,
        help="The number of optuna tasks to use."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The random seed to use."
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=6.0,
        help="The maximum time in hours to run for."
    )
    return


def setup_optimise(
    cv: "CVData",
    model: "BaseRunner",
    stats: "OptimiseStats",
    tracker: "Optional[TRACKER_TYPE]" = None
) -> "Tuple[Callable[[optuna.Trial], float], TRACKER_TYPE]":
    import json

    if tracker is None:
        tracker_: TRACKER_TYPE = []
    else:
        tracker_ = list(tracker)

    data = [(tr, te) for tr, te in cv()]
    train_data = [tr for tr, _ in data]
    test_data = [te for _, te in data]

    def inner(trial):
        params, models, _, _ = model.sample(trial, train_data)

        results = []
        for m, test in zip(models, test_data):
            yhat = model.predict(m, test)
            results.append(stats(test.y, yhat))

        eval_metric = np.nanmean(np.array([
            r.get(stats.target_metric, np.nan)
            for r
            in results
        ]))

        for result in results:
            result["params"] = json.dumps(params)
            tracker_.append(pd.Series(result))
        return eval_metric

    return inner, tracker_


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


def runner(args: argparse.Namespace) -> None:

    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.enable_default_handler()
    model_cls = args.model.get_model()
    task = str(args.task)
    stats = args.task.get_stats()
    markers = pd.read_csv(args.markers, sep="\t")

    exp = pd.read_csv(args.experiment, sep="\t")

    if args.model == ModelOptimiser.tf:
        strategy = check_tf_session(args.cpu)
    else:
        strategy = DummyContext()

    data = pd.merge(exp, markers, on=args.name_col, how="inner")

    if isinstance(args.group_cols, list) and len(args.group_cols) == 0:
        args.group_cols = None

    if args.group_cols is not None:
        groups: "Optional[np.ndarray]" = (
            data
            .loc[:, args.group_cols]
            .values
            .astype(float)
        )
    else:
        groups = None

    if isinstance(args.covariate_cols, list) and len(args.covariate_cols) == 0:
        args.covariate_cols = None

    if (args.covariate_cols is not None):
        covariates: "Optional[np.ndarray]" = (
            data
            .loc[:, args.covariate_cols]
            .values
            .astype(float)
        )
    else:
        covariates = None

    markers = (
        data
        .loc[:, markers.columns.difference([args.name_col])]
        .values
        .astype(float)
    )

    y = data.loc[:, [args.response_col]].values.astype(float)

    if args.seed is not None:
        np.random.set_state(args.seed)

    cv = CVData(
        markers=markers,
        groups=groups,
        covariates=covariates,
        y=y,
    )

    model = model_cls(task=task, ploidy=2)

    optimiser, tracker = setup_optimise(
        cv=cv,
        model=model,
        stats=stats,
    )

    if task == "regression":
        direction = "minimize"
    else:
        direction = "maximize"

    study = optuna.create_study(
        direction=direction,
        load_if_exists=True,
    )

    if args.continue_ is not None:
        import pickle
        existing_trials = pickle.load(args.continue_)
        study.add_trials(existing_trials)
    else:
        pass
        # Expect that these have already been run.
        # for trial in model.starting_points():
        #     study.enqueue_trial(trial)

    warnings.filterwarnings("ignore")

    try:
        # Timeout after 6 hours
        with strategy.scope():
            study.optimize(
                optimiser,
                timeout=round(args.timeout) * 60 * 60,
                callbacks=[MaxTrialsCallback(args.ntrials, states=(TrialState.COMPLETE,))],
                n_jobs=args.ntasks,
                gc_after_trial=True,
                catch=(MemoryError, OSError, ValueError, KeyError, np.linalg.LinAlgError, RRuntimeError)
            )
    finally:
        if args.pickle is not None:
            import pickle
            pickle.dump(study.trials, args.pickle)

        trial_df = study.trials_dataframe()
        trial_df.to_csv(args.outfile, sep="\t", index=False)

        if args.best is not None:
            best = study.best_params
            best["model"] = str(args.model)
            best["task"] = str(args.task)
            json.dump(best, args.best)

        if args.importance is not None:
            importance = optuna.importance.get_param_importances(study)
            json.dump(importance, args.importance)

    return
