#!/usr/bin/env python3

import sys
import argparse

from .model_enum import ModelOptimiser


def cli(parser: argparse.ArgumentParser) -> None:

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
        "-r", "--response",
        type=str,
        help="The column to use from experiment as the y value"
    )

    parser.add_argument(
        "-g", "--groups",
        type=str,
        nargs="+",
        default=[],
        help=(
            "The column(s) in the experiment table to use for grouping "
            "factors (e.g. different environments) that should be included."
        )
    )

    parser.add_argument(
        "-i", "--individuals",
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
        "--importances",
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
        "-n", "--ntrials",
        type=int,
        default=200,
        help="The number of iterations to try for optimisation."
    )

    parser.add_argument(
        "-c", "--cpu",
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


def runner(args: argparse.Namespace) -> None:
    import json
    import numpy as np
    import pandas as pd
    import optuna

    optuna.logging.set_verbosity(optuna.logging.INFO)
    optuna.logging.enable_default_handler()
    model_cls = args.model.get_model()
    markers = pd.read_csv(args.markers, sep="\t")
    exp = pd.read_csv(args.experiment, sep="\t")

    if args.seed is not None:
        np.random.set_state(args.seed)

    model = model_cls(
        experiment=exp,
        markers=markers,
        response_columns=args.response,
        grouping_columns=args.groups,
        individual_columns=args.individuals,
        seed=args.seed
    )

    study = optuna.create_study(direction="minimize")
    for trial in model.starting_points():
        study.enqueue_trial(trial)

    # Timeout after 6 hours
    study.optimize(model, timeout=6 * 60 * 60, n_trials=args.ntrials, n_jobs=args.cpu, gc_after_trial=True, catch=(MemoryError, OSError, ValueError, KeyError))

    trial_df = study.trials_dataframe()
    trial_df.to_csv(args.outfile, sep="\t")

    if args.best is not None:
        best = study.best_params
        json.dump(best, args.best)

    if args.importance is not None:
        importance = optuna.importance.get_param_importances(study)
        json.dump(importance, args.importance)

    return
