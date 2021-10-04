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
        "params",
        type=argparse.FileType('r'),
        help="The parameters to model with."
    )

    parser.add_argument(
        "train_markers",
        type=argparse.FileType('r'),
        help="The marker tsv file to parse as input."
    )

    parser.add_argument(
        "train_experiment",
        type=argparse.FileType('r'),
        help="The experimental data tsv file."
    )

    parser.add_argument(
        "test_markers",
        type=argparse.FileType('r'),
        help="The marker tsv file to parse as input."
    )

    parser.add_argument(
        "test_experiment",
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
        "-c", "--cpu",
        type=int,
        default=-1,
        help="The number CPUs to use."
    )

    return


def runner(args: argparse.Namespace) -> None:
    import json
    import pandas as pd

    model_cls = args.model.get_model()
    train_markers = pd.read_csv(args.train_markers, sep="\t")
    train_exp = pd.read_csv(args.train_experiment, sep="\t")
    model = model_cls(
        experiment=train_exp,
        markers=train_markers,
        response_columns=args.response,
        grouping_columns=args.groups,
        individual_columns=args.individuals,
    )

    params = json.load(args.params)
    trained = model.train_from_params(params)

    test_markers = pd.read_csv(args.test_markers, sep="\t")
    test_exp = pd.read_csv(args.test_experiment, sep="\t")

    data = pd.merge(
        test_exp,
        test_markers,
        on=model.individual_columns,
        how="inner",
    )

    X = data.loc[:, model.grouping_columns + model.marker_columns]
    preds = model.predict(trained, X)

    data["predictions"] = preds

    selection = data[[
        c for c in data.columns if c not in model.marker_columns
    ]].drop_duplicates()

    selection.to_csv(args.outfile, sep="\t", index=False)
    return
