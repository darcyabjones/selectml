#!/usr/bin/env python3

import sys
import argparse


def cli(parser: argparse.ArgumentParser) -> None:

    parser.add_argument(
        "predictions",
        type=argparse.FileType('r'),
        help="The predictions tsv file."
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
        "-o", "--outfile",
        type=argparse.FileType('w'),
        default=sys.stdout,
        help="Where to write the output to. Default: stdout"
    )

    return


def runner(args: argparse.Namespace) -> None:
    import json
    import pandas as pd

    return
