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
