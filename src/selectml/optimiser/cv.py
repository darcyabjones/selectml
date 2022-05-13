import random

from typing import NamedTuple

import numpy as np

from sklearn.model_selection import KFold

from typing import cast
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from typing import Tuple
    from typing import Iterator
    from typing import Dict
    import numpy.typing as npt

from selectml.higher import fmap


class Dataset(NamedTuple):

    markers: "np.ndarray"
    y: "np.ndarray"
    groups: "Optional[np.ndarray]"
    covariates: "Optional[np.ndarray]"

    def keys(self) -> "Iterator[str]":
        for k in ["markers", "y", "groups", "covariates"]:
            yield k
        return

    def values(self) -> "Iterator[Optional[np.ndarray]]":
        for key in self.keys():
            a0 = getattr(self, key)
            a1 = fmap(np.asarray, a0)
            a2 = cast("Optional[np.ndarray]", a1)
            yield a2
        return

    def items(self) -> "Iterator[Tuple[str, Optional[np.ndarray]]]":
        for key in self.keys():
            a0 = getattr(self, key)
            a1 = fmap(np.asarray, a0)
            a2 = cast("Optional[np.ndarray]", a1)
            yield key, a2

    def to_dict(self) -> "Dict[str, Optional[np.ndarray]]":
        return {k: v for k, v in self.items()}


class CVData(object):

    def __init__(
        self,
        markers: "npt.ArrayLike",
        groups: "Optional[npt.ArrayLike]",
        covariates: "Optional[npt.ArrayLike]",
        y: "npt.ArrayLike",
        nsplits: int = 5,
        seed: "Optional[int]" = None
    ):
        self.markers = np.asarray(markers)
        self.groups = fmap(np.asarray, groups)
        self.covariates = fmap(np.asarray, covariates)
        self.y = np.asarray(y)

        self.rng = random.Random(seed)

        self.nsplits = nsplits
        return

    def reset(self):
        del self.splits

    @property
    def nsplits(self):
        return self._nsplits

    @nsplits.setter
    def nsplits(self, value: int):
        self._nsplits = value
        self._gen_splits(value)

    def _gen_splits(self, k: int):
        # TODO Set a numpy array with integers
        cv = KFold(
            n_splits=self.nsplits,
            shuffle=True,
            random_state=self.rng.getrandbits(32)
        )

        ints = cv.split(self.markers)

        self.splits = np.zeros(self.markers.shape[0])
        for i, (_, cv_ints) in enumerate(ints):
            self.splits[cv_ints] = i

        return

    def __call__(self) -> "Iterator[Tuple[Dataset, Dataset]]":
        # Run CV and evaluate

        for i in np.unique(self.splits):
            train_mask = self.splits != i
            test_mask = self.splits == i

            train_markers = self.markers[train_mask]
            test_markers = self.markers[test_mask]

            if self.groups is not None:
                train_groups = self.groups[train_mask]
                test_groups = self.groups[test_mask]
            else:
                train_groups = test_groups = None

            if self.covariates is not None:
                train_covariates = self.covariates[train_mask]
                test_covariates = self.covariates[test_mask]
            else:
                train_covariates = test_covariates = None

            train_y = self.y[train_mask]
            test_y = self.y[test_mask]

            yield (
                Dataset(train_markers, train_y, train_groups, train_covariates),  # noqa
                Dataset(test_markers, test_y, test_groups, test_covariates)
            )
        return
