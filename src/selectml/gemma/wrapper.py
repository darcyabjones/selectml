#!/usr/bin/env python3

from contextlib import contextmanager


import numpy as np
import pandas as pd
from os.path import join as pjoin

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import IO, Union, Optional, List, Any, Iterator
    import numpy.typing as npt


class GEMMA(object):

    def __init__(
        self,
        exe: str = "gemma",
        AA: "Union[int, float]" = 2,
        Aa: "Union[int, float]" = 1,
        aa: "Union[int, float]" = 0
    ):
        self.exe = exe
        self.AA = AA
        self.Aa = Aa
        self.aa = aa
        return

    def genos_to_bimbam(
        self,
        X: "npt.ArrayLike",
        marker_columns: "Optional[List[Any]]" = None
    ) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = df.copy()

        if marker_columns is None:
            marker_columns = list(df.columns)

        df = df.loc[:, marker_columns].T.astype(float)
        df[np.isclose(df.values, float(self.AA))] = 1
        df[np.isclose(df.values, float(self.Aa))] = 0.5
        df[np.isclose(df.values, float(self.aa))] = 0
        df.index.name = "SNP_ID"
        df.reset_index(inplace=True, drop=False)
        df["MAJ"] = "X"
        df["MIN"] = "Y"
        df = df[["SNP_ID", "MAJ", "MIN"] + list(df.columns[1:-2])]
        return df

    @staticmethod
    def prep_covariates(X: "npt.ArrayLike") -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            df = X.values
        else:
            df = np.array(X)
        singular = np.all(df.sum(axis=1) == 1)

        if singular:
            if isinstance(df, pd.DataFrame):
                df = df.iloc[:, 2:].values
            else:
                df = df[:, 2:]

        df = np.column_stack((df, np.ones(df.shape[0])))
        return df

    @contextmanager
    def write_inputs(
        self,
        X: "npt.ArrayLike",
        y: "npt.ArrayLike",
        covariates: "Optional[npt.ArrayLike]" = None
    ) -> "Iterator":
        from tempfile import NamedTemporaryFile
        X_file = NamedTemporaryFile(mode="w+")
        y_file = NamedTemporaryFile(mode="w+")

        if covariates is not None:
            cov_file: "Optional[IO[str]]" = NamedTemporaryFile(mode="w+")
        else:
            cov_file = None

        try:
            X_ = self.genos_to_bimbam(X)
            X_.to_csv(X_file, index=False, header=False, na_rep="NA")
            X_file.seek(0)
            del X_

            y_ = np.array(y)
            if len(y_.shape) == 1:
                y_ = np.expand_dims(y_, -1)

            pd.DataFrame(y_).to_csv(
                y_file,
                index=False,
                header=False,
                na_rep="NA"
            )
            y_file.seek(0)
            del y_

            if covariates is not None:
                cov = self.prep_covariates(covariates)
                pd.DataFrame(cov).to_csv(
                    cov_file,
                    index=False,
                    header=False,
                    na_rep="NA"
                )
                assert cov_file is not None
                cov_file.seek(0)
                del cov

            yield X_file, y_file, cov_file
        finally:
            X_file.close()
            y_file.close()

            if cov_file is not None:
                cov_file.close()

        return

    def get_assocs(
        self,
        X: "npt.ArrayLike",
        y: "npt.ArrayLike",
        covariates: "Optional[npt.ArrayLike]" = None
    ) -> pd.DataFrame:
        from tempfile import TemporaryDirectory
        import subprocess

        y_ = np.array(y)

        if len(y_.shape) == 1:
            y_ = np.expand_dims(y_, -1)

        n_vars = y_.shape[1]

        with TemporaryDirectory() as tdir_name, \
                self.write_inputs(X, y_) as (X_handle, y_handle, cov_handle):

            del y_
            cmd = [
                self.exe,
                "-g", X_handle.name,
                "-p", y_handle.name,
                "-gk",
                "-outdir", tdir_name
            ]

            if cov_handle is not None:
                cmd.extend(["-c", cov_handle.name])

            subprocess.run(cmd, check=True, capture_output=True)

            cmd = [
                self.exe,
                "-g", X_handle.name,
                "-p", y_handle.name,
                "-k", pjoin(tdir_name, "result.cXX.txt"),
                "-outdir", tdir_name,
                "-lmm", "1",
                "-miss", "1",
                "-r2", "1.0",
                "-notsnp",
                "-n"
            ]

            cmd.extend(map(str, range(1, n_vars + 1)))

            if cov_handle is not None:
                cmd.extend(["-c", cov_handle.name])

            subprocess.run(cmd, check=True, capture_output=True)
            df = pd.read_csv(pjoin(tdir_name, "result.assoc.txt"), sep="\t")
        return df
