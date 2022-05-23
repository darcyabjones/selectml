
from ..higher import fmap
import numpy as np

from typing import Literal
from typing import Sequence, List
from typing import Union
from numpy import typing as npt


class BGLR(object):

    def __init__(
        self,
        models: "List[Literal['FIXED', 'BRR', 'BL', 'BayesB', 'BayesC', 'RKHS']]",  # noqa: E501
        niter: int = 1500,
        burnin: int = 500,
        thin: int = 5,
        saveat: str = "",
        response_type: "Literal['gaussian', 'ordinal']" = "gaussian",
        verbose: bool = True,
        rm_existing_files: bool = True,
    ):
        self.models = models
        self.niter = niter
        self.burnin = burnin
        self.thin = thin
        self.saveat = saveat
        self.response_type = response_type
        self.verbose = verbose
        self.rm_existing_files = rm_existing_files
        return

    def fit(self, X, y, sample_weight=None, group=None):
        model = self.run_bglr(X, y, sample_weight, group)
        self.bglr_model = model
        return self

    def predict(self, X):
        return

    def predict_bglr(
        self,
        model: str,
        X: "Sequence[npt.ArrayLike]",
        group=None,
    ) -> "np.ndarray":
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()

        X_ = [np.asarray(xi) for xi in X]
        """
        prediction = model.mu
        for data, mtype, eta in zip(X_, self.models, model.ETA):
            prediction += data.dot(eta$b)

        # it's unclear if need to do something different for RKHS
        yTRN=y[-tst]   ; yTST=y[tst]
        X.TRN=X[-tst,] ; X.TST=X[tst,]

        fm=BGLR(y=yTRN,ETA=list(list(X=X.TRN,model='BRR')),nIter=6000,burnIn=1000)
        yHat_2=fm$mu+as.vector(X.TST%*%fm$ETA[[1]]$b)

        G=tcrossprod(X)/ncol(X)

        G11=G[-tst,-tst] # genomic relationships in the training data
        G21=G[tst,-tst]

        fm=BGLR(y=yTRN,ETA=list(list(K=G11,model='RKHS')),nIter=6000,burnIn=1000)
        yHat_3=fm$mu+as.vector(G21%*%solve(G11)%*%fm$ETA[[1]]$u)
        cor(cbind(yHat_1,yHat_2,yHat_3))
        """

        return

    def fit_bglr(
        self,
        X: "Sequence[npt.ArrayLike]",
        y: "npt.ArrayLike",
        sample_weight=None,
        group=None,
    ) -> "np.ndarray":
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()

        X_ = [np.asarray(xi) for xi in X]
        y_ = np.asarray(y)

        BGLR = importr("BGLR")
        ETA = []

        for data, mtype in zip(X_, self.models):
            if mtype == "RKHS":
                ETA.append(ro.ListVector({"K": data, "model": mtype}))
            else:
                ETA.append(ro.ListVector({"X": data, "model": mtype}))

        results = BGLR.BGLR(
            y=y_,
            response_type=self.response_type,
            ETA=ETA,
            weights=sample_weight,
            nIter=self.niter,
            burnIn=self.bunin,
            thin=self.thin,
            saveAt=self.saveat,
            verbose=self.verbose,
            rmExistingFiles=self.rm_existing_files,
            groups=fmap(ro.vectors.FactorVector, group),
        )

        preds = BGLR.predict_BGLR(results)

        if len(preds.shape) == 1:
            preds = np.expand_dims(preds, -1)

        if self.target_trans is not None:
            preds = self.target_trans.inverse_transform(
                preds,
            )

        numpy2ri.deactivate()
        return results
