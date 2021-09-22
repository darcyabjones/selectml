import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

numpy2ri.activate()


class BGLRModel(object):

    def predict(self, X, y, model):
        BGLR = importr("BGLR")

        ETA = [
            ro.ListVector({"X": X, "model": model})
        ]

        results = BGLR.BGLR(
            y=y,
            ETA=ETA,
            verbose=False
        )
        return BGLR.predict_BGLR(results)
