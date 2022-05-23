import numpy as np
from dataclasses import dataclass
from baikal import Step, Input, Model

from typing import cast
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, Sequence, List, Tuple
    from typing import Optional, Union
    from typing import Literal
    import optuna
    import numpy.typing as npt
    BaseTypes = Union[None, bool, str, int, float]
    Params = Dict[str, BaseTypes]

from .cv import Dataset
from .optimise import OptimiseBase


@dataclass
class ModelPath:
    """ Just for making return tuples more pleasant. """

    X_input: Input
    y_input: Input
    g_input: "Optional[Input]" = None
    c_input: "Optional[Input]" = None
    target_transformer: "Optional[Step]" = None
    target_model: "Optional[Step]" = None
    marker_model: "Optional[Step]" = None
    dist_model: "Optional[Step]" = None
    nonlinear_model: "Optional[Step]" = None
    grouping_model: "Optional[Step]" = None
    covariate_model: "Optional[Step]" = None
    joined_model: "Optional[Step]" = None
    interactions_model: "Optional[Step]" = None

    def items(self):
        from dataclass import fields
        return {
            f: getattr(self, f)
            for f
            in fields(self.__class__)
        }


@dataclass
class DataPath:

    cv: Dataset
    y: np.ndarray
    X_marker: "Optional[np.ndarray]" = None
    X_dist: "Optional[np.ndarray]" = None
    X_nonlinear: "Optional[np.ndarray]" = None
    covariates: "Optional[np.ndarray]" = None
    groups: "Optional[np.ndarray]" = None
    joined: "Optional[np.ndarray]" = None
    interactions: "Optional[np.ndarray]" = None

    def items(self):
        from dataclass import fields
        return {
            f: getattr(self, f)
            for f
            in fields(self.__class__)
        }


class BaseRunner(object):
    """ Subclass this object.

    If you wish to use the same feature selection settings as default,
    use: super().__init__()

    Otherwise copy paste init to suit your needs.
    """

    def _init_preprocessors(
        self,
        ploidy: int = 2,
        target_options: "List[str]" = [
            "passthrough",
            "stdnorm",
            "quantile"
        ],
        covariate_options: "List[str]" = [
            "passthrough",
            "stdnorm",
            "robust",
            "quantile",
            "power",
        ],
        group_options: "List[str]" = [
            "passthrough",
            "onehot",
            "pca"
        ],
        marker_fs_options: "List[str]" = [
            "passthrough",
            "drop",
            "maf",
            "relief",
            "gemma"
        ],
        marker_options: "List[str]" = [
            "passthrough",
            "maf",
            "onehot",
            "noia_add",
            "pca"
        ],
        dist_fs_options: "List[str]" = [
            "passthrough",
            "drop",
            "maf",
            "relief",
            "gemma"
        ],
        dist_options: "List[str]" = [
            "vanraden",
            "manhattan",
            "euclidean"
        ],
        nonlinear_fs_options: "List[str]" = [
            "passthrough",
            "drop",
            "maf",
            "relief",
            "gemma"
        ],
        nonlinear_options: "List[str]" = [
            "rbf",
            "laplacian",
            "poly"
        ],
        interactions_options: "List[str]" = [
            "drop",
            "rbf",
            "laplacian",
            "poly"
        ],
        use_fs_cache: bool = True,
        use_covariate_polynomial: bool = True,
    ):
        from selectml.optimiser.optimise import (
            OptimiseTarget,
            OptimiseCovariates,
            OptimiseFeatureSelector,
            OptimiseMarkerTransformer,
            OptimiseDistTransformer,
            OptimiseNonLinear,
            OptimiseGrouping,
            OptimiseInteractions,
        )

        self.ploidy = ploidy

        self.target_transformer = OptimiseTarget(target_options)
        self.covariate_transformer = OptimiseCovariates(
            covariate_options,
            use_polynomial=use_covariate_polynomial
        )
        self.marker_feature_selector = OptimiseFeatureSelector(
            options=marker_fs_options,
            name="marker_feature_selector",
            use_cache=use_fs_cache,
        )
        self.marker_transformer = OptimiseMarkerTransformer(
            options=marker_options,
            ploidy=ploidy,
            max_ncomponents=200
        )

        self.dist_feature_selector = OptimiseFeatureSelector(
            options=dist_fs_options,
            name="dist_feature_selector",
            use_cache=use_fs_cache,
        )
        self.dist_transformer = OptimiseDistTransformer(
            ploidy=ploidy,
            options=dist_options
        )

        self.nonlinear_feature_selector = OptimiseFeatureSelector(
            options=nonlinear_fs_options,
            name="nonlinear_feature_selector",
            use_cache=use_fs_cache,
        )
        self.nonlinear_transformer = OptimiseNonLinear(
            options=nonlinear_options,
            ploidy=ploidy,
            max_ncomponents=200,
        )

        self.grouping_transformer = OptimiseGrouping(
            options=group_options,
            max_ncomponents=50
        )
        self.interactions_transformer = OptimiseInteractions(
            options=interactions_options,
            max_ncomponents=200
        )
        return

    def sample(
        self,
        trial: "optuna.Trial",
        cv: "List[Dataset]",
    ) -> "Tuple[Model, Params, npt.ArrayLike, npt.ArrayLike]":
        raise NotImplementedError()
        return

    def model(
        self,
        params: "Params",
        data: "Dataset",
    ) -> "Tuple[Model, Params, npt.ArrayLike, npt.ArrayLike]":
        raise NotImplementedError()
        return

    def _sample_preprocessing_step(
        self,
        trial: "optuna.Trial",
        params: "Params",
        transformer: "Optional[OptimiseBase]",
        data: "List[Optional[np.ndarray]]",
        data2: "Optional[List[Optional[np.ndarray]]]" = None,
        predict: bool = False,
        assert_not_none: bool = False,
        drop_if_none: bool = True,
    ) -> "Tuple[Params, List[Model], List[Optional[np.ndarray]]]":
        from copy import copy
        from itertools import cycle
        from selectml.higher import ffmap

        params = copy(params)

        if (transformer is not None) and not any(d is None for d in data):
            trans_params = transformer.sample(trial, data)
            params.update(trans_params)

            if data2 is not None:
                models: "List[Optional[Model]]" = list(map(
                    transformer.fit,
                    cycle([params]),
                    map(np.asarray, data),
                    map(np.asarray, data2)
                ))
            else:
                models = list(map(
                    transformer.fit,
                    cycle([params]),
                    map(np.asarray, data)
                ))

            if assert_not_none:
                assert not any(m is None for m in models)

            if predict:
                attr = "predict"
            else:
                attr = "transform"

            def safe_attr(cls, attr):
                if cls is None:
                    return None
                else:
                    return getattr(cls, attr)

            dataout: "List[Optional[np.ndarray]]" = [
                ffmap(safe_attr(mi, attr), di)
                for mi, di
                in zip(models, data)
            ]
        else:
            models = [None for _ in data]

            if drop_if_none:
                dataout = [None for _ in data]
            else:
                dataout = data

        if (
            any(m is None for m in models)
            and not all(m is None for m in models)
        ):
            raise ValueError("Either all models are None, or none are None.")

        if (
            any(d is None for d in dataout)
            and not all(d is None for d in dataout)
        ):
            raise ValueError("Either all data is None, or no data is None.")

        return params, models, dataout

    def _sample_preprocessing(  # noqa
        self,
        trial: "optuna.Trial",
        cv: "Sequence[Dataset]",
    ):
        """ This is appropriate for SKLearn APIs
        (use super().sample(trial) to get preprocessing).
        For tensorflow and BGLR you should overwrite sample.
        """

        from selectml.higher import ffmap, ffmap2
        from baikal import Input
        from baikal.steps import ColumnStack

        params: "Params" = {}
        X: "List[Optional[np.ndarray]]" = [cvi.markers for cvi in cv]
        assert not any(xi is None for xi in X)
        y: "List[Optional[np.ndarray]]" = [cvi.y for cvi in cv]
        assert not any(yi is None for yi in y)
        g: "List[Optional[np.ndarray]]" = [cvi.groups for cvi in cv]
        c: "List[Optional[np.ndarray]]" = [cvi.covariates for cvi in cv]

        model_paths = []
        for xi, yi, gi, ci in zip(X, y, g, c):
            assert yi is not None

            mi = ModelPath(
                X_input=Input(name="markers"),
                y_input=Input(name="y"),
                g_input=None if (gi is None) else Input(name="groups"),
                c_input=None if (ci is None) else Input(name="covariates"),
            )
            model_paths.append(mi)

        params, target_models, y_ = self._sample_preprocessing_step(
            trial,
            params,
            self.target_transformer,
            y,
            assert_not_none=True,
            drop_if_none=False,
        )

        for mi, tm in zip(model_paths, target_models):
            mi.target_model = tm(mi.y_input)

        for mi, tm in zip(model_paths, target_models):
            mi.target_transformer = tm

        params, covariate_models, c = self._sample_preprocessing_step(
            trial,
            params,
            self.covariate_transformer,
            c,
        )
        for mi, cm in zip(model_paths, covariate_models):
            mi.covariate_model = ffmap(cm, mi.c_input)

        params, grouping_models, g = self._sample_preprocessing_step(
            trial,
            params,
            self.grouping_transformer,
            g,
        )
        for mi, gm in zip(model_paths, grouping_models):
            mi.grouping_model = ffmap(gm, mi.g_input)

        (
            params,
            feature_selection_models,
            X_marker
        ) = self._sample_preprocessing_step(
            trial,
            params,
            self.marker_feature_selector,
            X,
            data2=y,
        )
        feature_selection_models = [
            ffmap2(fsm, mi.X_input, mi.target_model)
            for fsm, mi
            in zip(feature_selection_models, model_paths)
        ]

        params, marker_models, X_marker = self._sample_preprocessing_step(
            trial,
            params,
            self.marker_transformer,
            X_marker,
        )

        for mi, mm, fs in zip(
            model_paths,
            marker_models,
            feature_selection_models
        ):
            mi.marker_model = ffmap(mm, fs)

        (
            params,
            dist_feature_selection_models,
            X_dist
        ) = self._sample_preprocessing_step(
            trial,
            params,
            self.dist_feature_selector,
            X,
            data2=y
        )
        dist_feature_selection_models = [
            ffmap2(fsm, mi.X_input, mi.target_model)
            for fsm, mi
            in zip(dist_feature_selection_models, model_paths)
        ]

        params, dist_models, X_dist = self._sample_preprocessing_step(
            trial,
            params,
            self.dist_transformer,
            X_dist,
        )

        for mi, dm, fs in zip(
            model_paths,
            dist_models,
            dist_feature_selection_models
        ):
            mi.dist_model = ffmap(dm, fs)

        (
            params,
            nonlinear_feature_selection_models,
            X_nonlinear
        ) = self._sample_preprocessing_step(
            trial,
            params,
            self.nonlinear_feature_selector,
            X,
            data2=y,
        )

        nonlinear_feature_selection_models = [
            ffmap2(fsm, mi.X_input, mi.target_model)
            for fsm, mi
            in zip(nonlinear_feature_selection_models, model_paths)
        ]

        (
            params,
            nonlinear_models,
            X_nonlinear
        ) = self._sample_preprocessing_step(
            trial,
            params,
            self.nonlinear_transformer,
            X_nonlinear,
        )

        for mi, nlm, fs in zip(
            model_paths,
            nonlinear_models,
            nonlinear_feature_selection_models
        ):
            mi.nonlinear_model = ffmap(nlm, fs)

        joined: "List[np.ndarray]" = []
        for xmi, xdi, xli, gi, ci in zip(X_marker, X_dist, X_nonlinear, g, c):
            joined_i: "List[np.ndarray]" = []

            if xmi is not None:
                joined_i.append(self._sparse_to_dense(xmi))

            if xdi is not None:
                joined_i.append(self._sparse_to_dense(xdi))

            if xli is not None:
                joined_i.append(self._sparse_to_dense(xli))

            if gi is not None:
                joined_i.append(self._sparse_to_dense(gi))

            if ci is not None:
                joined_i.append(self._sparse_to_dense(ci))

            # zi = mi, di, nli, gi, ci
            joined.append(np.concatenate(joined_i, axis=1))

        for mpi in model_paths:
            zm = [
                mpi.marker_model,
                mpi.dist_model,
                mpi.nonlinear_model,
                mpi.grouping_model,
                mpi.covariate_model
            ]

            # We checked along the way that if one is none, all are none.
            # So this should be safe.
            zm = [zmi for zmi in zm if zmi is not None]
            mpi.joined_model = ColumnStack()(zm)

        (
            params,
            interactions_models,
            interactions
        ) = self._sample_preprocessing_step(
            trial,
            params,
            self.interactions_transformer,
            cast("List[Optional[np.ndarray]]", joined),
        )

        for mi, jm in zip(model_paths, interactions_models):
            mi.interactions_model = ffmap(jm, mi.joined_model)

        data_paths = []
        for cvi, yi, xmi, xdi, xli, ci, gi, ji, ii in zip(
            cv,
            y,
            X_marker,
            X_dist,
            X_nonlinear,
            c,
            g,
            joined,
            interactions,
        ):
            assert yi is not None
            data_paths.append(DataPath(
                cvi,
                yi,
                xmi,
                xdi,
                xli,
                ci,
                gi,
                ji,
                ii,
            ))

        return params, model_paths, data_paths

    def _model_preprocessing(
        self,
        params: "Params",
        data: "Dataset",
        **kwargs
    ):
        from baikal import Input
        from baikal.steps import ColumnStack
        from selectml.higher import ffmap, ffmap2

        X_input = Input(name="markers")
        y_input = Input(name="y")
        g_input = None if data.groups is None else Input(name="groups")
        c_input = None if data.covariates is None else Input(name="covariates")

        target_transformer = self.target_transformer.model(params)
        target = ffmap(target_transformer, y_input)

        covariates = ffmap(self.covariate_transformer.model(params), c_input)
        groups = ffmap(self.grouping_transformer.model(params), g_input)

        marker_fs = ffmap2(
            self.marker_feature_selector.model(params),
            X_input,
            target
        )
        markers = ffmap(
            self.marker_transformer.model(params),
            marker_fs
        )

        dist_fs = ffmap2(
            self.dist_feature_selector.model(params),
            X_input,
            target
        )
        dists = ffmap(self.dist_transformer.model(params), dist_fs)

        nonlinear_fs = ffmap2(
            self.nonlinear_feature_selector.model(params),
            X_input,
            target
        )
        nonlinear = ffmap(
            self.nonlinear_transformer.model(params),
            nonlinear_fs
        )

        joined = ColumnStack()(list(
            filter(
                lambda x: x is not None,
                [markers, dists, nonlinear, groups, covariates]
            )))

        interactions = ffmap(
            self.interactions_transformer.model(params),
            joined
        )

        return ModelPath(
            X_input,
            y_input,
            g_input,
            c_input,
            target_transformer,
            target,
            markers,
            dists,
            nonlinear,
            groups,
            covariates,
            joined,
            interactions
        )

    @staticmethod
    def _select_inputs(
        model: "Model",
        data: "Dataset"
    ) -> "Dict[str, np.ndarray]":
        out = {}
        for key, value in data.items():
            if key == "y":
                continue
            try:
                model.get_step(key)
                assert value is not None
                out[key] = np.asarray(value)
            except ValueError:
                pass

        return out

    @staticmethod
    def _sparse_to_dense(X):
        if isinstance(X, np.ndarray):
            return X
        else:
            return X.todense()

    @classmethod
    def fit(cls, model, data: "Dataset"):
        assert data.y is not None
        return model.fit(cls._select_inputs(model, data), data.y)

    @classmethod
    def predict(cls, model: "Model", data: "Dataset") -> "np.ndarray":
        return model.predict(cls._select_inputs(model, data))


class SKRunner(BaseRunner):

    predictor: "Model"

    def __init__(
        self,
        task: "Literal['regression', 'ranking', 'ordinal', 'classification']",
        ploidy: int = 2
    ):
        raise NotImplementedError("Subclasses must implement this.")

    def sample(  # noqa
        self,
        trial: "optuna.Trial",
        cv: "List[Dataset]",
    ):
        from baikal.steps import ColumnStack

        from selectml.optimiser.wrapper import Make2D, Make1D

        from selectml.higher import ffmap2, fmap
        (
            params,
            model_paths,
            data_paths
        ) = self._sample_preprocessing(trial, cv)

        joined = []
        joined_models = []

        for mp, dp in zip(model_paths, data_paths):
            di = [
                self._sparse_to_dense(dii)
                for dii
                in [dp.joined, dp.interactions]
                if dii is not None
            ]
            assert len(di) > 0, "Both joined and interactions were None."
            joined.append(np.concatenate(di, axis=1))

            mi = [
                mii
                for mii
                in [mp.joined_model, mp.interactions_model]
                if mii is not None
            ]
            assert len(mi) > 0, "Both joined and interactions models were None"
            if len(mi) == 1:
                joined_models.append(mi[0])
            else:
                joined_models.append(ColumnStack()(mi))

        params.update(self.predictor.sample(trial, joined))

        make1d = Make1D()
        params, predictor_models, yhat = self._sample_preprocessing_step(
            trial,
            params,
            self.predictor,
            data=joined,
            data2=[make1d.transform(di.y) for di in data_paths],
            predict=True,
            assert_not_none=True,
            drop_if_none=False,
        )

        predictor_models = [
            ffmap2(pm, mi, Make1D()(mmi.target_model))
            for pm, mi, mmi
            in zip(predictor_models, joined_models, model_paths)
        ]

        # Some models output 1d, others output 2D.
        # We want consistency
        two_d_models = [
            fmap(Make2D(), pm)
            for pm
            in predictor_models
        ]

        def safe_partial(tm, pm):
            if tm is None:
                return pm
            if pm is None:
                return None

            return tm(pm, compute_func="inverse_transform", trainable=False)

        yhat_models = [
            safe_partial(mp.target_transformer, pm)
            for mp, pm
            in zip(model_paths, two_d_models)
        ]

        def get_yhat(
            mi: "Model",
            yhi: "Optional[np.ndarray]"
        ) -> "Optional[np.ndarray]":
            if yhi is None:
                return None
            elif len(yhi.shape) == 1:
                yhi_ = np.expand_dims(yhi, 1)
            else:
                yhi_ = cast("np.ndarray", yhi)

            if mi.target_transformer is None:
                return yhi_

            else:
                return mi.target_transformer.inverse_transform(yhi_)

        yhat = [
            get_yhat(mp, yi)
            for mp, yi
            in zip(model_paths, yhat)
        ]

        models = []
        for mp, tdi in zip(model_paths, yhat_models):
            any_x = any([
                getattr(mp, a) is not None
                for a in ["marker_model", "dist_model", "nonlinear_model"]]
            )
            inputs = []
            if any_x:
                inputs.append(mp.X_input)

            if mp.grouping_model is not None:
                inputs.append(mp.g_input)

            if mp.covariate_model is not None:
                inputs.append(mp.c_input)

            models.append(Model(
                inputs,
                tdi,
                mp.y_input
            ))

        return (
            params,
            models,
            yhat,
            joined
        )

    def model(self, params, data: "Dataset") -> "Model":
        from baikal.steps import ColumnStack
        from selectml.optimiser.wrapper import Make2D, Make1D

        fp = self._model_preprocessing(params, data)

        target = Make1D()(fp.target_model)

        if fp.interactions_model is not None:
            preprocessed = ColumnStack()([
                fp.joined_model,
                fp.interactions_model
            ])
        else:
            preprocessed = fp.joined_model

        model = self.predictor.model(params)(preprocessed, target)
        model = Make2D()(model)
        yhat = fp.target_transformer(
            model,
            compute_func="inverse_transform",
            trainable=False
        )

        any_x = any([
            getattr(fp, a) is not None
            for a in ["marker_model", "dist_model", "nonlinear_model"]]
        )
        inputs = [
            fp.X_input if any_x else None,
            fp.g_input if (fp.grouping_model is not None) else None,
            fp.c_input if (fp.covariate_model is not None) else None,
        ]

        return Model([ii for ii in inputs if ii is not None], yhat, fp.y_input)


class XGBRunner(SKRunner):

    def __init__(
        self,
        task: "Literal['regression', 'ranking', 'ordinal', 'classification']",
        ploidy: int = 2
    ):
        from selectml.optimiser.optimise import OptimiseXGB

        if task == "regression":
            objectives = ["reg:squarederror"]
            target = ["passthrough", "stdnorm", "quantile"]
        elif task == "ranking":
            objectives = [
                "reg:squarederror",
                "rank:pairwise",
                "reg:logistic",
                "binary:logistic"
            ]
            target = ["passthrough", "stdnorm", "quantile", "ordinal"]
        elif task == "ordinal":
            objectives = ["binary:logistic"]
            target = ["ordinal"]
        elif task == "classification":
            objectives = ["reg:logistic", "binary:logistic"]
            target = ["passthrough"]
        else:
            raise ValueError(
                "task must be regression, ranking, "
                "ordinal, or classification.")

        self._init_preprocessors(
            ploidy=ploidy,
            target_options=target,
            marker_fs_options=["passthrough", "maf", "relief", "gemma"],
            nonlinear_fs_options=["drop"],
            interactions_options=["drop"],
            use_covariate_polynomial=False,
        )

        self.predictor = OptimiseXGB(objectives=objectives)
        return


class TFRunner(SKRunner):

    def __init__(
        self,
        task: "Literal['regression', 'ranking', 'ordinal', 'classification']",
        ploidy: int = 2
    ):
        from selectml.optimiser.optimise import OptimiseConvMLP

        if task == "regression":
            objectives: "List[Literal['mse', 'mae', 'binary_crossentropy', 'pairwise']]" = ["mae", "mse"]  # noqa: E501
            target = ["passthrough", "stdnorm", "quantile"]
        elif task == "ranking":
            objectives = [
                "mae",
                "mse",
                "pairwise",
            ]
            target = ["passthrough", "stdnorm", "quantile", "ordinal"]
        elif task == "ordinal":
            objectives = ["binary_crossentropy"]
            target = ["ordinal"]
        elif task == "classification":
            objectives = ["binary_crossentropy"]
            target = ["passthrough"]
        else:
            raise ValueError(
                "task must be regression, ranking, "
                "ordinal or classification.")

        # Keras seems to override something to do with deepcopy or mutexes
        # that means we can't cache the feature selector.
        self._init_preprocessors(
            ploidy=ploidy,
            target_options=target,
            marker_fs_options=["passthrough", "maf", "relief", "gemma"],
            nonlinear_fs_options=["drop"],
            interactions_options=["drop"],
            use_fs_cache=False,
            use_covariate_polynomial=False,
        )

        self.predictor = OptimiseConvMLP(loss=objectives)
        return

    def _sample_tf_step(  # noqa: C901
        self,
        trial: "optuna.Trial",
        params: "Params",
        transformer: "Optional[OptimiseBase]",
        markers: "List[Optional[np.ndarray]]",
        dists: "List[Optional[np.ndarray]]",
        groups: "List[Optional[np.ndarray]]",
        covariates: "List[Optional[np.ndarray]]",
        y: "Optional[List[Optional[np.ndarray]]]" = None,
        predict: bool = False,
        assert_not_none: bool = False,
        drop_if_none: bool = True,
    ) -> "Tuple[Params, List[Model], List[Optional[np.ndarray]]]":
        from copy import copy
        from itertools import cycle
        from selectml.higher import ffmap

        params = copy(params)

        if (transformer is not None) and not any(d is None for d in markers):
            trans_params = transformer.sample(
                trial,
                markers,
                dists=dists,
                groups=groups,
                covariates=covariates,
            )
            params.update(trans_params)

            data: "List[List[np.ndarray]]" = []
            if all(d is not None for d in dists):
                data.append(cast("List[np.ndarray]", markers))

            if all(d is not None for d in dists):
                data.append(cast("List[np.ndarray]", dists))

            if all(d is not None for d in groups):
                data.append(cast("List[np.ndarray]", groups))

            if all(d is not None for d in covariates):
                data.append(cast("List[np.ndarray]", covariates))

            if len(data) == 1:
                data_: "Union[List[np.ndarray], List[List[np.ndarray]]]" = data[0]  # noqa: E501
            elif len(data) > 1:
                data_ = list(map(lambda x: list(x), zip(*data)))
            else:
                raise ValueError("didn't get enough data")

            if y is not None:
                models: "List[Optional[Model]]" = list(map(
                    transformer.fit,
                    cycle([params]),
                    data_,
                    y
                ))
            else:
                models = list(map(
                    transformer.fit,
                    cycle([params]),
                    data_
                ))

            if assert_not_none:
                assert not any(m is None for m in models)

            if predict:
                attr = "predict"
            else:
                attr = "transform"

            def safe_attr(cls, attr):
                if cls is None:
                    return None
                else:
                    return getattr(cls, attr)

            dataout: "List[Optional[np.ndarray]]" = [
                ffmap(safe_attr(mi, attr), di)
                for mi, di
                in zip(models, data_)
            ]
        else:
            models = [None for _ in markers]

            if drop_if_none:
                dataout = [None for _ in markers]
            else:
                dataout = markers

        if (
            any(m is None for m in models)
            and not all(m is None for m in models)
        ):
            raise ValueError("Either all models are None, or none are None.")

        if (
            any(d is None for d in dataout)
            and not all(d is None for d in dataout)
        ):
            raise ValueError("Either all data is None, or no data is None.")

        return params, models, dataout

    def sample(  # noqa
        self,
        trial: "optuna.Trial",
        cv: "List[Dataset]",
    ):
        from selectml.optimiser.wrapper import Make2D, Make1D

        from selectml.higher import ffmap2, fmap
        (
            params,
            model_paths,
            data_paths
        ) = self._sample_preprocessing(trial, cv)

        joined = []
        joined_models = []

        name_map = {
            "X_marker": "markers",
            "X_dist": "dists",
        }

        model_map = {
            "marker_model": "markers",
            "dist_model": "dists",
            "grouping_model": "groups",
            "covariate_model": "covariates",
        }
        for mp, dp in zip(model_paths, data_paths):
            di = {
                name_map.get(dk, dk): self._sparse_to_dense(getattr(dp, dk))
                for dk
                in ["X_marker", "X_dist", "groups", "covariates"]
                if getattr(dp, dk) is not None
            }
            assert len(di) > 0, "Both joined and interactions were None."
            joined.append(di)
            mi = {
                model_map.get(mk, mk): getattr(mp, mk)
                for mk
                in ["marker_model", "dist_model", "grouping_model", "covariate_model"]  # noqa: E501
                if getattr(mp, mk) is not None
            }
            assert len(mi) > 0, "Both joined and interactions models were None"
            joined_models.append(mi)

        #  params.update(self.predictor.sample(trial, joined))

        make1d = Make1D()
        params, predictor_models, yhat = self._sample_tf_step(
            trial,
            params,
            self.predictor,
            markers=[j.get("markers", None) for j in joined],
            dists=[j.get("dists", None) for j in joined],
            groups=[j.get("groups", None) for j in joined],
            covariates=[j.get("covariates", None) for j in joined],
            y=[make1d.transform(di.y) for di in data_paths],
            predict=True,
            assert_not_none=True,
            drop_if_none=False,
        )

        predictor_models = [
            ffmap2(pm, list(mi.values()), Make1D()(mmi.target_model))
            for pm, mi, mmi
            in zip(predictor_models, joined_models, model_paths)
        ]

        # Some models output 1d, others output 2D.
        # We want consistency
        two_d_models = [
            fmap(Make2D(), pm)
            for pm
            in predictor_models
        ]

        def safe_partial(tm, pm):
            if tm is None:
                return pm
            if pm is None:
                return None

            return tm(pm, compute_func="inverse_transform", trainable=False)

        yhat_models = [
            safe_partial(mp.target_transformer, pm)
            for mp, pm
            in zip(model_paths, two_d_models)
        ]

        def get_yhat(
            mi: "Model",
            yhi: "Optional[np.ndarray]"
        ) -> "Optional[np.ndarray]":
            if yhi is None:
                return None
            elif len(yhi.shape) == 1:
                yhi_ = np.expand_dims(yhi, 1)
            else:
                yhi_ = cast("np.ndarray", yhi)

            if mi.target_transformer is None:
                return yhi_

            else:
                return mi.target_transformer.inverse_transform(yhi_)

        yhat = [
            get_yhat(mp, yi)
            for mp, yi
            in zip(model_paths, yhat)
        ]

        models = []
        for mp, tdi in zip(model_paths, yhat_models):
            any_x = any([
                getattr(mp, a) is not None
                for a in ["marker_model", "dist_model", "nonlinear_model"]]
            )
            inputs = []
            if any_x:
                inputs.append(mp.X_input)

            if mp.grouping_model is not None:
                inputs.append(mp.g_input)

            if mp.covariate_model is not None:
                inputs.append(mp.c_input)

            models.append(Model(
                inputs,
                tdi,
                mp.y_input
            ))

        return (
            params,
            models,
            yhat,
            joined
        )

    def model(self, params, data: "Dataset") -> "Model":
        from selectml.optimiser.wrapper import Make2D, Make1D

        fp = self._model_preprocessing(params, data)

        target = Make1D()(fp.target_model)

        preprocessed = []
        if fp.marker_model is not None:
            preprocessed.append(fp.marker_model)

        if fp.dist_model is not None:
            preprocessed.append(fp.dist_model)

        if fp.grouping_model is not None:
            preprocessed.append(fp.grouping_model)

        if fp.covariate_model is not None:
            preprocessed.append(fp.covariate_model)

        model = self.predictor.model(params)(preprocessed, target)
        model = Make2D()(model)
        yhat = fp.target_transformer(
            model,
            compute_func="inverse_transform",
            trainable=False
        )

        any_x = any([
            getattr(fp, a) is not None
            for a in ["marker_model", "dist_model", "nonlinear_model"]]
        )
        inputs = [
            fp.X_input if any_x else None,
            fp.g_input if (fp.grouping_model is not None) else None,
            fp.c_input if (fp.covariate_model is not None) else None,
        ]

        return Model([ii for ii in inputs if ii is not None], yhat, fp.y_input)
