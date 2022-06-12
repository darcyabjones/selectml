import random

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
from .optimise import MyModel


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
    additive_model: "Optional[Step]" = None
    dominance_model: "Optional[Step]" = None
    epistasis_add_model: "Optional[Step]" = None
    epistasis_dom_model: "Optional[Step]" = None
    epistasis_addxdom_model: "Optional[Step]" = None

    def items(self):
        from dataclasses import fields
        for f in fields(self.__class__):
            yield f.name, getattr(self, f.name)
        return


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
    X_add: "Optional[np.ndarray]" = None
    X_dom: "Optional[np.ndarray]" = None
    X_epiadd: "Optional[np.ndarray]" = None
    X_epidom: "Optional[np.ndarray]" = None
    X_epiaddxdom: "Optional[np.ndarray]" = None

    def items(self):
        from dataclasses import fields
        for f in fields(self.__class__):
            yield f.name, getattr(self, f.name)
        return


class BaseRunner(object):
    """ Subclass this object.

    If you wish to use the same feature selection settings as default,
    use: super().__init__()

    Otherwise copy paste init to suit your needs.
    """

    rng: random.Random

    def __init__(
        self,
        task: "Literal['regression', 'ranking', 'ordinal', 'classification']",
        ploidy: int = 2,
        seed: "Optional[int]" = None,
        **kwargs,
    ):
        raise NotImplementedError("Subclasses must implement this.")

    def starting_points(self) -> "List[Params]":
        import inspect
        import copy

        members = inspect.getmembers(self)

        starting_points: "List[Params]" = []
        for key, member in members:
            if not isinstance(member, OptimiseBase):
                continue

            if len(starting_points) == 0:
                starting_points.extend(member.starting_points())
                continue

            new_starting_points: "List[Params]" = []
            for sp1 in starting_points:
                for sp2 in member.starting_points():
                    sp1 = copy.copy(sp1)
                    sp1.update(copy.copy(sp2))
                    new_starting_points.append(sp1)

            starting_points = new_starting_points

        return starting_points

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
        group_allow_pca: bool = True,
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
            "noia_additive",
            "noia_dominance",
        ],
        dist_post_fs_options: "List[str]" = [
            "passthrough",
            "f_classif",
            "chi2",
            "f_regression",
            "mutual_info_regression",
            "mutual_info_classif",
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
        nonlinear_post_fs_options: "List[str]" = [
            "passthrough",
            "f_classif",
            "chi2",
            "f_regression",
            "mutual_info_regression",
            "mutual_info_classif",
        ],
        interactions_options: "List[str]" = [
            "drop",
            "rbf",
            "laplacian",
            "poly"
        ],
        interactions_post_fs_options: "List[str]" = [
            "passthrough",
            "f_classif",
            "chi2",
            "f_regression",
            "mutual_info_regression",
            "mutual_info_classif",
        ],
        use_fs_cache: bool = True,
        use_covariate_polynomial: bool = True,
        *kwargs,

    ):
        from selectml.optimiser.optimise import (
            OptimiseTarget,
            OptimiseCovariates,
            OptimiseFeatureSelector,
            OptimiseMarkerTransformer,
            OptimiseDistTransformer,
            OptimiseNonLinear,
            OptimiseGrouping,
            OptimisePostFeatureSelector,
            OptimiseInteractions,
        )

        self.ploidy = ploidy

        self.target_transformer = OptimiseTarget(options=target_options)
        self.covariate_transformer = OptimiseCovariates(
            options=covariate_options,
            use_polynomial=use_covariate_polynomial
        )
        self.marker_feature_selector = OptimiseFeatureSelector(
            options=marker_fs_options,
            name="marker_feature_selector",
            use_cache=use_fs_cache,
            seed=self.rng.getrandbits(32),
        )
        self.marker_transformer = OptimiseMarkerTransformer(
            options=marker_options,
            ploidy=ploidy,
            max_ncomponents=200,
            seed=self.rng.getrandbits(32),
        )

        self.dist_feature_selector = OptimiseFeatureSelector(
            options=dist_fs_options,
            name="dist_feature_selector",
            use_cache=use_fs_cache,
            seed=self.rng.getrandbits(32),
        )
        self.dist_transformer = OptimiseDistTransformer(
            ploidy=ploidy,
            options=dist_options
        )
        self.dist_post_feature_selector = OptimisePostFeatureSelector(
            options=dist_post_fs_options,
            name="dist_post_feature_selector",
        )

        self.nonlinear_feature_selector = OptimiseFeatureSelector(
            options=nonlinear_fs_options,
            name="nonlinear_feature_selector",
            use_cache=use_fs_cache,
            seed=self.rng.getrandbits(32),
        )
        self.nonlinear_transformer = OptimiseNonLinear(
            options=nonlinear_options,
            ploidy=ploidy,
            max_ncomponents=200,
        )
        self.nonlinear_post_feature_selector = OptimisePostFeatureSelector(
            options=nonlinear_post_fs_options,
            name="nonlinear_post_feature_selector",
        )

        self.grouping_transformer = OptimiseGrouping(
            allow_pca=group_allow_pca,
            max_ncomponents=50,
            seed=self.rng.getrandbits(32),
        )

        self.interactions_transformer = OptimiseInteractions(
            options=interactions_options,
            max_ncomponents=200,
            seed=self.rng.getrandbits(32),
        )
        self.interactions_post_feature_selector = OptimisePostFeatureSelector(
            options=interactions_post_fs_options,
            name="interactions_post_feature_selector",
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
        from selectml.higher import ffmap2

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
                ffmap2(safe_attr(transformer, attr), mi, di)
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

    def _sample_extended_step(  # noqa: C901
        self,
        trial: "optuna.Trial",
        params: "Params",
        transformer: "Optional[OptimiseBase]",
        markers: "List[Optional[np.ndarray]]",
        groups: "List[Optional[np.ndarray]]",
        covariates: "List[Optional[np.ndarray]]",
        y: "Optional[List[Optional[np.ndarray]]]" = None,
        predict: bool = False,
        assert_not_none: bool = False,
        drop_if_none: bool = True,
    ) -> "Tuple[Params, List[Model], List[Optional[np.ndarray]], bool, bool, bool]":  # noqa
        from copy import copy
        from selectml.higher import ffmap2

        params = copy(params)

        if (transformer is not None) and not any(d is None for d in markers):
            trans_params = transformer.sample(
                trial,
                markers,
                groups=groups,
                covariates=covariates,
            )
            params.update(trans_params)

            use_markers: bool = True
            use_groups = params.get(
                f"{transformer.name}_use_groups",
                False
            )
            assert isinstance(use_groups, bool)
            use_covariates = params.get(
                f"{transformer.name}_use_covariates",
                False
            )
            assert isinstance(use_covariates, bool)

            data: "List[List[np.ndarray]]" = []
            if all(d is not None for d in markers) and use_markers:
                data.append(cast("List[np.ndarray]", markers))

            if all(d is not None for d in groups) and use_groups:
                data.append(cast("List[np.ndarray]", groups))

            if all(d is not None for d in covariates) and use_covariates:
                data.append(cast("List[np.ndarray]", covariates))

            if len(data) == 1:
                data_: "Union[List[np.ndarray], List[List[np.ndarray]]]" = list(data[0])  # noqa: E501
            elif len(data) > 1:
                data_ = [list(d) for d in zip(*data)]
            else:
                raise ValueError("didn't get enough data")

            if predict:
                attr = "predict"
            else:
                attr = "transform"

            def safe_attr(cls, attr):
                if cls is None:
                    return None
                else:
                    return getattr(cls, attr)

            models: "List[Optional[Model]]" = []
            dataout: "List[Optional[np.ndarray]]" = []

            if y is not None:
                for di, yi in zip(data_, y):
                    if isinstance(di, np.ndarray):
                        model = transformer.fit(
                            params,
                            di,
                            yi,
                        )
                    elif isinstance(di, list):
                        assert all(isinstance(dii, np.ndarray) for dii in di)
                        model = transformer.fit(
                            params,
                            di,
                            yi,
                        )

                    models.append(model)

                    dout = ffmap2(
                        safe_attr(transformer, attr),
                        model,
                        di,
                    )
                    dataout.append(dout)

            else:
                for di in data_:
                    model = transformer.fit(
                        params,
                        di,
                    )
                    models.append(model)

                    dout = ffmap2(
                        safe_attr(transformer, attr),
                        model,
                        di,
                    )
                    dataout.append(dout)

            if assert_not_none:
                assert not any(m is None for m in models)

        else:
            models = [None for _ in markers]

            if drop_if_none:
                dataout = [None for _ in markers]
            else:
                dataout = markers

            use_markers = False
            use_groups = False
            use_covariates = False

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

        return params, models, dataout, use_markers, use_groups, use_covariates

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
        assert not any(len(cast("np.ndarray", yi).shape) != 2 for yi in y)
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
            X_marker,
            feature_selection_use_markers,
            feature_selection_use_groups,
            feature_selection_use_covariates
        ) = self._sample_extended_step(
            trial,
            params,
            self.marker_feature_selector,
            markers=X,
            groups=g,
            covariates=c,
            y=y,
        )

        feature_selection_inputs = []
        if feature_selection_use_markers:
            feature_selection_inputs.append([
                mi.X_input for mi in model_paths
            ])

        if feature_selection_use_groups:
            feature_selection_inputs.append([
                mi.grouping_model for mi in model_paths
            ])

        if feature_selection_use_covariates:
            feature_selection_inputs.append([
                mi.covariate_model for mi in model_paths
            ])

        if len(feature_selection_inputs) == 1:
            feature_selection_models = [
                ffmap2(fsm, inp, mi.y_input)
                for fsm, inp, mi
                in zip(
                    feature_selection_models,
                    feature_selection_inputs[0],
                    model_paths
                )
            ]
        else:
            feature_selection_models = [
                ffmap2(fsm, inp, mi.y_input)
                for fsm, inp, mi
                in zip(
                    feature_selection_models,
                    map(list, zip(*feature_selection_inputs)),
                    model_paths
                )
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
            X_dist,
            dist_feature_selection_use_markers,
            dist_feature_selection_use_groups,
            dist_feature_selection_use_covariates
        ) = self._sample_extended_step(
            trial,
            params,
            self.dist_feature_selector,
            markers=X,
            groups=g,
            covariates=c,
            y=y,
        )

        dist_feature_selection_inputs = []
        if dist_feature_selection_use_markers:
            dist_feature_selection_inputs.append([
                mi.X_input for mi in model_paths
            ])

        if dist_feature_selection_use_groups:
            dist_feature_selection_inputs.append([
                mi.grouping_model for mi in model_paths
            ])

        if dist_feature_selection_use_covariates:
            dist_feature_selection_inputs.append([
                mi.covariate_model for mi in model_paths
            ])

        dist_feature_selection_models = [
            ffmap2(fsm, inp, mi.y_input)
            for fsm, inp, mi
            in zip(
                dist_feature_selection_models,
                map(list, zip(*dist_feature_selection_inputs)),
                model_paths
            )
        ]

        params, dist_models, X_dist = self._sample_preprocessing_step(
            trial,
            params,
            self.dist_transformer,
            X_dist,
        )

        dist_models = [
            ffmap(dm, fsm)
            for dm, fsm
            in zip(dist_models, dist_feature_selection_models)
        ]

        (
            params,
            dist_post_models,
            X_dist
        ) = self._sample_preprocessing_step(
            trial,
            params,
            self.dist_post_feature_selector,
            X_dist,
            data2=y,
        )

        for mi, dp, dm in zip(
            model_paths,
            dist_post_models,
            dist_models,
        ):
            mi.dist_model = ffmap2(dp, dm, mi.y_input)

        (
            params,
            nonlinear_feature_selection_models,
            X_nonlinear,
            nonlinear_feature_selection_use_markers,
            nonlinear_feature_selection_use_groups,
            nonlinear_feature_selection_use_covariates
        ) = self._sample_extended_step(
            trial,
            params,
            self.nonlinear_feature_selector,
            markers=X,
            groups=g,
            covariates=c,
            y=y,
        )

        nonlinear_feature_selection_inputs = []
        if nonlinear_feature_selection_use_markers:
            nonlinear_feature_selection_inputs.append([
                mi.X_input for mi in model_paths
            ])

        if nonlinear_feature_selection_use_groups:
            nonlinear_feature_selection_inputs.append([
                mi.grouping_model for mi in model_paths
            ])

        if nonlinear_feature_selection_use_covariates:
            nonlinear_feature_selection_inputs.append([
                mi.covariate_model for mi in model_paths
            ])

        nonlinear_feature_selection_models = [
            ffmap2(fsm, inp, mi.y_input)
            for fsm, inp, mi
            in zip(
                nonlinear_feature_selection_models,
                map(list, zip(*nonlinear_feature_selection_inputs)),
                model_paths
            )
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

        nonlinear_models = [
            ffmap(nlm, fsm)
            for nlm, fsm
            in zip(nonlinear_models, nonlinear_feature_selection_models)
        ]

        (
            params,
            nonlinear_post_models,
            X_nonlinear
        ) = self._sample_preprocessing_step(
            trial,
            params,
            self.nonlinear_post_feature_selector,
            X_nonlinear,
            data2=y,
        )

        for mi, fsm, nlm in zip(
            model_paths,
            nonlinear_post_models,
            nonlinear_models,
        ):
            mi.nonlinear_model = ffmap2(fsm, nlm, mi.y_input)

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

        interactions_models = [
            ffmap(im, mi.joined_model)
            for im, mi
            in zip(interactions_models, model_paths)
        ]

        (
            params,
            interactions_post_models,
            interactions
        ) = self._sample_preprocessing_step(
            trial,
            params,
            self.interactions_post_feature_selector,
            interactions,
            data2=y,
        )
        for mi, fsm, im in zip(
            model_paths,
            interactions_post_models,
            interactions_models
        ):
            mi.interactions_model = ffmap2(fsm, im, mi.y_input)

        data_paths = []
        for cvi, yi, xmi, xdi, xli, ci, gi, ji, ii in zip(
            cv,
            y_,
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
                cv=cvi,
                y=yi,
                X_marker=xmi,
                X_dist=xdi,
                X_nonlinear=xli,
                covariates=ci,
                groups=gi,
                joined=ji,
                interactions=ii,
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

        if self.marker_feature_selector is not None:
            marker_fs = self.marker_feature_selector.model(params)
            if isinstance(marker_fs, MyModel):
                marker_fs_inputs = [
                    i for
                    i in [X_input, g_input, c_input]
                    if i is not None
                ]
                marker_fs = marker_fs(marker_fs_inputs, y_input)
            else:
                marker_fs = ffmap2(
                    marker_fs,
                    X_input,
                    y_input
                )
        else:
            marker_fs = None

        markers = ffmap(
            self.marker_transformer.model(params),
            marker_fs
        )

        if self.dist_feature_selector is not None:
            dist_fs = self.dist_feature_selector.model(params)
            if isinstance(dist_fs, MyModel):
                dist_fs_inputs = [
                    i for
                    i in [X_input, g_input, c_input]
                    if i is not None
                ]
                dist_fs = dist_fs(dist_fs_inputs, y_input)
            else:
                dist_fs = ffmap2(
                    dist_fs,
                    X_input,
                    y_input
                )
        else:
            dist_fs = None

        dists = ffmap(self.dist_transformer.model(params), dist_fs)
        dists = ffmap2(
            self.dist_post_feature_selector.model(
                params,
                markers=X_input,
                groups=g_input,
                covariates=c_input,
                target=y_input,
            ),
            dists,
            y_input
        )

        if self.nonlinear_feature_selector is not None:
            nonlinear_fs = self.nonlinear_feature_selector.model(params)
            if isinstance(nonlinear_fs, MyModel):
                nonlinear_fs_inputs = [
                    i for
                    i in [X_input, g_input, c_input]
                    if i is not None
                ]
                nonlinear_fs = nonlinear_fs(nonlinear_fs_inputs, y_input)
            else:
                nonlinear_fs = ffmap2(
                    nonlinear_fs,
                    X_input,
                    y_input
                )
        else:
            nonlinear_fs = None

        nonlinear = ffmap(
            self.nonlinear_transformer.model(params),
            nonlinear_fs
        )
        nonlinear = ffmap2(
            self.nonlinear_post_feature_selector.model(params),
            nonlinear,
            y_input
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

        interactions = ffmap2(
            self.interactions_post_feature_selector.model(params),
            interactions,
            y_input
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
        inputs = cls._select_inputs(model, data)
        return model.fit(inputs, data.y)

    @classmethod
    def predict(cls, model: "Model", data: "Dataset") -> "np.ndarray":
        return model.predict(cls._select_inputs(model, data))


class SKRunner(BaseRunner):

    predictor: "Model"

    def __init__(
        self,
        task: "Literal['regression', 'ranking', 'ordinal', 'classification']",
        ploidy: int = 2,
        seed: "Optional[int]" = None,
        **kwargs
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

        for data_path in data_paths:
            any_none = False
            for k, v in data_path.items():
                if v is None:
                    continue

                if isinstance(v, np.ndarray):
                    if np.isnan(v).any():
                        print("#### NANs!!!!", params)
                        print("FAILED", k, v)
                        any_none = True
                if isinstance(v, Dataset):
                    for m in ["markers", "y", "groups", "covariates"]:
                        vi = getattr(v, m)
                        if vi is None:
                            continue
                        if np.isnan(vi).any():
                            print("#### NANs!!!!", params)
                            print("FAILED", k, m, vi)
                            any_none = True
            if any_none:
                raise ValueError("Nans")

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


class TFRunner(SKRunner):

    def __init__(
        self,
        task: "Literal['regression', 'ranking', 'classification']",
        ploidy: int = 2,
        seed: "Optional[int]" = None,
        *kwargs,
    ):
        from selectml.optimiser.optimise import OptimiseConvMLP

        if task == "regression":
            objectives: "List[Literal['mse', 'mae', 'binary_crossentropy', 'pairwise']]" = ["mae", "mse"]  # noqa: E501
            target = ["stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
        elif task == "ranking":
            objectives = [
                "mae",
                "mse",
                "pairwise",
            ]
            target = ["passthrough", "stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
        elif task == "ordinal":
            # this doesn't work because of triplet loss etc
            objectives = ["binary_crossentropy"]
            target = ["ordinal"]
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
        elif task == "classification":
            objectives = ["binary_crossentropy"]
            target = ["passthrough"]
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
        else:
            raise ValueError(
                "task must be regression, ranking, "
                "ordinal or classification.")

        self.rng = random.Random(seed)

        # Keras seems to override something to do with deepcopy or mutexes
        # that means we can't cache the feature selector.
        self._init_preprocessors(
            ploidy=ploidy,
            target_options=target,
            group_allow_pca=False,
            marker_fs_options=["passthrough", "maf", "relief", "gemma"],
            marker_options=["passthrough", "maf", "onehot"],
            nonlinear_fs_options=["drop"],
            interactions_options=["drop"],
            dist_post_fs_options=post_fs_options,
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
        from selectml.higher import ffmap2

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
            if all(d is not None for d in markers):
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
                ffmap2(safe_attr(transformer, attr), mi, di)
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
        from selectml.optimiser.wrapper import Make2D

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

        params, predictor_models, yhat = self._sample_tf_step(
            trial,
            params,
            self.predictor,
            markers=[j.get("markers", None) for j in joined],
            dists=[j.get("dists", None) for j in joined],
            groups=[j.get("groups", None) for j in joined],
            covariates=[j.get("covariates", None) for j in joined],
            y=[di.y for di in data_paths],
            predict=True,
            assert_not_none=True,
            drop_if_none=False,
        )

        predictor_models = [
            ffmap2(pm, list(mi.values()), mmi.target_model)
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
        fp = self._model_preprocessing(params, data)

        target = fp.target_model

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


class BGLRSKRunner(SKRunner):

    def __init__(
        self,
        task: "Literal['regression', 'ranking', 'ordinal']",
        ploidy: int = 2,
        seed: "Optional[int]" = None,
        *kwargs,
    ):
        from selectml.optimiser.optimise import OptimiseSKBGLR

        if task == "regression":
            objective: "Literal['gaussian', 'ordinal']" = 'gaussian'
            target = ["stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
        elif task == "ranking":
            objective = 'gaussian'
            target = ["passthrough", "stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
        elif task == "ordinal":
            objective = 'ordinal'
            target = ["passthrough"]
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
        else:
            raise ValueError(
                "task must be regression, ranking, "
                "ordinal.")

        self.rng = random.Random(seed)

        self._init_preprocessors(
            ploidy=ploidy,
            target_options=target,
            marker_fs_options=["passthrough", "maf", "relief", "gemma"],
            dist_post_fs_options=post_fs_options,
            nonlinear_post_fs_options=post_fs_options,
            interactions_post_fs_options=post_fs_options,
        )

        self.predictor = OptimiseSKBGLR(objective=objective)
        return

    def _sample_bglr_step(  # noqa: C901
        self,
        trial: "optuna.Trial",
        params: "Params",
        transformer: "Optional[OptimiseBase]",
        markers: "List[Optional[np.ndarray]]",
        dists: "List[Optional[np.ndarray]]",
        nonlinear: "List[Optional[np.ndarray]]",
        groups: "List[Optional[np.ndarray]]",
        covariates: "List[Optional[np.ndarray]]",
        interactions: "List[Optional[np.ndarray]]",
        y: "Optional[List[Optional[np.ndarray]]]",
        predict: bool = True,
        assert_not_none: bool = True,
        drop_if_none: bool = False,
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
                nonlinear=nonlinear,
                groups=groups,
                covariates=covariates,
                interactions=interactions,
            )
            params.update(trans_params)

            use_markers = f"{self.predictor.name}_markers" in trans_params  # noqa: E501
            use_dists = f"{self.predictor.name}_dists" in trans_params
            use_nonlinear = f"{self.predictor.name}_nonlinear" in trans_params
            use_groups = f"{self.predictor.name}_groups" in trans_params
            use_covariates = f"{self.predictor.name}_covariates" in trans_params  # noqa: E501
            use_interactions = f"{self.predictor.name}_interactions" in trans_params  # noqa: E501

            data: "List[List[np.ndarray]]" = []
            if use_markers and all(d is not None for d in markers):
                data.append(cast("List[np.ndarray]", markers))

            if use_dists:
                assert all(d is not None for d in dists)
                data.append(cast("List[np.ndarray]", dists))

            if use_nonlinear:
                assert all(d is not None for d in groups)
                data.append(cast("List[np.ndarray]", nonlinear))

            if use_groups:
                assert all(d is not None for d in groups)
                data.append(cast("List[np.ndarray]", groups))

            if use_covariates:
                assert all(d is not None for d in covariates)
                data.append(cast("List[np.ndarray]", covariates))

            if use_interactions:
                assert all(d is not None for d in interactions)
                data.append(cast("List[np.ndarray]", interactions))

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
        from selectml.optimiser.wrapper import Make2D

        from selectml.higher import ffmap2, fmap
        (
            params,
            model_paths,
            data_paths
        ) = self._sample_preprocessing(trial, cv)

        params, predictor_models, yhat = self._sample_bglr_step(
            trial,
            params,
            self.predictor,
            markers=[j.X_marker for j in data_paths],
            dists=[j.X_dist for j in data_paths],
            nonlinear=[j.X_nonlinear for j in data_paths],
            covariates=[j.covariates for j in data_paths],
            groups=[j.groups for j in data_paths],
            interactions=[j.interactions for j in data_paths],
            y=[di.y for di in data_paths],
            predict=True,
            assert_not_none=True,
            drop_if_none=False,
        )

        bglr_inputs = []
        if f"{self.predictor.name}_markers" in params:
            bglr_inputs.append([m.marker_model for m in model_paths])
            if any(z is None for z in bglr_inputs[-1]):
                raise ValueError(str(bglr_inputs) + "\n" + str(params))

        if f"{self.predictor.name}_dists" in params:
            bglr_inputs.append([m.dist_model for m in model_paths])
            if any(z is None for z in bglr_inputs[-1]):
                raise ValueError(str(bglr_inputs) + "\n" + str(params))

        if f"{self.predictor.name}_nonlinear" in params:
            bglr_inputs.append([m.nonlinear_model for m in model_paths])
            if any(z is None for z in bglr_inputs[-1]):
                raise ValueError(str(bglr_inputs) + "\n" + str(params))

        if f"{self.predictor.name}_groups" in params:
            bglr_inputs.append([m.grouping_model for m in model_paths])
            if any(z is None for z in bglr_inputs[-1]):
                raise ValueError(str(bglr_inputs) + "\n" + str(params))

        if f"{self.predictor.name}_covariates" in params:
            bglr_inputs.append([m.covariate_model for m in model_paths])
            if any(z is None for z in bglr_inputs[-1]):
                raise ValueError(str(bglr_inputs) + "\n" + str(params))

        if f"{self.predictor.name}_interactions" in params:
            bglr_inputs.append([m.interactions_model for m in model_paths])
            if any(z is None for z in bglr_inputs[-1]):
                raise ValueError(str(bglr_inputs) + "\n" + str(params))

        if len(bglr_inputs) == 1:
            bglr_inputs = bglr_inputs[0]
        else:
            bglr_inputs = list(zip(*bglr_inputs))

        predictor_models = [
            ffmap2(pm, bi, mmi.target_model)
            for pm, bi, mmi
            in zip(predictor_models, bglr_inputs, model_paths)
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
            None
        )

    def model(self, params, data: "Dataset") -> "Model":
        fp = self._model_preprocessing(params, data)

        target = fp.target_model

        model = self.predictor.model(params)
        preprocessed = []
        if "markers" in model.component_names:
            preprocessed.append(fp.marker_model)

        if "dists" in model.component_names:
            preprocessed.append(fp.dist_model)

        if "nonlinear" in model.component_names:
            preprocessed.append(fp.nonlinear_model)

        if "groups" in model.component_names:
            preprocessed.append(fp.grouping_model)

        if "covariates" in model.component_names:
            preprocessed.append(fp.covariate_model)

        if "interactions" in model.component_names:
            preprocessed.append(fp.interactions_model)

        model2 = model(preprocessed, target)
        yhat = fp.target_transformer(
            model2,
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


class BGLRRunner(BaseRunner):

    predictor: "Model"

    def __init__(
        self,
        task: "Literal['regression', 'ranking', 'ordinal', 'classification']",
        ploidy: int = 2,
        seed: "Optional[int]" = None,
        *kwargs,

    ):
        from selectml.optimiser.optimise import OptimiseBGLR

        if task == "regression":
            objective: "Literal['gaussian', 'ordinal']" = 'gaussian'
            target = ["stdnorm", "quantile"]
        elif task == "ranking":
            objective = 'gaussian'
            target = ["passthrough", "stdnorm", "quantile"]
        elif task == "ordinal":
            objective = 'ordinal'
            target = ["passthrough"]
        else:
            raise ValueError(
                "task must be regression, ranking, "
                "ordinal.")

        self.rng = random.Random(seed)

        self._init_bglr_preprocessors(
            target_options=target,
            ploidy=ploidy,
        )

        self.predictor = OptimiseBGLR(objective=objective)
        return

    def _init_bglr_preprocessors(
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
        group_allow_pca: bool = True,
        marker_fs_options: "List[str]" = [
            "passthrough",
            "maf",
            "relief",
            "gemma"
        ],
        marker_options: "List[str]" = [
            "maf",
            "noia_add",
        ],
        add_fs_options: "List[str]" = [
            "drop",
            "passthrough",
            "maf",
            "relief",
            "gemma"
        ],
        dom_fs_options: "List[str]" = [
            "drop",
            "passthrough",
            "maf",
            "relief",
            "gemma"
        ],
        epiadd_fs_options: "List[str]" = [
            "drop",
            "passthrough",
            "maf",
            "relief",
            "gemma"
        ],
        epidom_fs_options: "List[str]" = [
            "passthrough",
            "drop",
            "maf",
            "relief",
            "gemma"
        ],
        epiaddxdom_fs_options: "List[str]" = [
            "passthrough",
            "drop",
            "maf",
            "relief",
            "gemma"
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
            OptimiseAddEpistasis,
            OptimiseDomEpistasis,
            OptimiseAddDomEpistasis,
            OptimiseGrouping,
        )

        self.ploidy = ploidy

        self.target_transformer = OptimiseTarget(options=target_options)

        self.covariate_transformer = OptimiseCovariates(
            options=covariate_options,
            use_polynomial=use_covariate_polynomial
        )

        self.grouping_transformer = OptimiseGrouping(
            allow_pca=group_allow_pca,
            max_ncomponents=50,
            seed=self.rng.getrandbits(32),
        )

        self.marker_feature_selector = OptimiseFeatureSelector(
            options=marker_fs_options,
            name="marker_feature_selector",
            use_cache=use_fs_cache,
            seed=self.rng.getrandbits(32),
        )
        self.marker_transformer = OptimiseMarkerTransformer(
            options=marker_options,
            ploidy=ploidy,
            max_ncomponents=200
        )

        self.add_feature_selector = OptimiseFeatureSelector(
            options=add_fs_options,
            name="additive_feature_selector",
            use_cache=use_fs_cache,
            seed=self.rng.getrandbits(32),
        )
        self.add_transformer = OptimiseDistTransformer(
            ploidy=ploidy,
            options=["noia_additive"],
            name="additive_dist",
        )

        self.dom_feature_selector = OptimiseFeatureSelector(
            options=dom_fs_options,
            name="dominance_feature_selector",
            use_cache=use_fs_cache,
            seed=self.rng.getrandbits(32),
        )
        self.dom_transformer = OptimiseDistTransformer(
            ploidy=ploidy,
            options=["noia_dominance"],
            name="dominance_dist",
        )

        self.epiadd_feature_selector = OptimiseFeatureSelector(
            options=epiadd_fs_options,
            name="additive_epistasis_feature_selector",
            use_cache=use_fs_cache,
            seed=self.rng.getrandbits(32),
        )
        self.epiadd_transformer = OptimiseAddEpistasis()

        self.epidom_feature_selector = OptimiseFeatureSelector(
            options=epidom_fs_options,
            name="dominance_epistasis_feature_selector",
            use_cache=use_fs_cache,
            seed=self.rng.getrandbits(32),
        )
        self.epidom_transformer = OptimiseDomEpistasis()

        self.epiaddxdom_feature_selector = OptimiseFeatureSelector(
            options=epiaddxdom_fs_options,
            name="addxdom_epistasis_feature_selector",
            use_cache=use_fs_cache,
            seed=self.rng.getrandbits(32),
        )
        self.epiaddxdom_transformer = OptimiseAddDomEpistasis()
        return

    def _sample_feature_selector(
        self,
        trial,
        params,
        transformer,
        markers,
        groups,
        covariates,
        y,
        model_paths,
    ):
        from selectml.higher import ffmap2
        (
            params,
            feature_selection_models,
            markers,
            feature_selection_use_markers,
            feature_selection_use_groups,
            feature_selection_use_covariates
        ) = self._sample_extended_step(
            trial,
            params,
            transformer,
            markers=markers,
            groups=groups,
            covariates=covariates,
            y=y,
        )

        feature_selection_inputs = []
        if feature_selection_use_markers:
            feature_selection_inputs.append([
                mi.X_input for mi in model_paths
            ])

        if feature_selection_use_groups:
            feature_selection_inputs.append([
                mi.grouping_model for mi in model_paths
            ])

        if feature_selection_use_covariates:
            feature_selection_inputs.append([
                mi.covariate_model for mi in model_paths
            ])

        if len(feature_selection_inputs) == 1:
            feature_selection_models = [
                ffmap2(fsm, inp, mi.y_input)
                for fsm, inp, mi
                in zip(
                    feature_selection_models,
                    feature_selection_inputs[0],
                    model_paths
                )
            ]
        else:
            feature_selection_models = [
                ffmap2(fsm, inp, mi.y_input)
                for fsm, inp, mi
                in zip(
                    feature_selection_models,
                    map(list, zip(*feature_selection_inputs)),
                    model_paths
                )
            ]
        return params, markers, feature_selection_models

    def _sample_preprocessing(  # noqa
        self,
        trial: "optuna.Trial",
        cv: "Sequence[Dataset]",
    ):

        from selectml.higher import ffmap
        from baikal import Input

        params: "Params" = {}
        X: "List[Optional[np.ndarray]]" = [cvi.markers for cvi in cv]
        assert not any(xi is None for xi in X)
        y: "List[Optional[np.ndarray]]" = [cvi.y for cvi in cv]

        assert not any(yi is None for yi in y)
        assert not any(len(cast("np.ndarray", yi).shape) != 2 for yi in y)
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

        params, X_marker, feature_selection_models = self._sample_feature_selector(  # noqa
            trial,
            params,
            self.marker_feature_selector,
            X,
            g,
            c,
            y,
            model_paths
        )

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

        params, X_add, add_feature_selection_models = self._sample_feature_selector(  # noqa
            trial,
            params,
            self.add_feature_selector,
            X,
            g,
            c,
            y,
            model_paths
        )

        params, add_models, X_add = self._sample_preprocessing_step(
            trial,
            params,
            self.add_transformer,
            X_add,
        )

        for mi, mm, fs in zip(
            model_paths,
            add_models,
            add_feature_selection_models
        ):
            mi.additive_model = ffmap(mm, fs)

        params, X_dom, dom_feature_selection_models = self._sample_feature_selector(  # noqa
            trial,
            params,
            self.dom_feature_selector,
            X,
            g,
            c,
            y,
            model_paths
        )

        params, dom_models, X_dom = self._sample_preprocessing_step(
            trial,
            params,
            self.dom_transformer,
            X_dom,
        )

        for mi, mm, fs in zip(
            model_paths,
            dom_models,
            dom_feature_selection_models
        ):
            mi.dominance_model = ffmap(mm, fs)

        params, X_epiadd, epiadd_feature_selection_models = self._sample_feature_selector(  # noqa
            trial,
            params,
            self.epiadd_feature_selector,
            X,
            g,
            c,
            y,
            model_paths
        )

        params, epiadd_models, X_epiadd = self._sample_preprocessing_step(
            trial,
            params,
            self.epiadd_transformer,
            X_epiadd,
        )

        for mi, mm, fs in zip(
            model_paths,
            epiadd_models,
            epiadd_feature_selection_models
        ):
            mi.epistasis_add_model = ffmap(mm, fs)

        params, X_epidom, epidom_feature_selection_models = self._sample_feature_selector(  # noqa
            trial,
            params,
            self.epidom_feature_selector,
            X,
            g,
            c,
            y,
            model_paths
        )

        params, epidom_models, X_epidom = self._sample_preprocessing_step(
            trial,
            params,
            self.epidom_transformer,
            X_epidom,
        )

        for mi, mm, fs in zip(
            model_paths,
            epidom_models,
            epidom_feature_selection_models
        ):
            mi.epistasis_dom_model = ffmap(mm, fs)

        params, X_epiaddxdom, epiaddxdom_feature_selection_models = self._sample_feature_selector(  # noqa
            trial,
            params,
            self.epiaddxdom_feature_selector,
            X,
            g,
            c,
            y,
            model_paths
        )

        params, epiaddxdom_models, X_epiaddxdom = self._sample_preprocessing_step(  # noqa
            trial,
            params,
            self.epiaddxdom_transformer,
            X_epiaddxdom,
        )

        for mi, mm, fs in zip(
            model_paths,
            epiaddxdom_models,
            epiaddxdom_feature_selection_models
        ):
            mi.epistasis_addxdom_model = ffmap(mm, fs)

        data_paths = []
        for cvi, yi, xmi, xai, xdi, xeai, xedi, xeadi, ci, gi in zip(
            cv,
            y_,
            X_marker,
            X_add,
            X_dom,
            X_epiadd,
            X_epidom,
            X_epiaddxdom,
            c,
            g
        ):
            assert yi is not None
            data_paths.append(DataPath(
                cv=cvi,
                y=yi,
                X_marker=xmi,
                X_add=xai,
                X_dom=xdi,
                X_epiadd=xeai,
                X_epidom=xedi,
                X_epiaddxdom=xeadi,
                covariates=ci,
                groups=gi,
            ))

        return params, model_paths, data_paths

    def _model_preprocessing(  # noqa: C901
        self,
        params: "Params",
        data: "Dataset",
        **kwargs
    ):
        from baikal import Input
        from selectml.higher import ffmap, ffmap2

        X_input = Input(name="markers")
        y_input = Input(name="y")
        g_input = None if data.groups is None else Input(name="groups")
        c_input = None if data.covariates is None else Input(name="covariates")

        target_transformer = self.target_transformer.model(params)
        target = ffmap(target_transformer, y_input)

        covariates = ffmap(self.covariate_transformer.model(params), c_input)
        groups = ffmap(self.grouping_transformer.model(params), g_input)

        if self.marker_feature_selector is not None:
            marker_fs = self.marker_feature_selector.model(params)
            if isinstance(marker_fs, MyModel):
                marker_fs_inputs = [
                    i for
                    i in [X_input, g_input, c_input]
                    if i is not None
                ]
                marker_fs = marker_fs(marker_fs_inputs, y_input)
            else:
                marker_fs = ffmap2(
                    marker_fs,
                    X_input,
                    y_input
                )
        else:
            marker_fs = None

        markers = ffmap(
            self.marker_transformer.model(params),
            marker_fs
        )

        if self.add_feature_selector is not None:
            add_fs = self.add_feature_selector.model(params)
            if isinstance(add_fs, MyModel):
                add_fs_inputs = [
                    i for
                    i in [X_input, g_input, c_input]
                    if i is not None
                ]
                add_fs = add_fs(add_fs_inputs, y_input)
            else:
                add_fs = ffmap2(
                    add_fs,
                    X_input,
                    y_input
                )
        else:
            add_fs = None

        adds = ffmap(self.add_transformer.model(params), add_fs)

        if self.dom_feature_selector is not None:
            dom_fs = self.dom_feature_selector.model(params)
            if isinstance(dom_fs, MyModel):
                dom_fs_inputs = [
                    i for
                    i in [X_input, g_input, c_input]
                    if i is not None
                ]
                dom_fs = dom_fs(dom_fs_inputs, y_input)
            else:
                dom_fs = ffmap2(
                    dom_fs,
                    X_input,
                    y_input
                )
        else:
            dom_fs = None

        doms = ffmap(
            self.dom_transformer.model(params),
            dom_fs
        )

        if self.epiadd_feature_selector is not None:
            epiadd_fs = self.epiadd_feature_selector.model(params)
            if isinstance(epiadd_fs, MyModel):
                epiadd_fs_inputs = [
                    i for
                    i in [X_input, g_input, c_input]
                    if i is not None
                ]
                epiadd_fs = epiadd_fs(epiadd_fs_inputs, y_input)
            else:
                epiadd_fs = ffmap2(
                    epiadd_fs,
                    X_input,
                    y_input
                )
        else:
            epiadd_fs = None

        epiadds = ffmap(
            self.epiadd_transformer.model(params),
            epiadd_fs
        )

        if self.epidom_feature_selector is not None:
            epidom_fs = self.epidom_feature_selector.model(params)
            if isinstance(epidom_fs, MyModel):
                epidom_fs_inputs = [
                    i for
                    i in [X_input, g_input, c_input]
                    if i is not None
                ]
                epidom_fs = epidom_fs(epidom_fs_inputs, y_input)
            else:
                epidom_fs = ffmap2(
                    epidom_fs,
                    X_input,
                    y_input
                )
        else:
            epidom_fs = None

        epidoms = ffmap(
            self.epidom_transformer.model(params),
            epidom_fs
        )

        if self.epiaddxdom_feature_selector is not None:
            epiaddxdom_fs = self.epiaddxdom_feature_selector.model(params)
            if isinstance(epiaddxdom_fs, MyModel):
                epiaddxdom_fs_inputs = [
                    i for
                    i in [X_input, g_input, c_input]
                    if i is not None
                ]
                epiaddxdom_fs = epiaddxdom_fs(epiaddxdom_fs_inputs, y_input)
            else:
                epiaddxdom_fs = ffmap2(
                    epiaddxdom_fs,
                    X_input,
                    y_input
                )
        else:
            epiaddxdom_fs = None

        epiaddxdoms = ffmap(
            self.epiaddxdom_transformer.model(params),
            epiaddxdom_fs
        )

        return ModelPath(
            X_input=X_input,
            y_input=y_input,
            g_input=g_input,
            c_input=c_input,
            target_transformer=target_transformer,
            target_model=target,
            grouping_model=groups,
            covariate_model=covariates,
            marker_model=markers,
            additive_model=adds,
            dominance_model=doms,
            epistasis_add_model=epiadds,
            epistasis_dom_model=epidoms,
            epistasis_addxdom_model=epiaddxdoms,
        )

    def _sample_bglr_step(  # noqa: C901
        self,
        trial: "optuna.Trial",
        params: "Params",
        transformer: "Optional[OptimiseBase]",
        markers: "List[Optional[np.ndarray]]",
        adds: "List[Optional[np.ndarray]]",
        doms: "List[Optional[np.ndarray]]",
        epiadds: "List[Optional[np.ndarray]]",
        epidoms: "List[Optional[np.ndarray]]",
        epiaddxdoms: "List[Optional[np.ndarray]]",
        groups: "List[Optional[np.ndarray]]",
        covariates: "List[Optional[np.ndarray]]",
        y: "Optional[List[Optional[np.ndarray]]]",
        predict: bool = True,
        assert_not_none: bool = True,
        drop_if_none: bool = False,
    ) -> "Tuple[Params, List[Model], List[Optional[np.ndarray]]]":
        from copy import copy
        from itertools import cycle
        from selectml.higher import ffmap

        params = copy(params)

        if (transformer is not None):
            trans_params = transformer.sample(
                trial,
                Xs=markers,
                adds=adds,
                doms=doms,
                epiadds=epiadds,
                epidoms=epidoms,
                epiaddxdoms=epiaddxdoms,
                groups=groups,
                covariates=covariates,
            )
            params.update(trans_params)

            use_markers = f"{transformer.name}_markers" in params  # noqa: E501
            use_add = f"{transformer.name}_add" in params
            use_dom = f"{transformer.name}_dom" in params
            use_epiadd = f"{transformer.name}_epiadd" in params
            use_epidom = f"{transformer.name}_epidom" in params
            use_epiaddxdom = f"{transformer.name}_epiaddxdom" in params  # noqa: E501
            use_groups = f"{transformer.name}_groups" in params
            use_covariates = f"{transformer.name}_covariates" in params  # noqa: E501

            data: "List[List[np.ndarray]]" = []
            if use_markers and all(d is not None for d in markers):
                data.append(cast("List[np.ndarray]", markers))

            if use_add:
                assert all(d is not None for d in adds)
                data.append(cast("List[np.ndarray]", adds))

            if use_dom:
                assert all(d is not None for d in doms)
                data.append(cast("List[np.ndarray]", doms))

            if use_epiadd:
                assert all(d is not None for d in epiadds)
                data.append(cast("List[np.ndarray]", epiadds))

            if use_epidom:
                assert all(d is not None for d in epidoms)
                data.append(cast("List[np.ndarray]", epidoms))

            if use_epiaddxdom:
                assert all(d is not None for d in epiaddxdoms)
                data.append(cast("List[np.ndarray]", epiaddxdoms))

            if use_groups:
                assert all(d is not None for d in groups)
                data.append(cast("List[np.ndarray]", groups))

            if use_covariates:
                assert all(d is not None for d in covariates)
                data.append(cast("List[np.ndarray]", covariates))

            if len(data) == 1:
                data_: "Union[List[np.ndarray], List[List[np.ndarray]]]" = data[0]  # noqa: E501
            elif len(data) > 1:
                data_ = list(map(lambda x: list(x), zip(*data)))
            else:
                raise ValueError(f"didn't get enough data, {data} \n {params}")

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

        Xs = [d.X_marker for d in data_paths]
        adds = [d.X_add for d in data_paths]
        doms = [d.X_dom for d in data_paths]
        epiadds = [d.X_epiadd for d in data_paths]
        epidoms = [d.X_epidom for d in data_paths]
        epiaddxdoms = [d.X_epiaddxdom for d in data_paths]
        groups = [d.groups for d in data_paths]
        covariates = [d.covariates for d in data_paths]
        y = [d.y for d in data_paths]

        params, predictor_models, yhat = self._sample_bglr_step(
            trial=trial,
            params=params,
            transformer=self.predictor,
            markers=Xs,
            adds=adds,
            doms=doms,
            epiadds=epiadds,
            epidoms=epidoms,
            epiaddxdoms=epiaddxdoms,
            y=y,
            groups=groups,
            covariates=covariates,
            predict=True
        )

        bglr_inputs = []
        if f"{self.predictor.name}_markers" in params:
            bglr_inputs.append([m.marker_model for m in model_paths])

        if f"{self.predictor.name}_add" in params:
            bglr_inputs.append([m.additive_model for m in model_paths])

        if f"{self.predictor.name}_dom" in params:
            bglr_inputs.append([m.dominance_model for m in model_paths])

        if f"{self.predictor.name}_epiadd" in params:
            bglr_inputs.append([m.epistasis_add_model for m in model_paths])

        if f"{self.predictor.name}_epidom" in params:
            bglr_inputs.append([m.epistasis_dom_model for m in model_paths])

        if f"{self.predictor.name}_epiaddxdom" in params:
            bglr_inputs.append([
                m.epistasis_addxdom_model
                for m
                in model_paths
            ])

        if f"{self.predictor.name}_groups" in params:
            bglr_inputs.append([m.grouping_model for m in model_paths])

        if f"{self.predictor.name}_covariates" in params:
            bglr_inputs.append([m.covariate_model for m in model_paths])

        if len(bglr_inputs) == 1:
            bglr_inputs = bglr_inputs[0]
        else:
            bglr_inputs = list(zip(*bglr_inputs))

        predictor_models = [
            ffmap2(pm, mi, Make1D()(mmi.target_model))
            for pm, mi, mmi
            in zip(predictor_models, bglr_inputs, model_paths)
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
                for a in [
                    "marker_model", "additive_model",
                    "dominance_model", "epistasis_add_model",
                    "epistasis_dom_model", "epistasis_addxdom_model",
                ]]
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
            None
        )

    def model(self, params, data: "Dataset") -> "Model":
        fp = self._model_preprocessing(params, data)

        target = fp.target_model

        model = self.predictor.model(params)
        preprocessed = []
        if "markers" in model.component_names:
            preprocessed.append(fp.marker_model)

        if "add" in model.component_names:
            preprocessed.append(fp.additive_model)

        if "dom" in model.component_names:
            preprocessed.append(fp.dominance_model)

        if "epiadd" in model.component_names:
            preprocessed.append(fp.epistasis_add_model)

        if "epidom" in model.component_names:
            preprocessed.append(fp.epistasis_dom_model)

        if "epiaddxdom" in model.component_names:
            preprocessed.append(fp.epistasis_addxdom_model)

        if "groups" in model.component_names:
            preprocessed.append(fp.grouping_model)

        if "covariates" in model.component_names:
            preprocessed.append(fp.covariate_model)

        model2 = model(preprocessed, target)
        yhat = fp.target_transformer(
            model2,
            compute_func="inverse_transform",
            trainable=False
        )

        any_x = any([
            getattr(fp, a) is not None
            for a in [
                "marker_model", "additive_model", "dominance_model",
                "epistasis_add_model", "epistasis_dom_model",
                "epistasis_addxdom_model"
            ]]
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
        ploidy: int = 2,
        seed: "Optional[int]" = None,
        *kwargs,

    ):
        from selectml.optimiser.optimise import OptimiseXGB

        if task == "regression":
            objectives = ["reg:squarederror"]
            target = ["passthrough", "stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
        elif task == "ranking":
            objectives = [
                "reg:squarederror",
                "rank:pairwise",
            ]
            target = ["passthrough", "stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
        elif task == "ordinal":
            objectives = ["binary:logistic"]
            target = ["ordinal"]
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
        elif task == "classification":
            objectives = ["reg:logistic", "binary:logistic"]
            target = ["passthrough"]
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
        else:
            raise ValueError(
                "task must be regression, ranking, "
                "ordinal, or classification.")

        self.rng = random.Random(seed)

        self._init_preprocessors(
            ploidy=ploidy,
            target_options=target,
            marker_fs_options=["passthrough", "maf", "relief", "gemma"],
            nonlinear_fs_options=["drop"],
            interactions_options=["drop"],
            dist_post_fs_options=post_fs_options,
            use_covariate_polynomial=False,
        )

        self.predictor = OptimiseXGB(objectives=objectives)
        return


class NGBRunner(SKRunner):

    def __init__(
        self,
        task: "Literal['regression', 'ranking', 'ordinal', 'classification']",
        ploidy: int = 2,
        seed: "Optional[int]" = None,
        *kwargs,

    ):
        from .optimise import OptimiseNGB

        if task == "regression":
            objectives = ["normal"]
            target = ["passthrough", "stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
        elif task == "ranking":
            objectives = ["normal"]
            target = ["passthrough", "stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
        elif task == "ordinal":
            objectives = ["bernoulli"]
            target = ["ordinal"]
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
        elif task == "classification":
            objectives = ["bernoulli"]
            target = ["passthrough"]
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
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
            dist_post_fs_options=post_fs_options,
            use_covariate_polynomial=False,
        )

        self.rng = random.Random(seed)

        self.predictor = OptimiseNGB(distribution=objectives)
        return


class SVMRunner(SKRunner):

    def __init__(
        self,
        task: "Literal['regression', 'ranking', 'ordinal', 'classification']",
        ploidy: int = 2,
        seed: "Optional[int]" = None,
        *kwargs,

    ):
        from .optimise import OptimiseSVM

        if task == "regression":
            objectives = ["epsilon_insensitive", "squared_epsilon_insensitive"]
            target = ["passthrough", "stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
        elif task == "ranking":
            objectives = ["epsilon_insensitive", "squared_epsilon_insensitive"]
            target = ["passthrough", "stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
        elif task == "ordinal":
            objectives = ["hinge", "squared_hinge"]
            target = ["passthrough"]
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
        elif task == "classification":
            objectives = ["hinge", "squared_hinge"]
            target = ["passthrough"]
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
        else:
            raise ValueError(
                "task must be regression, ranking, "
                "ordinal, or classification.")

        self.rng = random.Random(seed)

        self._init_preprocessors(
            ploidy=ploidy,
            target_options=target,
            marker_fs_options=["passthrough", "maf", "relief", "gemma"],
            dist_post_fs_options=post_fs_options,
            nonlinear_post_fs_options=post_fs_options,
            interactions_post_fs_options=post_fs_options,
            use_covariate_polynomial=True,
        )

        self.predictor = OptimiseSVM(loss=objectives)
        return


class SGDRunner(SKRunner):

    def __init__(
        self,
        task: "Literal['regression', 'ranking', 'ordinal', 'classification']",
        ploidy: int = 2,
        seed: "Optional[int]" = None,
        *kwargs,

    ):
        from .optimise import OptimiseSGD

        if task == "regression":
            objectives = [
                "squared_error",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive",
            ]
            target = ["passthrough", "stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
        elif task == "ranking":
            objectives = [
                "squared_error",
                "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive",
            ]
            target = ["passthrough", "stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
        elif task == "ordinal":
            objectives = [
                "hinge",
                "log",
                "modified_huber",
                "squared_hinge",
                "perceptron",
            ]
            target = ["ordinal"]
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
        elif task == "classification":
            objectives = [
                "hinge",
                "log",
                "modified_huber",
                "squared_hinge",
                "perceptron",
            ]
            target = ["passthrough"]
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
        else:
            raise ValueError(
                "task must be regression, ranking, "
                "ordinal, or classification.")

        self.rng = random.Random(seed)

        self._init_preprocessors(
            ploidy=ploidy,
            target_options=target,
            marker_fs_options=["passthrough", "maf", "relief", "gemma"],
            dist_post_fs_options=post_fs_options,
            nonlinear_post_fs_options=post_fs_options,
            interactions_post_fs_options=post_fs_options,
            use_covariate_polynomial=True,
        )

        self.predictor = OptimiseSGD(loss=objectives)
        return


class LarsRunner(SKRunner):

    def __init__(
        self,
        task: "Literal['regression', 'ranking']",
        ploidy: int = 2,
        seed: "Optional[int]" = None,
        *kwargs,

    ):
        from .optimise import OptimiseLars

        post_fs_options = [
            "passthrough",
            "f_regression",
            "mutual_info_regression"
        ]

        self.rng = random.Random(seed)

        self._init_preprocessors(
            ploidy=ploidy,
            target_options=["passthrough", "stdnorm", "quantile"],
            marker_fs_options=["passthrough", "maf", "relief", "gemma"],
            dist_post_fs_options=post_fs_options,
            nonlinear_post_fs_options=post_fs_options,
            interactions_post_fs_options=post_fs_options,
            use_covariate_polynomial=True,
        )

        self.predictor = OptimiseLars()
        return


class RFRunner(SKRunner):

    def __init__(
        self,
        task: "Literal['regression', 'ranking', 'ordinal', 'classification']",
        ploidy: int = 2,
        seed: "Optional[int]" = None,
        *kwargs,

    ):
        from .optimise import OptimiseRF

        self.task = task
        if task in ("regression", "ranking"):
            target = ["passthrough", "stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
            criterion_options = [
                "squared_error",
                "absolute_error",
            ]
        else:
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
            criterion_options = [
                "gini",
                "entropy"
            ]
            if task == "ordinal":
                target = ["ordinal"]
            else:
                target = ["passthrough"]

        self.rng = random.Random(seed)

        self._init_preprocessors(
            ploidy=ploidy,
            target_options=target,
            dist_post_fs_options=post_fs_options,
            marker_fs_options=["passthrough", "maf", "relief", "gemma"],
            nonlinear_fs_options=["drop"],
            interactions_options=["drop"],
            use_covariate_polynomial=True,
        )

        self.predictor = OptimiseRF(criterion=criterion_options)
        return


class KNNRunner(SKRunner):

    def __init__(
        self,
        task: "Literal['regression', 'ranking', 'ordinal', 'classification']",
        ploidy: int = 2,
        seed: "Optional[int]" = None,
        *kwargs,

    ):
        from .optimise import OptimiseKNN

        if task in ("regression", "ranking"):
            objective: "Literal['classification', 'regression']" = "regression"
            target = ["passthrough", "stdnorm", "quantile"]
            post_fs_options = [
                "passthrough",
                "f_regression",
                "mutual_info_regression"
            ]
        elif task == "ordinal":
            objective = "classification"
            target = ["ordinal"]
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
        elif task == "classification":
            objective = "classification"
            target = ["passthrough"]
            post_fs_options = [
                "passthrough",
                "f_classif",
                "mutual_info_classif"
            ]
        else:
            raise ValueError(
                "task must be regression, ranking, "
                "ordinal, or classification.")

        self.rng = random.Random(seed)

        self._init_preprocessors(
            ploidy=ploidy,
            target_options=target,
            dist_post_fs_options=post_fs_options,
            nonlinear_post_fs_options=post_fs_options,
            interactions_post_fs_options=post_fs_options,
            use_covariate_polynomial=True,
        )

        self.predictor = OptimiseKNN(objective=objective)
        return
