# This is a reimplementation of scikeras to allow multi-input and output models

import inspect
import os
from types import FunctionType
import random
from contextlib import contextmanager
from collections import defaultdict

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_sample_weight, check_array
from sklearn.exceptions import NotFittedError

import tensorflow as tf
from tensorflow.keras import optimizers as optimizers_mod
from tensorflow.keras import metrics as metrics_mod
from tensorflow.keras import losses as losses_mod
from tensorflow.python.eager import context
from tensorflow.python.framework import config, ops
from tensorflow.keras.losses import get as keras_loss_get
from tensorflow.keras.metrics import get as keras_metric_get

from typing import TYPE_CHECKING
from typing import cast
if TYPE_CHECKING:
    from typing import Dict, List, Sequence, Tuple
    from typing import Optional, Union, Any
    from typing import Type
    from typing import Callable
    from typing import Iterable, Generator
    from typing import Literal
    import numpy.typing as npt

DIGITS = frozenset(str(i) for i in range(10))


@contextmanager
def tensorflow_random_state(seed: int) -> "Generator[None, None, None]":
    # Save values
    origin_gpu_det = os.environ.get("TF_DETERMINISTIC_OPS", None)
    orig_random_state = random.getstate()
    orig_np_random_state = np.random.get_state()
    if context.executing_eagerly():
        tf_random_seed = context.global_seed()
    else:
        tf_random_seed = ops.get_default_graph().seed

    determism_enabled = config.is_op_determinism_enabled()
    config.enable_op_determinism()

    # Set values
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    yield

    # Reset values
    if origin_gpu_det is not None:
        os.environ["TF_DETERMINISTIC_OPS"] = origin_gpu_det
    else:
        os.environ.pop("TF_DETERMINISTIC_OPS")
    random.setstate(orig_random_state)
    np.random.set_state(orig_np_random_state)
    tf.random.set_seed(tf_random_seed)
    if not determism_enabled:
        config.disable_op_determinism()


def metric_name(
    metric: "Union[str, metrics_mod.Metric, Callable]"
) -> str:
    """Retrieves a metric's full name (eg: "mean_squared_error").
    Parameters
    ----------
    metrics : Union[str, Metric, Callable]
        Instance of Keras Metric, metric callable or string
        shothand (eg: "mse") or full name ("mean_squared_error").
    Returns
    -------
    str
        Full name for Keras metric. Ex: "mean_squared_error".
    Notes
    -----
    The result of this function will always be in snake case, not camel case.
    Examples
    --------
    >>> metric_name("BinaryCrossentropy")
    'BinaryCrossentropy'
    >>> metric_name("binary_crossentropy")
    'binary_crossentropy'
    >>> import tensorflow.keras.metrics as metrics
    >>> metric_name(metrics.BinaryCrossentropy)
    'BinaryCrossentropy'
    >>> metric_name(metrics.binary_crossentropy)
    'binary_crossentropy'
    Raises
    ------
    TypeError
        If metric is not a string, a tf.keras.metrics.Metric instance a class
        inheriting from tf.keras.metrics.Metric.
    """
    if inspect.isclass(metric):
        metric = metric()  # get_metric accepts instances, not classes
    if not (isinstance(metric, (str, metrics_mod.Metric)) or callable(metric)):
        raise TypeError(
            "``metric`` must be a string, a function, an instance of"
            " ``tf.keras.metrics.Metric`` or a type inheriting from"
            " ``tf.keras.metrics.Metric``"
        )
    fn_or_cls = keras_metric_get(metric)
    if isinstance(fn_or_cls, metrics_mod.Metric):
        return _camel2snake(fn_or_cls.__class__.__name__)
    return fn_or_cls.__name__r


def loss_name(loss: "Union[str, losses_mod.Loss, Callable]") -> str:
    """Retrieves a loss's full name (eg: "mean_squared_error").
    Parameters
    ----------
    loss : Union[str, Loss, Callable]
        Instance of Keras Loss, loss callable or string
        shorthand (eg: "mse") or full name ("mean_squared_error").
    Returns
    -------
    str
        String name of the loss.
    Notes
    -----
    The result of this function will always be in snake case, not camel case.
    Examples
    --------
    >>> loss_name("BinaryCrossentropy")
    'binary_crossentropy'
    >>> loss_name("binary_crossentropy")
    'binary_crossentropy'
    >>> import tensorflow.keras.losses as losses
    >>> loss_name(losses.BinaryCrossentropy)
    'binary_crossentropy'
    >>> loss_name(losses.binary_crossentropy)
    'binary_crossentropy'
    Raises
    ------
    TypeError
        If loss is not a string, tf.keras.losses.Loss instance or a callable.
    """
    if inspect.isclass(loss):
        loss = loss()
    if not (isinstance(loss, (str, losses_mod.Loss)) or callable(loss)):
        raise TypeError(
            "``loss`` must be a string, a function, an instance of "
            "``tf.keras.losses.Loss`` or a type inheriting from "
            "``tf.keras.losses.Loss``"
        )
    fn_or_cls = keras_loss_get(loss)
    if isinstance(fn_or_cls, losses_mod.Loss):
        return _camel2snake(fn_or_cls.__class__.__name__)
    return fn_or_cls.__name__


def _camel2snake(s: str) -> str:
    """from [1]
    [1]:https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    """
    return "".join([
        "_" + c.lower() if c.isupper() else c
        for c in s
    ]).lstrip("_")


class TFBase(BaseEstimator):

    """Implementation of the scikit-learn classifier API for Keras.
    Below are a list of SciKeras specific parameters.
    For details on other parameters, please see the see the
    [tf.keras.Model documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Model).  # noqa: E501

    Parameters
    ----------
    model : Union[None, Callable[..., tf.keras.Model], tf.keras.Model]
        default None
        Used to build the Keras Model. When called,
        must return a compiled instance of a Keras Model
        to be used by `fit`, `predict`, etc.
        If None, you must implement ``_keras_build_fn``.
    optimizer : Union[str, tf.keras.optimizers.Optimizer, Type[tf.keras.optimizers.Optimizer]]
        default "adam"
        This can be a string for Keras' built in optimizers,
        an instance of tf.keras.optimizers.Optimizer
        or a class inheriting from tf.keras.optimizers.Optimizer.
        Only strings and classes support parameter routing.
    loss : Union[Union[str, tf.keras.losses.Loss, Type[tf.keras.losses.Loss], Callable], None],
        default None
        The loss function to use for training.
        This can be a string for Keras' built in losses,
        an instance of tf.keras.losses.Loss
        or a class inheriting from tf.keras.losses.Loss .
        Only strings and classes support parameter routing.
    random_state : Union[int, np.random.RandomState, None]
        default None
        Set the Tensorflow random number generators to a
        reproducible deterministic state using this seed.
        Pass an int for reproducible results across multiple
        function calls.
    warm_start : bool
        default False
        If True, subsequent calls to fit will _not_ reset
        the model parameters but *will* reset the epoch to zero.
        If False, subsequent fit calls will reset the entire model.
        This has no impact on partial_fit, which always trains
        for a single epoch starting from the current epoch.
    batch_size : Union[int, None], default None
        Number of samples per gradient update.
        This will be applied to both `fit` and `predict`. To specify different
        numbers, pass `fit__batch_size=32` and `predict__batch_size=1000`
        (for example). To auto-adjust the batch size to use all samples, pass
        `batch_size=-1`.

    Attributes
    ----------
    model_ : tf.keras.Model
        The instantiated and compiled Keras Model. For pre-built models, this
        will just be a reference to the passed Model instance.
    history_ : Dict[str, List[Any]]
        Dictionary of the format
        ``{metric_str_name: [epoch_0_data, epoch_1_data, ..., epoch_n_data]}``.
    initialized_ : bool
        True if this estimator has been initialized (i.e. predict can be
        called upon it). Note that this does not guarantee that the model
        is "fitted": if ``BaseWrapper.initialize`` was called instead of fit
        the model wil likely have random weights.
    target_encoder_ : sklearn-transformer
        Transformer used to pre/post process the target y.
    feature_encoder_ : sklearn-transformer
        Transformer used to pre/post process the features/input X.
    n_outputs_expected_ : int
        The number of outputs the Keras Model is expected to have, as
        determined by ``target_transformer_``.
    target_type_ : str
        One of:
        * 'continuous': y is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': y is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': y contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': y contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': y is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': y is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': y is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.
    y_shape_ : Tuple[int]
        Shape of the target y that the estimator was fitted on.
    y_dtype_ : np.dtype
        Dtype of the target y that the estimator was fitted on.
    X_shape_ : Tuple[int]
        Shape of the input X that the estimator was fitted on.
    X_dtype_ : np.dtype
        Dtype of the input X that the estimator was fitted on.
    n_features_in_ : int
        The number of features seen during `fit`.
    """

    _tags = {
        "poor_score": True,
        "multioutput": True,
    }

    _fit_kwargs = {
        # parameters destined to keras.Model.fit
        "batch_size",
        "epochs",
        "verbose",
        "validation_split",
        "shuffle",
        "class_weight",
        "sample_weight",
        "initial_epoch",
        "validation_steps",
        "validation_batch_size",
        "validation_freq",
    }

    _predict_kwargs = {
        # parameters destined to keras.Model.predict
        "batch_size",
        "verbose",
        "steps",
    }

    _compile_kwargs = {
        # parameters destined to keras.Model.compile
        "optimizer",
        "loss",
        "metrics",
        "loss_weights",
        "weighted_metrics",
        "run_eagerly",
    }

    _wrapper_params = {
        # parameters consumed by the wrappers themselves
        "warm_start",
        "random_state",
    }

    _routing_prefixes = {
        "model",
        "fit",
        "compile",
        "predict",
        "optimizer",
        "loss",
        "metrics",
    }

    def __init__(
        self,
        model: "Union[None, Callable[..., tf.keras.Model], tf.keras.Model]" = None,  # noqa: E501
        *,
        warm_start: bool = False,
        random_state: "Union[int, np.random.RandomState, None]" = None,
        optimizer: "Union[str, tf.keras.optimizers.Optimizer, Type[tf.keras.optimizers.Optimizer]]" = "rmsprop",  # noqa: E501
        loss: "Union[str, tf.keras.losses.Loss, Type[tf.keras.losses.Loss], Callable, None]" = None,  # noqa: E501
        metrics: "Union[List[Union[str, tf.keras.metrics.Metric, Type[tf.keras.metrics.Metric], Callable, str, None]]]" = None,  # noqa: E501
        batch_size: "Union[int, None]" = None,
        validation_batch_size: "Union[int, None]" = None,
        verbose: int = 0,
        callbacks: "Optional[List[Union[tf.keras.callbacks.Callback, Type[tf.keras.callbacks.Callback]]]]" = None,  # noqa: E501
        validation_split: float = 0.0,
        shuffle: bool = True,
        run_eagerly: bool = False,
        epochs: int = 1,
        **kwargs,
    ):

        # Parse hardcoded params
        self.model = model
        self.warm_start = warm_start
        self.random_state = random_state
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.run_eagerly = run_eagerly
        self.epochs = epochs
        self.history_: "Dict" = {}

        # Unpack kwargs
        vars(self).update(**kwargs)

        # Save names of kwargs into set
        if kwargs:
            self._user_params = set(kwargs)

    @property
    def __name__(self):
        return self.__class__.__name__

    @property
    def current_epoch(self) -> int:
        """Returns the current training epoch.
        Returns
        -------
        int
            Current training epoch.
        """
        if not hasattr(self, "history_"):
            return 0
        return len(self.history_.get("loss", []))

    @staticmethod
    def _validate_sample_weight(
        X: np.ndarray,
        sample_weight: "Union[np.ndarray, Iterable]",
    ) -> "Tuple[np.ndarray, np.ndarray]":
        """Validate that the passed sample_weight and
        ensure it is a Numpy array."""
        sample_weight_: np.ndarray = _check_sample_weight(
            sample_weight, X, dtype=np.dtype(tf.keras.backend.floatx())
        )
        if np.all(sample_weight_ == 0):
            raise ValueError(
                "No training samples had any weight; only zeros were "
                "passed in sample_weight. That means there's nothing "
                "to train on by definition, so training can not be completed."
            )
        return X, sample_weight_

    def _check_model_param(self):
        """Checks ``model`` and returns model building
        function to use.
        Raises
        ------
            ValueError: if ``self.model`` is not valid.
        """
        model = self.model
        if model is None:
            # no model, use this class' _keras_build_fn
            if not hasattr(self, "_keras_build_fn"):
                raise ValueError(
                    "If not using the ``build_fn`` param, "
                    "you must implement ``_keras_build_fn``"
                )
            final_build_fn = self._keras_build_fn
        elif isinstance(model, tf.keras.Model):
            # pre-built Keras Model
            def final_build_fn():
                return model

        elif inspect.isfunction(model):
            if hasattr(self, "_keras_build_fn"):
                raise ValueError(
                    "This class cannot implement ``_keras_build_fn`` if"
                    " using the `model` parameter"
                )
            # a callable method/function
            final_build_fn = model
        else:
            raise TypeError(
                "``model`` must be a callable, a Keras Model instance or None"
            )

        return final_build_fn

    def _get_compile_kwargs(self):
        """Convert all __init__ params destined to
        `compile` into valid kwargs for `Model.compile` by parsing
        routed parameters and compiling optimizers, losses and metrics
        as needed.
        Returns
        -------
        dict
            Dictionary of kwargs for `Model.compile`.
        """
        init_params = self.get_params()
        compile_kwargs = self.route_params(
            init_params,
            destination="compile",
            pass_filter=self._compile_kwargs,
        )
        compile_kwargs["optimizer"] = self.try_to_convert_strings_to_classes(
            self.optimizer,
            self.get_optimizer_class
        )
        compile_kwargs["optimizer"] = self.unflatten_params(
            items=compile_kwargs["optimizer"],
            params=self.route_params(
                init_params,
                destination="optimizer",
                pass_filter=set(),
                strict=True,
            ),
        )
        compile_kwargs["loss"] = self.try_to_convert_strings_to_classes(
            compile_kwargs["loss"],
            self.get_loss_class_function_or_string
        )
        compile_kwargs["loss"] = self.unflatten_params(
            items=compile_kwargs["loss"],
            params=self.route_params(
                init_params,
                destination="loss",
                pass_filter=set(),
                strict=False,
            ),
        )
        compile_kwargs["metrics"] = self.try_to_convert_strings_to_classes(
            compile_kwargs.get("metrics", None),
            self.get_metric_class
        )
        compile_kwargs["metrics"] = self.unflatten_params(
            items=compile_kwargs["metrics"],
            params=self.route_params(
                init_params,
                destination="metrics",
                pass_filter=set(),
                strict=False,
            ),
        )
        return compile_kwargs

    def get_metric_class(
        self,
        metric: "Union[str, metrics_mod.Metric, Type[metrics_mod.Metric]]"
    ) -> "Union[metrics_mod.Metric, str]":
        if metric in ("acc", "accuracy", "ce", "crossentropy"):
            # Keras matches "acc" and others in this list to the right function
            # based on the Model's loss function, output shape, etc.
            # We pass them through here to let Keras deal with these.
            return metric
        return metrics_mod.get(metric)  # always returns a class

    def get_optimizer_class(
        self,
        optimizer: "Union[str, optimizers_mod.Optimizer, Type[optimizers_mod.Optimizer]]"  # noqa
    ) -> optimizers_mod.Optimizer:
        # optimizers.get returns instances instead of classes
        return type(optimizers_mod.get(optimizer))

    def get_loss_class_function_or_string(
        self,
        loss: str
    ) -> "Union[losses_mod.Loss, Callable]":
        got = losses_mod.get(loss)
        if type(got) == FunctionType:
            return got
        return type(got)  # a class, e.g. if loss="BinaryCrossentropy"

    def try_to_convert_strings_to_classes(
        self,
        items: "Union[str, dict, tuple, list]",
        class_getter: "Callable"
    ):
        """Convert shorthand optimizer/loss/metric names to classes."""
        from typing import Sequence, Mapping

        if isinstance(items, str):
            return class_getter(items)  # single item, despite parameter name
        elif isinstance(items, Sequence):
            return type(items)([
                self.try_to_convert_strings_to_classes(item, class_getter)
                for item in items
            ])
        elif isinstance(items, Mapping):
            return {
                k: self.try_to_convert_strings_to_classes(item, class_getter)
                for k, item in items.items()
            }
        else:
            return items  # not a string or known collection

    def route_params(
        self,
        params: "Dict[str, Any]",
        destination: str,
        pass_filter: "Union[None, Iterable[str]]",
        strict: bool = False,
    ) -> "Dict[str, Any]":
        """Route and trim parameter names.
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to route/filter.
        destination : str
            Destination to route to, ex: `build` or `compile`.
        pass_filter: Iterable[str]
            Only keys from `params` that are in the iterable are passed.
            This does not affect routed parameters.
        strict: bool
            Only include routed parameters target at `destination__...`
            and not any further routing (i.e.
            exclude `destination__inner__...`).
        Returns
        -------
        Dict[str, Any]
            Filtered parameters, with any routing prefixes removed.
        """
        res = dict()
        routed = {k: v for k, v in params.items() if "__" in k}
        non_routed = {k: params[k] for k in (params.keys() - routed.keys())}
        for key, val in non_routed.items():
            if pass_filter is None or key in pass_filter:
                res[key] = val
        for key, val in routed.items():
            prefix = destination + "__"
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                if strict and "__" in new_key:
                    continue
                res[new_key] = val
        return res

    def unflatten_params(self, items, params, base_params=None):
        """Recursively compile nested structures of classes
        using parameters from params.
        """
        if inspect.isclass(items):
            item = items
            new_base_params = {
                p: v
                for p, v
                in params.items()
                if "__" not in p
            }
            base_params = base_params or dict()
            args_and_kwargs = {**base_params, **new_base_params}
            for p, v in args_and_kwargs.items():
                args_and_kwargs[p] = self.unflatten_params(
                    items=v,
                    params=self.route_params(
                        params=params,
                        destination=f"{p}",
                        pass_filter=set(),
                        strict=False,
                    ),
                )
            kwargs = {
                k: v for k, v
                in args_and_kwargs.items()
                if k[0] not in DIGITS
            }
            args = [
                (int(k), v)
                for k, v
                in args_and_kwargs.items()
                if k not in kwargs
            ]
            args = (v for _, v in sorted(args))  # sorts by key / arg num
            return item(*args, **kwargs)

        if isinstance(items, (list, tuple)):
            iter_type_ = type(items)
            res = list()
            new_base_params = {
                p: v
                for p, v
                in params.items()
                if "__" not in p
            }
            for idx, item in enumerate(items):
                item_params = self.route_params(
                    params=params,
                    destination=f"{idx}",
                    pass_filter=set(),
                    strict=False,
                )
                res.append(
                    self.unflatten_params(
                        items=item,
                        params=item_params,
                        base_params=new_base_params
                    )
                )
            return iter_type_(res)

        if isinstance(items, (dict,)):
            res = dict()
            new_base_params = {
                p: v
                for p, v
                in params.items()
                if "__" not in p
            }
            for key, item in items.items():
                item_params = self.route_params(
                    params=params,
                    destination=f"{key}",
                    pass_filter=set(),
                    strict=False,
                )
                res[key] = self.unflatten_params(
                    items=item,
                    params=item_params,
                    base_params=new_base_params,
                )
            return res
        # non-compilable item, check if it has any routed parameters
        item = items
        new_base_params = {p: v for p, v in params.items() if "__" not in p}
        base_params = base_params or dict()
        kwargs = {**base_params, **new_base_params}
        if kwargs:
            raise TypeError(
                f'TypeError: "{str(item)}" object of type "{type(item)}"'
                "does not accept parameters because it's not a class."
                f' However, it received parameters "{kwargs}"'
            )
        return item

    def has_param(self, func: "Callable", param: str) -> bool:
        """Check if func has a parameter named param.
        Parameters
        ----------
        func : Callable
            Function to inspect.
        param : str
            Parameter name.
        Returns
        -------
        bool
            True if the parameter is part of func's signature,
            False otherwise.
        """
        return any(
            p.name == param
            for p in inspect.signature(func).parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        )

    def accepts_kwargs(self, func: "Callable") -> bool:
        """Check if ``func`` accepts kwargs."""
        return any(
            True
            for param in inspect.signature(func).parameters.values()
            if param.kind == param.VAR_KEYWORD
        )

    def _build_keras_model(self):
        """Build the Keras model.
        This method will process all arguments and call the model building
        function with appropriate arguments.
        Returns
        -------
        tensorflow.keras.Model
            Instantiated and compiled keras Model.
        """
        # dynamically build model, i.e. final_build_fn builds a Keras model

        # determine what type of build_fn to use
        final_build_fn = self._check_model_param()

        # collect parameters
        params = self.get_params()
        build_params = self.route_params(
            params,
            destination="model",
            pass_filter=getattr(self, "_user_params", set()),
            strict=True,
        )
        compile_kwargs = None
        if (
            self.has_param(final_build_fn, "meta")
            or self.accepts_kwargs(final_build_fn)
        ):
            # build_fn accepts `meta`, add it
            build_params["meta"] = self._get_metadata()
        if (
            self.has_param(final_build_fn, "compile_kwargs")
            or self.accepts_kwargs(final_build_fn)
        ):
            # build_fn accepts `compile_kwargs`, add it
            compile_kwargs = self._get_compile_kwargs()
            build_params["compile_kwargs"] = compile_kwargs
        if (
            self.has_param(final_build_fn, "params")
            or self.accepts_kwargs(final_build_fn)
        ):
            # build_fn accepts `params`, i.e. all of get_params()
            build_params["params"] = self.get_params()

        # build model
        if self._random_state is not None:
            with tensorflow_random_state(self._random_state):
                model = final_build_fn(**build_params)
        else:
            model = final_build_fn(**build_params)

        return model

    def _ensure_compiled_model(self) -> None:
        # compile model if user gave us an un-compiled model
        if (not (
            hasattr(self.model_, "loss")
            and hasattr(self.model_, "optimizer")
        )):
            kw = self._get_compile_kwargs()
            self.model_.compile(**kw)

    def _fit_keras_model(  # noqa: C901
        self,
        X: "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]",
        y: "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]",
        sample_weight: "Union[np.ndarray, None]",
        warm_start: bool,
        epochs: int,
        initial_epoch: int,
        **kwargs,
    ) -> None:
        """Fits the Keras model.
        Parameters
        ----------
        X : Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
            Training samples, as accepted by tf.keras.Model
        y : Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
            Target data, as accepted by tf.keras.Model
        sample_weight : Union[np.ndarray, None]
            Sample weights. Ignored by Keras if None.
        warm_start : bool
            If True, don't don't overwrite
            the history_ attribute and append to it instead.
        epochs : int
            Number of epochs for which the model will be trained.
        initial_epoch : int
            Epoch at which to begin training.
        **kwargs : Dict[str, Any]
            Extra arguments to route to ``Model.fit``.
        Returns
        -------
        BaseWrapper
            A reference to the instance that can be chain called
            (ex: instance.fit(X,y).transform(X) )
        """

        # Make sure model has a loss function
        loss = self.model_.loss
        no_loss = False
        if isinstance(loss, list) and not any(
            callable(loss_) or isinstance(loss_, str) for loss_ in loss
        ):
            no_loss = True
        if isinstance(loss, dict) and not any(
            callable(loss_) or isinstance(loss_, str)
            for loss_ in loss.values()
        ):
            no_loss = True
        if no_loss:
            raise ValueError(
                "No valid loss function found."
                " You must provide a loss function to train."
                "\n\nTo resolve this issue, do one of the following:"
                "\n 1. Provide a loss function via the loss parameter."
                "\n 2. Compile your model with a loss function inside the"
                " model-building method."
                "\n\nSee https://www.adriangb.com/scikeras/stable/advanced.html#compilation-of-model"  # noqa: E501
                " for more information on compiling SciKeras models."
                "\n\nSee https://www.tensorflow.org/api_docs/python/tf/keras/losses"  # noqa: E501
                " for more information on Keras losses."
            )

        # collect parameters
        params = self.get_params()
        fit_args = self.route_params(
            params,
            destination="fit",
            pass_filter=self._fit_kwargs
        )
        fit_args["sample_weight"] = sample_weight
        fit_args["epochs"] = initial_epoch + epochs
        fit_args["initial_epoch"] = initial_epoch
        fit_args.update(kwargs)
        for bs_kwarg in ("batch_size", "validation_batch_size"):
            if bs_kwarg in fit_args:
                if fit_args[bs_kwarg] == -1:
                    try:
                        if isinstance(X, list):
                            bs = X[0].shape[0]
                        elif isinstance(X, dict):
                            bs = list(X.values())[0].shape[0]
                        else:
                            bs = X.shape[0]

                        fit_args[bs_kwarg] = bs
                    except AttributeError:
                        raise ValueError(
                            f"`{bs_kwarg}=-1` requires "
                            "that `X` implement `shape`"
                        )
        fit_args = {
            k: v for k, v
            in fit_args.items()
            if not k.startswith("callbacks")
        }
        fit_args["verbose"] = self.verbose
        fit_args["callbacks"] = self._fit_callbacks

        if self._random_state is not None:
            with tensorflow_random_state(self._random_state):
                hist = self.model_.fit(x=X, y=y, **fit_args)
        else:
            hist = self.model_.fit(x=X, y=y, **fit_args)

        if (
            not warm_start
            or not hasattr(self, "history_")
            or initial_epoch == 0
        ):
            self.history_ = defaultdict(list)

        for key, val in hist.history.items():
            try:
                key = metric_name(key)
            except ValueError as e:
                # Keras puts keys like "val_accuracy" and "loss" and
                # "val_loss" in hist.history
                if "Unknown metric function" not in str(e):
                    raise e
            self.history_[key] += val

    def _check_model_compatibility(
        self,
        y: "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]",
    ) -> None:
        """Checks that the model output number and y shape match.
        This is in place to avoid cryptic TF errors.
        """
        # check if this is a multi-output model
        if getattr(self, "n_outputs_expected_", None):
            # n_outputs_expected_ is generated by data transformers
            # we recognize the attribute but do not force it to be
            # generated
            if self.n_outputs_expected_ != len(self.model_.outputs):
                raise ValueError(
                    "Detected a Keras model input of size"
                    f" {self.n_outputs_expected_ }, but {self.model_} has"
                    f" {len(self.model_.outputs)} outputs"
                )
        # check that if the user gave us a loss function it ended up in
        # the actual model
        init_params = inspect.signature(self.__class__).parameters
        if "loss" in init_params:
            default_val = init_params["loss"].default
            if all(
                isinstance(x, (str, losses_mod.Loss, type))
                for x in [self.loss, self.model_.loss]
            ):  # filter out loss list/dicts/etc.
                if default_val is not None:
                    default_val = loss_name(default_val)
                given = loss_name(self.loss)
                got = loss_name(self.model_.loss)
                if given != default_val and got != given:
                    raise ValueError(
                        f"loss={self.loss} but model compiled with "
                        "{self.model_.loss}. Data may not match "
                        "loss function!"
                    )

    def _check_array_dtype(self, arr, force_numeric):
        if not isinstance(arr, np.ndarray):
            return self._check_array_dtype(
                np.asarray(arr),
                force_numeric=force_numeric
            )
        elif (
            arr.dtype.kind not in ("O", "U", "S") or not force_numeric
        ):  # object, unicode or string
            # already numeric
            return None  # check_array won't do any casting with dtype=None
        else:
            # default to TFs backend float type
            # instead of float64 (sklearn's default)
            return tf.keras.backend.floatx()

    @staticmethod
    def _all_sample_first_dim(
        arrs: "Sequence[np.ndarray]"
    ) -> "Tuple[bool, Optional[int]]":
        s = set(it.shape[0] for it in arrs)
        is_single = len(s) <= 1

        if len(s) == 1:
            length: "Optional[int]" = s.pop()
        else:
            length = None
        return is_single, length

    def _all_sample_same_first_dim(self, arrs: "Sequence[np.ndarray]") -> bool:
        b, _ = self._all_sample_first_dim(arrs)
        return b

    def _check_y(  # noqa: C901
        self,
        y: "Union[npt.ArrayLike, List[npt.ArrayLike], Dict[str, npt.ArrayLike]]",  # noqa: E501
        reset: bool,
        y_numeric: bool = False,
        estimator: "Optional[str]" = None,
    ) -> "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]":
        multi_output = (
            isinstance(y, (list, dict))
            and not any(isinstance(yi, (int, float)) for yi in y)
        )

        def _check_y_array(arr: "npt.ArrayLike") -> "np.ndarray":
            return np.asarray(check_array(
                arr,
                accept_sparse="csr",
                force_all_finite=True,
                input_name="y",
                ensure_2d=False,
                allow_nd=False,
                dtype=self._check_array_dtype(arr, y_numeric),
            ))

        if multi_output and isinstance(y, list):
            y_: "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]" = [  # noqa: E501
                _check_y_array(yi)
                for yi
                in y
            ]
            assert isinstance(y_, list)
            if not self._all_sample_same_first_dim(y_):
                raise ValueError(
                    "The y-arrays have different numbers of samples."
                )

            target_type: "Union[str, List[str], Dict[str, str]]" = [
                self._type_of_target(yi) for yi in y_
            ]
            y_dtype_: "Union[List[np.dtype], Dict[str, np.dtype], np.dtype]" = [  # noqa: E501
                yi.dtype for yi in y_
            ]
            y_ndim_: "Union[List[int], Dict[str, int], int]" = [yi.ndim for yi in y_]  # noqa: E501
        elif multi_output and isinstance(y, dict):
            y_ = {k: _check_y_array(yi) for k, yi in y.items()}

            assert isinstance(y_, dict)
            if not self._all_sample_same_first_dim(list(y_.values())):
                raise ValueError(
                    "The y-arrays have different numbers of samples."
                )

            target_type = {k: self._type_of_target(yi) for k, yi in y_.items()}
            y_dtype_ = {k: yi.dtype for k, yi in y_.items()}
            y_ndim_ = {k: yi.ndim for k, yi in y_.items()}
        else:
            y_ = _check_y_array(np.asarray(y))
            assert isinstance(y_, np.ndarray)
            target_type = self._type_of_target(y_)
            y_dtype_ = y_.dtype
            y_ndim_ = y_.ndim

        if reset:
            self.target_type_ = target_type
            self.y_dtype_ = y_dtype_
            self.y_ndim_ = y_ndim_
        elif multi_output:
            if isinstance(y_, dict):
                keys: "Iterable[Union[str, int]]" = y_.keys()
            else:
                keys = range(len(y_))

            assert isinstance(y_, (list, dict))
            assert isinstance(y_dtype_, (list, dict))
            assert isinstance(y_ndim_, (list, dict))
            assert isinstance(self.y_dtype_, (list, dict))
            assert isinstance(self.y_ndim_, (list, dict))
            for k, yi_dtype_, dtype_, yi_ndim_, ndim_ in zip(
                keys,
                y_dtype_,
                self.y_dtype_,
                y_ndim_,
                self.y_ndim_
            ):
                assert isinstance(yi_dtype_, np.dtype)
                assert isinstance(dtype_, np.dtype)
                if not np.can_cast(yi_dtype_, dtype_):
                    raise ValueError(
                        f"Got y input {k} with dtype {yi_dtype_},"
                        f" but this {self.__name__} expected {dtype_}"
                        f" and casting from {yi_dtype_} to {dtype_} "
                        "is not safe!"
                    )
                if ndim_ != yi_ndim_:
                    raise ValueError(
                        f"y input {k} has {yi_ndim_} dimensions, but "
                        f"this {self.__name__} is expecting {ndim_} "
                        f"dimensions in y {k}."
                    )
        else:
            assert isinstance(y_, np.ndarray)
            assert isinstance(y_dtype_, np.dtype)
            assert isinstance(self.y_dtype_, np.dtype)
            if not np.can_cast(y_dtype_, self.y_dtype_):
                raise ValueError(
                    f"Got y with dtype {y_dtype_},"
                    f" but this {self.__name__} expected {self.y_dtype_}"
                    f" and casting from {y_dtype_} to {self.y_dtype_} "
                    "is not safe!"
                )
            if self.y_ndim_ != y_ndim_:
                raise ValueError(
                    f"y has {y_ndim_} dimensions, but this {self.__name__}"
                    f" is expecting {self.y_ndim_} dimensions in y."
                )
        return y_

    def _check_X(  # noqa: C901
        self,
        X: "Union[npt.ArrayLike, List[npt.ArrayLike], Dict[str, npt.ArrayLike]]",  # noqa: E501
        reset: bool,
        estimator: "Optional[str]" = None,
    ) -> "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]":  # noqa: E501
        multi_input = (
            isinstance(X, (list, dict))
            and not any(isinstance(xi, (int, float)) for xi in X)
        )

        def _check_X_array(arr: "npt.ArrayLike") -> "np.ndarray":
            return np.asarray(check_array(
                arr,
                accept_sparse=False,
                accept_large_sparse=False,
                force_all_finite=True,
                ensure_min_samples=1,
                ensure_min_features=1,
                allow_nd=True,
                ensure_2d=True,
                dtype=self._check_array_dtype(arr, True),
                estimator=estimator,
                input_name="X",
            ))

        if multi_input and isinstance(X, list):
            X_: "Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]" = [  # noqa: E501
                _check_X_array(xi)
                for xi
                in X
            ]

            assert isinstance(X_, list)
            if not self._all_sample_same_first_dim(X_):
                raise ValueError(
                    "The X-arrays have different numbers of samples."
                )

            X_dtype_: "Union[List[np.dtype], Dict[str, np.dtype], np.dtype]" = [  # noqa: E501
                xi.dtype for xi in X_
            ]
            n_features_in_: "Union[int, List[int], Dict[str, int]]" = [
                xi.shape[1] for xi in X_
            ]
            X_shape_: "Union[List[Tuple[int, ...]], Dict[str, Tuple[int, ...]], Tuple[int, ...]]" = [  # noqa: E501
                xi.shape for xi in X_
            ]
        elif multi_input and isinstance(X, dict):
            X_ = {k: _check_X_array(xi) for k, xi in X.items()}

            assert isinstance(X_, dict)
            if not self._all_sample_same_first_dim(list(X_.values())):
                raise ValueError(
                    "The X-arrays have different numbers of samples."
                )

            X_dtype_ = {k: xi.dtype for k, xi in X_.items()}
            n_features_in_ = {k: xi.shape[1] for k, xi in X_.items()}
            X_shape_ = {k: xi.shape for k, xi in X_.items()}
        else:
            assert isinstance(X, np.ndarray)
            X_ = _check_X_array(X)
            X_dtype_ = X_.dtype
            n_features_in_ = X_.shape[1]
            X_shape_ = X_.shape

        if reset:
            self.X_dtype_ = X_dtype_
            self.X_shape_ = X_shape_
            self.n_features_in_ = n_features_in_

        elif multi_input:
            isdict = isinstance(X_, dict)

            if isdict:
                assert isinstance(X_, dict)
                keys: "Iterable[Union[int, str]]" = X_.keys()
            else:
                assert isinstance(X_, list)
                keys = range(len(X_))

            for k in keys:
                if isdict:
                    assert isinstance(k, str)
                    assert isinstance(X_dtype_, dict)
                    assert isinstance(X_shape_, dict)
                    assert isinstance(self.X_dtype_, dict)
                    assert isinstance(self.X_shape_, dict)
                    xi_dtype_ = cast("Dict", X_dtype_)[k]
                    dtype_ = cast("Dict", self.X_dtype_)[k]
                    xi_shape_ = cast("Dict", X_shape_)[k]
                    shape_ = cast("Dict", self.X_shape_)[k]
                else:
                    assert isinstance(k, int)
                    assert isinstance(X_dtype_, list)
                    assert isinstance(X_shape_, list)
                    assert isinstance(self.X_dtype_, list)
                    assert isinstance(self.X_shape_, list)
                    xi_dtype_ = cast("List", X_dtype_)[k]
                    dtype_ = cast("List", self.X_dtype_)[k]
                    xi_shape_ = cast("List", X_shape_)[k]
                    shape_ = cast("List", self.X_shape_)[k]

                assert isinstance(xi_dtype_, np.dtype)
                assert isinstance(dtype_, np.dtype)
                if not np.can_cast(xi_dtype_, dtype_):
                    raise ValueError(
                        f"Got y input {k} with dtype {xi_dtype_},"
                        f" but this {self.__name__} expected {dtype_}"
                        f" and casting from {xi_dtype_} to {dtype_} "
                        "is not safe!"
                    )
                if len(shape_) != len(xi_shape_):
                    raise ValueError(
                        f"X input {k} has {len(xi_shape_)} dimensions, "
                        f"but this {self.__name__} is expecting "
                        f"{len(shape_)} dimensions in X {k}."
                    )
                if shape_[1:] != xi_shape_[1:]:
                    raise ValueError(
                        f"X has shape {xi_shape_[1:]}, but this "
                        f"{self.__name__} is expecting X of shape "
                        f"{shape_[1:]}"
                    )
        else:
            assert isinstance(X_dtype_, np.dtype)
            assert isinstance(self.X_dtype_, np.dtype), self.X_dtype_
            if not np.can_cast(X_dtype_, self.X_dtype_):
                raise ValueError(
                    f"Got X with dtype {X_dtype_},"
                    f" but this {self.__name__} expected {self.X_dtype_}"
                    f" and casting from {X_dtype_} to {self.X_dtype_} "
                    "is not safe!"
                )
            if len(self.X_shape_) != len(X_shape_):
                raise ValueError(
                    f"X has {len(X_shape_)} dimensions, but this "
                    f"{self.__name__} is expecting {len(self.X_shape)} "
                    "dimensions in X."
                )

            assert isinstance(X_shape_, tuple)
            assert isinstance(self.X_shape_, tuple)
            if X_shape_[1:] != self.X_shape_[1:]:
                raise ValueError(
                    f"X has shape {X_shape_[1:]}, but this {self.__name__}"
                    f" is expecting X of shape {self.X_shape_[1:]}"
                )
        return X_

    def _check_X_y(
        self,
        X: "Union[None, np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]" = None,  # noqa: E501
        y: "Union[None, np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]" = None,  # noqa: E501
        accept_sparse=False,
        *,
        accept_large_sparse=True,
        dtype="numeric",
        order=None,
        copy=False,
        force_all_finite=True,
        ensure_2d=True,
        allow_nd=False,
        multi_output=False,
        ensure_min_samples=1,
        ensure_min_features=1,
        y_numeric=False,
        estimator=None,
    ) -> """Tuple[
            Union[None, np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],  # noqa: E501
            Union[None, np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],  # noqa: E501
    ]""":
        """
        A rewrite of
        https://github.com/scikit-learn/scikit-learn/blob/4685cf624582cbc9a35d646f239347e54db798dc/sklearn/utils/validation.py#L941"
        to allow for the multi-input bits
        """
        from sklearn.utils.validation import check_consistent_length

        if y is None:
            if estimator is None:
                estimator_name = "estimator"
            else:
                estimator_name = ""

            raise ValueError(
                f"{estimator_name} requires y to be passed, "
                "but the target y is None."
            )

        arrays: "List[np.ndarray]" = []

        if isinstance(X, list):
            arrays.extend(X)
        elif isinstance(X, dict):
            arrays.extend(X.values())
        elif X is not None:
            arrays.append(X)

        if isinstance(y, list):
            arrays.extend(y)
        elif isinstance(y, dict):
            arrays.extend(y.values())
        elif y is not None:
            arrays.append(y)

        check_consistent_length(*arrays)
        return X, y

    def _validate_data(
        self,
        X: "Union[None, npt.ArrayLike, List[npt.ArrayLike], Dict[str, npt.ArrayLike]]" = None,  # noqa: E501
        y: "Union[None, npt.ArrayLike, List[npt.ArrayLike], Dict[str, npt.ArrayLike]]" = None,  # noqa: E501
        reset: bool = False,
        y_numeric: bool = False
    ) -> """Tuple[
            Union[None, np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],  # noqa: E501
            Union[None, np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],  # noqa: E501
    ]""":
        """Validate input arrays and set or check their meta-parameters.
        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of shape \
           (n_samples, n_features)
            The input samples. If None, ``check_array`` is called on y and
            ``check_X_y`` is called otherwise.
        y : Union[array-like, sparse matrix, dataframe] of shape \
            (n_samples,), default=None
            The targets. If None, ``check_array`` is called on X and
            ``check_X_y`` is called otherwise.
        reset : bool, default=False
            If True, override all meta attributes.
            If False, verify that they haven't changed.
        y_numeric : bool, default = False
            If True, ensure y is a numeric dtype.
            If False, allow non-numeric y to pass through.
        Returns
        -------
        Tuple[np.ndarray, Union[np.ndarray, None]]
            The validated input.
        """

        if y is not None:
            y_ = self._check_y(y, reset=reset, y_numeric=y_numeric)
        else:
            y_ = None

        if X is not None:
            X_ = self._check_X(X, reset=reset)
        else:
            X_ = None

        if X is not None and y is not None:
            X_, y_ = self._check_X_y(
                X_,
                y_,
                allow_nd=True,  # allow X to have more than 2 dimensions
                multi_output=True,  # allow y to be 2D
                dtype=None,
            )

        return X_, y_

    def _type_of_target(self, y: np.ndarray) -> str:
        return type_of_target(y)

    def fit(self, X, y, sample_weight=None, **kwargs) -> "TFBase":
        """Constructs a new model with ``model`` & fit the model to ``(X, y)``.
        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of
            shape (n_samples, n_features)
            Training samples, where n_samples is the number of samples
            and n_features is the number of features.
        y : Union[array-like, sparse matrix, dataframe] of shape
            (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        **kwargs : Dict[str, Any]
            Extra arguments to route to ``Model.fit``.
        Warnings
        --------
            Passing estimator parameters as keyword arguments
            (aka as ``**kwargs``) to ``fit`` is not supported by the
            Scikit-Learn API, and will be removed in a future version
            of SciKeras. These parameters can also be specified by
            prefixing ``fit__`` to a parameter at initialization
            (``BaseWrapper(..., fit__batch_size=32, predict__batch_size=1000)``)  # noqa: E501
            or by using ``set_params``
            (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``).
        Returns
        -------
        BaseWrapper
            A reference to the instance that can be chain called (``est.fit(X,y).transform(X)``).
        """
        # epochs via kwargs > fit__epochs > epochs
        kwargs["epochs"] = kwargs.get(
            "epochs", getattr(self, "fit__epochs", self.epochs)
        )
        kwargs["initial_epoch"] = kwargs.get("initial_epoch", 0)

        self._fit(
            X=X,
            y=y,
            sample_weight=sample_weight,
            warm_start=self.warm_start,
            **kwargs,
        )

        return self

    @property
    def initialized_(self) -> bool:
        """Checks if the estimator is intialized.
        Returns
        -------
        bool
            True if the estimator is initialized (i.e., it can
            be used for inference or is ready to train),
            otherwise False.
        """
        return hasattr(self, "model_")

    def _initialize_callbacks(self) -> None:
        from typing import Mapping

        params = self.get_params()

        def initialize(destination: str):
            if params.get(destination) is not None:
                callback_kwargs = self.route_params(
                    params,
                    destination=destination,
                    pass_filter=set()
                )
                callbacks = self.unflatten_params(
                    items=params[destination],
                    params=callback_kwargs
                )
                if isinstance(callbacks, Mapping):
                    # Keras does not officially support dicts,
                    # convert to a list
                    callbacks = list(callbacks.values())
                elif isinstance(callbacks, tf.keras.callbacks.Callback):
                    # a single instance, not officially supported
                    # so wrap in a list
                    callbacks = [callbacks]
                err = False
                if not isinstance(callbacks, List):
                    err = True
                for cb in callbacks:
                    if isinstance(cb, List):
                        for nested_cb in cb:
                            if not isinstance(
                                nested_cb,
                                tf.keras.callbacks.Callback
                            ):
                                err = True
                    elif not isinstance(cb, tf.keras.callbacks.Callback):
                        err = True
                if err:
                    raise TypeError(
                        "If specified, ``callbacks`` must be one of:"
                        "\n - A dict of string keys with callbacks "
                        "or lists of callbacks as values"
                        "\n - A list of callbacks or lists of callbacks"
                        "\n - A single callback"
                        "\nWhere each callback can be a instance of "
                        "`tf.keras.callbacks.Callback` or a sublass "
                        "of it to be compiled by SciKeras"
                    )
            else:
                callbacks = []
            return callbacks

        all_callbacks = initialize("callbacks")
        self._fit_callbacks = all_callbacks + initialize("fit__callbacks")
        self._predict_callbacks = (
            all_callbacks + initialize("predict__callbacks")
        )

    def _initialize(
        self,
        X: "Union[None, npt.ArrayLike, List[npt.ArrayLike], Dict[str, npt.ArrayLike]]",  # noqa: E501
        y: "Union[None, npt.ArrayLike, List[npt.ArrayLike], Dict[str, npt.ArrayLike]]" = None,  # noqa: E501
    ) -> """Tuple[
        Union[None, np.ndarray, List[np.ndarray], Dict[str, np.ndarray]],
        Union[None, np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]
    ]""":

        # Handle random state
        if isinstance(self.random_state, np.random.RandomState):
            # Keras needs an integer
            # we sample an integer and use that as a seed
            # Given the same RandomState, the seed will always be
            # the same, thus giving reproducible results
            state = self.random_state.get_state()
            r = np.random.RandomState()
            r.set_state(state)
            self._random_state: "Optional[int]" = r.randint(low=1)
        else:
            # int or None
            self._random_state = self.random_state

        X_, y_ = self._validate_data(X, y, reset=True)

        self.model_ = self._build_keras_model()
        self._initialize_callbacks()

        return X_, y_

    def initialize(
        self,
        X,
        y=None
    ) -> "TFBase":
        """Initialize the model without any fitting.
        You only need to call this model if you explicitly do not
        want to do any fitting (for example with a pretrained model).
        You should _not_ call this right before calling ``fit``, calling
        ``fit`` will do this automatically.

        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of
            shape (n_samples, n_features) Training samples where
            n_samples is the number of samples and `n_features`
            is the number of features.
        y : Union[array-like, sparse matrix, dataframe] of shape \
            (n_samples,) or (n_samples, n_outputs), default None
            True labels for X.
        Returns
        -------
        BaseWrapper
            A reference to the BaseWrapper instance for chained calling.
        """
        self._initialize(X, y)
        return self  # to allow chained calls like initialize(...).predict(...)

    def _fit(
        self,
        X,
        y,
        sample_weight,
        warm_start: bool,
        epochs: int,
        initial_epoch: int,
        **kwargs,
    ) -> None:
        """Constructs a new model with ``model`` & fit the model to ``(X, y)``.
        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe] of shape
            (n_samples, n_features). Training samples where
            `n_samples` is the number of samples and `n_features`
            is the number of features.
        y : Union[array-like, sparse matrix, dataframe] of shape
            (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        warm_start : bool
            If True, don't rebuild the model.
        epochs : int
            Number of passes over the entire dataset for which to train the
            model.
        initial_epoch : int
            Epoch at which to begin training.
        **kwargs : Dict[str, Any]
            Extra arguments to route to ``Model.fit``.
        """
        # Data checks
        if not ((self.warm_start or warm_start) and self.initialized_):
            X, y = self._initialize(X, y)
        else:
            X, y = self._validate_data(X, y)
        self._ensure_compiled_model()

        if sample_weight is not None:
            X, sample_weight = self._validate_sample_weight(X, sample_weight)

        self._check_model_compatibility(y)

        self._fit_keras_model(
            X,
            y,
            sample_weight=sample_weight,
            warm_start=warm_start,
            epochs=epochs,
            initial_epoch=initial_epoch,
            **kwargs,
        )

    def partial_fit(self, X, y, sample_weight=None, **kwargs) -> "TFBase":
        """Fit the estimator for a single epoch, preserving the current
        training history and model parameters.
        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe]
            of shape (n_samples, n_features)
            Training samples where n_samples is the number of samples
            and n_features is the number of features.
        y : Union[array-like, sparse matrix, dataframe] of shape \
            (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        **kwargs : Dict[str, Any]
            Extra arguments to route to ``Model.fit``.
        Returns
        -------
        BaseWrapper
            A reference to the instance that can be chain called
            (ex: instance.partial_fit(X, y).transform(X) )
        """
        if "epochs" in kwargs:
            raise TypeError(
                "Invalid argument `epochs` to `partial_fit`: "
                "`partial_fit` always trains for 1 epoch"
            )
        if "initial_epoch" in kwargs:
            raise TypeError(
                "Invalid argument `initial_epoch` to `partial_fit`: "
                "`partial_fit` always trains for from the current epoch"
            )

        self._fit(
            X,
            y,
            sample_weight=sample_weight,
            warm_start=True,
            epochs=1,
            initial_epoch=self.current_epoch,
            **kwargs,
        )
        return self

    def _predict_raw(self, X, **kwargs):
        """Obtain raw predictions from Keras Model.
        For classification, this corresponds to predict_proba.
        For regression, this corresponds to predict.
        """
        # check if fitted
        if not self.initialized_:
            raise NotFittedError(
                "Estimator needs to be fit before `predict` " "can be called"
            )
        # basic input checks
        X, _ = self._validate_data(X=X, y=None)

        # filter kwargs and get attributes for predict
        params = self.get_params()
        pred_args = self.route_params(
            params,
            destination="predict",
            pass_filter=self._predict_kwargs,
            strict=True
        )
        pred_args = {
            k: v for k, v in pred_args.items() if not k.startswith("callbacks")
        }
        pred_args["callbacks"] = self._predict_callbacks
        pred_args.update(kwargs)
        if "batch_size" in pred_args:
            if pred_args["batch_size"] == -1:
                try:
                    pred_args["batch_size"] = X.shape[0]
                except AttributeError:
                    raise ValueError(
                        "`batch_size=-1` requires that `X` implement `shape`"
                    )

        # predict with Keras model
        y_pred = self.model_.predict(x=X, **pred_args)

        return y_pred

    def predict(self, X, **kwargs):
        """Returns predictions for the given test data.
        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe]
            of shape (n_samples, n_features)
            Training samples where n_samples is the number of samples
            and n_features is the number of features.
        **kwargs : Dict[str, Any]
            Extra arguments to route to ``Model.predict``.
        Warnings
        --------
            Passing estimator parameters as keyword arguments (aka as
            ``**kwargs``) to ``predict`` is not supported by the
            Scikit-Learn API,
            and will be removed in a future version of SciKeras.
            These parameters can also be specified by prefixing
            ``predict__`` to a parameter at initialization
            (``BaseWrapper(..., fit__batch_size=32, predict__batch_size=1000)``)  # noqa: E501
            or by using ``set_params``
            (``est.set_params(fit__batch_size=32, predict__batch_size=1000)``).
        Returns
        -------
        array-like
            Predictions, of shape shape (n_samples,) or (n_samples, n_outputs).
        """
        # predict with Keras model
        y_pred = self._predict_raw(X=X, **kwargs)

        return y_pred

    @staticmethod
    def scorer(y_true, y_pred, **kwargs) -> float:
        """Scoring function for model.
        This is not implemented in BaseWrapper, it exists
        as a stub for documentation.
        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels.
        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted labels.
        **kwargs: dict
            Extra parameters passed to the scorer.
        Returns
        -------
        float
            Score for the test data set.
        """
        raise NotImplementedError("Scoring is not implemented on BaseWrapper.")

    def score(self, X, y, sample_weight=None) -> float:
        """Returns the score on the given test data and labels.
        No default scoring function is implemented in BaseWrapper,
        you must subclass and implement one.
        Parameters
        ----------
        X : Union[array-like, sparse matrix, dataframe]
            of shape (n_samples, n_features)
            Test input samples, where n_samples is the number of samples
            and n_features is the number of features.
        y : Union[array-like, sparse matrix, dataframe] of shape \
            (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        Returns
        -------
        float
            Score for the test data set.
        """
        # validate sample weights
        if sample_weight is not None:
            X, sample_weight = self._validate_sample_weight(
                X=X, sample_weight=sample_weight
            )

        # validate y
        _, y = self._validate_data(X=None, y=y)

        # compute Keras model score
        y_pred = self.predict(X)

        # filter kwargs and get attributes for score
        params = self.get_params()
        score_args = self.route_params(
            params,
            destination="score",
            pass_filter=set()
        )

        return self.scorer(
            y,
            y_pred,
            sample_weight=sample_weight,
            **score_args
        )

    def _get_metadata(self) -> "Dict[str, Any]":
        """Meta parameters (parameters created by fit, like
        n_features_in_ or target_type_).
        Returns
        -------
        Dict[str, Any]
            Dictionary of meta parameters
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if (len(k) > 1 and k[-1] == "_" and k[-2] != "_" and k[0] != "_")
        }

    def set_params(self, **params) -> "TFBase":
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        This also supports routed parameters, eg:
            ``classifier__optimizer__learning_rate``.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        BaseWrapper
            Estimator instance.
        """
        for param, value in params.items():
            if any(
                param.startswith(prefix + "__")
                for prefix
                in self._routing_prefixes
            ):
                # routed param
                setattr(self, param, value)
            else:
                try:
                    super().set_params(**{param: value})
                except ValueError:
                    # Give a SciKeras specific user message to aid
                    # in moving from the Keras wrappers
                    raise ValueError(
                        f"Invalid parameter {param} for estimator "
                        f"{self.__name__}.\nThis issue can likely be "
                        "resolved by setting this parameter"
                        f" in the {self.__name__} constructor:"
                        f"\n`{self.__name__}({param}={value})`"
                        "\nCheck the list of available parameters with"
                        " `estimator.get_params().keys()`"
                    ) from None
        return self

    '''
    def _get_param_names(self):
        """Get parameter names for the estimator"""
        return (
            k for k
            in self.__dict__
            if not k.endswith("_") and not k.startswith("_")
        )
    '''

    def _more_tags(self):
        """Get sklearn tags for the estimator"""
        tags = super()._more_tags()
        tags.update(self._tags)
        return tags

    def __repr__(self):
        repr_ = str(self.__name__)
        repr_ += "("
        params = self.get_params()
        if params:
            repr_ += "\n"
        for key, val in params.items():
            repr_ += "\t" + key + "=" + str(val) + "\n"
        repr_ += ")"
        return repr_


class ConvMLPWrapper(TFBase):

    def __init__(
        self,
        loss: "Literal[None, 'mse', 'mae', 'binary_crossentropy', 'pairwise']" = None,  # noqa: E501
        optimizer="adam",
        optimizer__learning_rate=0.001,
        epochs=200,
        verbose=0,
        conv_nlayers: int = 0,
        conv_filters: int = 5,
        conv_strides: int = 1,
        conv_kernel_size: int = 3,
        conv_activation: "Literal['linear', 'relu']" = "linear",
        conv_l1_rate: float = 0.0,
        conv_l2_rate: float = 0.0,
        conv_use_batchnorm: bool = True,
        adaptive_l1: bool = False,
        adaptive_l1_rate: float = 1e-6,
        adaptive_l2_rate: float = 1e-6,
        marker_embed_nlayers: int = 1,
        marker_embed_residual: bool = False,
        marker_embed_nunits: int = 10,
        marker_embed_final_nunits: "Optional[int]" = None,
        marker_embed_0_dropout_rate: float = 0.5,
        marker_embed_1_dropout_rate: float = 0.5,
        marker_embed_0_l1_rate: float = 0.0,
        marker_embed_1_l1_rate: float = 0.0,
        marker_embed_0_l2_rate: float = 0.0,
        marker_embed_1_l2_rate: float = 0.0,
        marker_embed_activation: "Literal['linear', 'relu']" = "linear",
        dist_embed_nlayers: int = 1,
        dist_embed_residual: bool = False,
        dist_embed_nunits: int = 10,
        dist_embed_final_nunits: "Optional[int]" = None,
        dist_embed_0_dropout_rate: float = 0.5,
        dist_embed_1_dropout_rate: float = 0.5,
        dist_embed_0_l1_rate: float = 0.0,
        dist_embed_1_l1_rate: float = 0.0,
        dist_embed_0_l2_rate: float = 0.0,
        dist_embed_1_l2_rate: float = 0.0,
        dist_embed_activation: "Literal['linear', 'relu']" = "linear",
        group_embed_nlayers: int = 1,
        group_embed_residual: bool = False,
        group_embed_nunits: int = 10,
        group_embed_final_nunits: "Optional[int]" = None,
        group_embed_0_dropout_rate: float = 0.5,
        group_embed_1_dropout_rate: float = 0.5,
        group_embed_0_l1_rate: float = 0.0,
        group_embed_1_l1_rate: float = 0.0,
        group_embed_0_l2_rate: float = 0.0,
        group_embed_1_l2_rate: float = 0.0,
        group_embed_activation: "Literal['linear', 'relu']" = "linear",
        covariate_embed_nlayers: int = 1,
        covariate_embed_residual: bool = False,
        covariate_embed_nunits: int = 10,
        covariate_embed_final_nunits: "Optional[int]" = None,
        covariate_embed_0_dropout_rate: float = 0.5,
        covariate_embed_1_dropout_rate: float = 0.5,
        covariate_embed_0_l1_rate: float = 0.0,
        covariate_embed_1_l1_rate: float = 0.0,
        covariate_embed_0_l2_rate: float = 0.0,
        covariate_embed_1_l2_rate: float = 0.0,
        covariate_embed_activation: "Literal['linear', 'relu']" = "linear",
        combine_method: "Literal['add', 'concatenate']" = "concatenate",
        post_embed_nlayers: int = 1,
        post_embed_residual: bool = False,
        post_embed_nunits: int = 10,
        post_embed_final_nunits: "Optional[int]" = None,
        post_embed_0_dropout_rate: float = 0.5,
        post_embed_1_dropout_rate: float = 0.5,
        post_embed_0_l1_rate: float = 0.0,
        post_embed_1_l1_rate: float = 0.0,
        post_embed_0_l2_rate: float = 0.0,
        post_embed_1_l2_rate: float = 0.0,
        post_embed_activation: "Literal['linear', 'relu']" = "linear",
        use_predictor: bool = True,
        predictor_l1_rate: float = 0.0,
        predictor_l2_rate: float = 0.0,
        predictor_dropout_rate: float = 0.0,
        predictor_use_bias: bool = True,
        hard_triplet_loss_rate: "Optional[float]" = None,
        semihard_triplet_loss_rate: "Optional[float]" = None,
        multisurf_loss_rate: "Optional[float]" = None,
        input_names: "Optional[List[Literal['markers', 'dists', 'groups', 'covariates']]]" = None,  # noqa: E501
        name="conv_mlp",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.loss = loss

        self.optimizer = optimizer
        self.optimizer__learning_rate = optimizer__learning_rate
        self.epochs = epochs
        self.verbose = verbose

        self.conv_nlayers = conv_nlayers
        self.conv_filters = conv_filters
        self.conv_strides = conv_strides
        self.conv_kernel_size = conv_kernel_size
        self.conv_activation = conv_activation
        self.conv_l1_rate = conv_l1_rate
        self.conv_l2_rate = conv_l2_rate
        self.conv_use_batchnorm = conv_use_batchnorm
        self.adaptive_l1 = adaptive_l1
        self.adaptive_l1_rate = adaptive_l1_rate
        self.adaptive_l2_rate = adaptive_l2_rate
        self.marker_embed_nlayers = marker_embed_nlayers
        self.marker_embed_residual = marker_embed_residual
        self.marker_embed_nunits = marker_embed_nunits
        self.marker_embed_final_nunits = marker_embed_final_nunits
        self.marker_embed_0_dropout_rate = marker_embed_0_dropout_rate
        self.marker_embed_1_dropout_rate = marker_embed_1_dropout_rate
        self.marker_embed_0_l1_rate = marker_embed_0_l1_rate
        self.marker_embed_1_l1_rate = marker_embed_1_l1_rate
        self.marker_embed_0_l2_rate = marker_embed_0_l2_rate
        self.marker_embed_1_l2_rate = marker_embed_1_l2_rate
        self.marker_embed_activation = marker_embed_activation
        self.dist_embed_nlayers = dist_embed_nlayers
        self.dist_embed_residual = dist_embed_residual
        self.dist_embed_nunits = dist_embed_nunits
        self.dist_embed_final_nunits = dist_embed_final_nunits
        self.dist_embed_0_dropout_rate = dist_embed_0_dropout_rate
        self.dist_embed_1_dropout_rate = dist_embed_1_dropout_rate
        self.dist_embed_0_l1_rate = dist_embed_0_l1_rate
        self.dist_embed_1_l1_rate = dist_embed_1_l1_rate
        self.dist_embed_0_l2_rate = dist_embed_0_l2_rate
        self.dist_embed_1_l2_rate = dist_embed_1_l2_rate
        self.dist_embed_activation = dist_embed_activation
        self.group_embed_nlayers = group_embed_nlayers
        self.group_embed_residual = group_embed_residual
        self.group_embed_nunits = group_embed_nunits
        self.group_embed_final_nunits = group_embed_final_nunits
        self.group_embed_0_dropout_rate = group_embed_0_dropout_rate
        self.group_embed_1_dropout_rate = group_embed_1_dropout_rate
        self.group_embed_0_l1_rate = group_embed_0_l1_rate
        self.group_embed_1_l1_rate = group_embed_1_l1_rate
        self.group_embed_0_l2_rate = group_embed_0_l2_rate
        self.group_embed_1_l2_rate = group_embed_1_l2_rate
        self.group_embed_activation = group_embed_activation
        self.covariate_embed_nlayers = covariate_embed_nlayers
        self.covariate_embed_residual = covariate_embed_residual
        self.covariate_embed_nunits = covariate_embed_nunits
        self.covariate_embed_final_nunits = covariate_embed_final_nunits
        self.covariate_embed_0_dropout_rate = covariate_embed_0_dropout_rate
        self.covariate_embed_1_dropout_rate = covariate_embed_1_dropout_rate
        self.covariate_embed_0_l1_rate = covariate_embed_0_l1_rate
        self.covariate_embed_1_l1_rate = covariate_embed_1_l1_rate
        self.covariate_embed_0_l2_rate = covariate_embed_0_l2_rate
        self.covariate_embed_1_l2_rate = covariate_embed_1_l2_rate
        self.covariate_embed_activation = covariate_embed_activation
        self.combine_method = combine_method
        self.post_embed_nlayers = post_embed_nlayers
        self.post_embed_residual = post_embed_residual
        self.post_embed_nunits = post_embed_nunits
        self.post_embed_final_nunits = post_embed_final_nunits
        self.post_embed_0_dropout_rate = post_embed_0_dropout_rate
        self.post_embed_1_dropout_rate = post_embed_1_dropout_rate
        self.post_embed_0_l1_rate = post_embed_0_l1_rate
        self.post_embed_1_l1_rate = post_embed_1_l1_rate
        self.post_embed_0_l2_rate = post_embed_0_l2_rate
        self.post_embed_1_l2_rate = post_embed_1_l2_rate
        self.post_embed_activation = post_embed_activation
        self.use_predictor = use_predictor
        self.predictor_l1_rate = predictor_l1_rate
        self.predictor_l2_rate = predictor_l2_rate
        self.predictor_dropout_rate = predictor_dropout_rate
        self.predictor_use_bias = predictor_use_bias
        self.hard_triplet_loss_rate = hard_triplet_loss_rate
        self.semihard_triplet_loss_rate = semihard_triplet_loss_rate
        self.multisurf_loss_rate = multisurf_loss_rate
        self.input_names = input_names
        return

    def _keras_build_fn(self, compile_kwargs: "Dict[str, Any]"):

        from .losses import RankLoss
        from .models import ConvMLP

        if self.target_type_ in ("continuous", "binary"):
            n_output_units = 1
        elif self.target_type_ in (
            "continous-multioutput",
            "multiclass",
            "multiclass-multioutput",
            "multilabel-indicator"
        ):
            n_output_units = self.n_classes
        else:
            raise ValueError("We can't handle this target type")

        assert self.loss is not None

        if self.loss in ("mse", "mae", "binary_crossentropy"):
            loss = self.loss
        elif self.loss == "pairwise":
            loss = RankLoss()
        else:
            raise ValueError(
                "We currently only support mse, mae, "
                "binary_classification, and pairwise ranking losses"
            )

        if self.loss == "binary_crossentropy":
            activation: "Literal['sigmoid', 'linear', 'softmax']" = "sigmoid"
        else:
            activation = "linear"

        assert activation in ("sigmoid", "linear", "softmax")

        model = ConvMLP(
            predictor_nunits=n_output_units,
            conv_nlayers=self.conv_nlayers,
            conv_filters=self.conv_filters,
            conv_strides=self.conv_strides,
            conv_kernel_size=self.conv_kernel_size,
            conv_activation=self.conv_activation,
            conv_l1_rate=self.conv_l1_rate,
            conv_l2_rate=self.conv_l2_rate,
            conv_use_batchnorm=self.conv_use_batchnorm,
            adaptive_l1=self.adaptive_l1,
            adaptive_l1_rate=self.adaptive_l1_rate,
            adaptive_l2_rate=self.adaptive_l2_rate,
            marker_embed_nlayers=self.marker_embed_nlayers,
            marker_embed_residual=self.marker_embed_residual,
            marker_embed_nunits=self.marker_embed_nunits,
            marker_embed_final_nunits=self.marker_embed_final_nunits,
            marker_embed_0_dropout_rate=self.marker_embed_0_dropout_rate,
            marker_embed_1_dropout_rate=self.marker_embed_1_dropout_rate,
            marker_embed_0_l1_rate=self.marker_embed_0_l1_rate,
            marker_embed_1_l1_rate=self.marker_embed_1_l1_rate,
            marker_embed_0_l2_rate=self.marker_embed_0_l2_rate,
            marker_embed_1_l2_rate=self.marker_embed_1_l2_rate,
            marker_embed_activation=self.marker_embed_activation,
            dist_embed_nlayers=self.dist_embed_nlayers,
            dist_embed_residual=self.dist_embed_residual,
            dist_embed_nunits=self.dist_embed_nunits,
            dist_embed_final_nunits=self.dist_embed_final_nunits,
            dist_embed_0_dropout_rate=self.dist_embed_0_dropout_rate,
            dist_embed_1_dropout_rate=self.dist_embed_1_dropout_rate,
            dist_embed_0_l1_rate=self.dist_embed_0_l1_rate,
            dist_embed_1_l1_rate=self.dist_embed_1_l1_rate,
            dist_embed_0_l2_rate=self.dist_embed_0_l2_rate,
            dist_embed_1_l2_rate=self.dist_embed_1_l2_rate,
            dist_embed_activation=self.dist_embed_activation,
            group_embed_nlayers=self.group_embed_nlayers,
            group_embed_residual=self.group_embed_residual,
            group_embed_nunits=self.group_embed_nunits,
            group_embed_final_nunits=self.group_embed_final_nunits,
            group_embed_0_dropout_rate=self.group_embed_0_dropout_rate,
            group_embed_1_dropout_rate=self.group_embed_1_dropout_rate,
            group_embed_0_l1_rate=self.group_embed_0_l1_rate,
            group_embed_1_l1_rate=self.group_embed_1_l1_rate,
            group_embed_0_l2_rate=self.group_embed_0_l2_rate,
            group_embed_1_l2_rate=self.group_embed_1_l2_rate,
            group_embed_activation=self.group_embed_activation,
            covariate_embed_nlayers=self.covariate_embed_nlayers,
            covariate_embed_residual=self.covariate_embed_residual,
            covariate_embed_nunits=self.covariate_embed_nunits,
            covariate_embed_final_nunits=self.covariate_embed_final_nunits,
            covariate_embed_0_dropout_rate=self.covariate_embed_0_dropout_rate,
            covariate_embed_1_dropout_rate=self.covariate_embed_1_dropout_rate,
            covariate_embed_0_l1_rate=self.covariate_embed_0_l1_rate,
            covariate_embed_1_l1_rate=self.covariate_embed_1_l1_rate,
            covariate_embed_0_l2_rate=self.covariate_embed_0_l2_rate,
            covariate_embed_1_l2_rate=self.covariate_embed_1_l2_rate,
            covariate_embed_activation=self.covariate_embed_activation,
            combine_method=self.combine_method,
            post_embed_nlayers=self.post_embed_nlayers,
            post_embed_residual=self.post_embed_residual,
            post_embed_nunits=self.post_embed_nunits,
            post_embed_final_nunits=self.post_embed_final_nunits,
            post_embed_0_dropout_rate=self.post_embed_0_dropout_rate,
            post_embed_1_dropout_rate=self.post_embed_1_dropout_rate,
            post_embed_0_l1_rate=self.post_embed_0_l1_rate,
            post_embed_1_l1_rate=self.post_embed_1_l1_rate,
            post_embed_0_l2_rate=self.post_embed_0_l2_rate,
            post_embed_1_l2_rate=self.post_embed_1_l2_rate,
            post_embed_activation=self.post_embed_activation,
            use_predictor=self.use_predictor,
            predictor_activation=activation,
            predictor_l1_rate=self.predictor_l1_rate,
            predictor_l2_rate=self.predictor_l2_rate,
            predictor_dropout_rate=self.predictor_dropout_rate,
            predictor_use_bias=self.predictor_use_bias,
            hard_triplet_loss_rate=self.hard_triplet_loss_rate,
            semihard_triplet_loss_rate=self.semihard_triplet_loss_rate,
            multisurf_loss_rate=self.multisurf_loss_rate,
            input_names=self.input_names,
        )

        model.compile(
            loss=loss,
            optimizer=compile_kwargs.get("optimizer", "adam")
        )
        model.optimizer.learning_rate.assign(self.optimizer__learning_rate)
        return model


class ConvMLPClassifier(ConvMLPWrapper, ClassifierMixin):

    def __init__(
        self,
        loss="binary_crossentropy",
        name="conv_mlp_classifier",
        **kwargs
    ):
        super().__init__(loss=loss, name=name, **kwargs)


class ConvMLPRegressor(ConvMLPWrapper, RegressorMixin):

    def __init__(self, loss="mse", name="conv_mlp_regressor", **kwargs):
        super().__init__(loss=loss, name=name, **kwargs)


class ConvMLPRanker(ConvMLPWrapper):

    _estimator_type = "ranker"

    def __init__(self, loss="pairwise", name="conv_mlp_ranker", **kwargs):
        super().__init__(loss=loss, name=name, **kwargs)

    def score(self, X, y, sample_weight=None):
        from ..sk.metrics import spearmans_correlation

        y_pred = self.predict(X)
        return spearmans_correlation(y, y_pred)

    def _more_tags(self):
        return {"requires_y": True}
