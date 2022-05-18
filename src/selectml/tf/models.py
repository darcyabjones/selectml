from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from typing import Literal

import tensorflow as tf

from .losses import (
    SemiHardTripletLoss,
    HardTripletLoss,
    MultiSURFTripletLoss,
)


class ConvMLP(tf.keras.models.Model):

    def __init__(
        self,
        predictor_nunits: int,
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
        predictor_activation: "Literal['linear', 'sigmoid', 'softmax']" = "linear",  # noqa
        predictor_l1_rate: float = 0.0,
        predictor_l2_rate: float = 0.0,
        predictor_dropout_rate: float = 0.0,
        predictor_use_bias: bool = True,
        hard_triplet_loss_rate: "Optional[float]" = None,
        semihard_triplet_loss_rate: "Optional[float]" = None,
        multisurf_loss_rate: "Optional[float]" = None,
        name="conv_mlp",
        **kwargs
    ):
        super(ConvMLP, self).__init__(name=name, **kwargs)

        self.predictor_nunits = predictor_nunits
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
        self.predictor_activation = predictor_activation
        self.predictor_l1_rate = predictor_l1_rate
        self.predictor_l2_rate = predictor_l2_rate
        self.predictor_dropout_rate = predictor_dropout_rate
        self.predictor_use_bias = predictor_use_bias
        self.hard_triplet_loss_rate = hard_triplet_loss_rate
        self.semihard_triplet_loss_rate = semihard_triplet_loss_rate
        self.multisurf_loss_rate = multisurf_loss_rate

        self._get_custom_losses()

        self.built = False
        return

    def _get_custom_losses(self):
        self.semihard_triplet_loss = SemiHardTripletLoss(
            name="semihard_triplet"
        )
        self.semihard_triplet_loss_tracker = tf.metrics.Mean(
            name="semihard_triplet_loss"
        )

        self.hard_triplet_loss = HardTripletLoss(
            name="hard_triplet"
        )
        self.hard_triplet_loss_tracker = tf.metrics.Mean(
            name="hard_triplet_loss"
        )

        self.multisurf_loss = MultiSURFTripletLoss(
            name="multisurf"
        )
        self.multisurf_loss_tracker = tf.metrics.Mean(
            name="multisurf_loss"
        )
        return

    def _check_all_same_final_units(
        self,
        markers,
        dists,
        groups,
        covariates
    ):
        if self.marker_embed_final_nunits is None:
            m = self.marker_embed_nunits
        else:
            m = self.marker_embed_final_nunits

        if self.dist_embed_final_nunits is None:
            d = self.dist_embed_nunits
        else:
            d = self.dist_embed_final_nunits

        if self.group_embed_final_nunits is None:
            g = self.group_embed_nunits
        else:
            g = self.group_embed_final_nunits

        if self.covariate_embed_final_nunits is None:
            c = self.covariate_embed_nunits
        else:
            c = self.covariate_embed_final_nunits

        embed_units = []
        nlayers = []

        if markers is not None:
            embed_units.append(m)
            nlayers.append(self.marker_embed_nlayers)

        if dists is not None:
            embed_units.append(d)
            nlayers.append(self.dist_embed_nlayers)

        if groups is not None:
            embed_units.append(g)
            nlayers.append(self.group_embed_nlayers)

        if covariates is not None:
            embed_units.append(c)
            nlayers.append(self.covariate_embed_nlayers)

        if any([n == 0 for n in nlayers]):
            raise ValueError(
                f"When combining with {self.combine_method}, "
                "the embedder layers must all have at least one layer."
            )

        if len(embed_units) == 0:
            raise ValueError("Recieved only None values")

        if any([e != embed_units[0] for e in embed_units]):
            raise ValueError(
                f"When combining with {self.combine_method}, "
                "all of the embedders must output the same length."
            )
        return embed_units[0]

    def build(self, input_shape):  # noqa
        from tensorflow.keras.layers import Concatenate, Add

        markers = input_shape.get("markers", None)
        dists = input_shape.get("dists", None)
        groups = input_shape.get("groups", None)
        covariates = input_shape.get("covariates", None)

        self.drop_markers = markers is None
        self.drop_dists = dists is None
        self.drop_groups = groups is None
        self.drop_covariates = covariates is None

        if markers is None:
            self.conv_nlayers = 0
            self.adaptive_l1 = False
            self.marker_embed_nlayers = 0

        if dists is None:
            self.dist_embed_nlayers = 0

        if groups is None:
            self.group_embed_nlayers = 0

        if covariates is None:
            self.covariate_embed_nlayers = 0

        if self.combine_method != "concatenate":
            embed_units = self._check_all_same_final_units(
                markers,
                dists,
                groups,
                covariates
            )

            if (
                (self.post_embed_nlayers == 0)
                and not self.use_predictor
                and (embed_units != self.predictor_nunits)
            ):
                raise ValueError(
                    "If skipping post embedding and predictor layers, "
                    "the final units of all layers must be the number "
                    "of output targets."
                )
        elif (
            (self.post_embed_nlayers == 0)
            and not self.use_predictor
        ):
            raise ValueError(
                "When skipping post embedding and predictor layers, "
                "the individual embedders must be combined using add."
            )
        elif not self.use_predictor:
            if self.post_embed_final_nunits is None:
                post_embed_final_nunits = self.post_embed_nunits
            else:
                post_embed_final_nunits = self.post_embed_final_nunits

            if post_embed_final_nunits != self.predictor_nunits:
                raise ValueError(
                    "If skipping the predictor layer, the final units "
                    "of the post embedder must be the same as the number "
                    "of ouput targets."
                )

        self.marker_layers = []
        self.marker_layers.extend(self._prep_conv(name="marker_conv"))
        self.marker_layers.extend(self._prep_adaptive(name="marker_adaptive"))
        self.marker_layers.extend(self._prep_embed(name="marker_embed"))

        self.dist_layers = self._prep_embed(name="dist_embed")
        self.group_layers = self._prep_embed(name="group_embed")
        self.covariate_layers = self._prep_embed(name="covariate_embed")
        self.post_layers = self._prep_embed(name="post_embed")

        if self.combine_method == "concatenate":
            self.combiner = Concatenate(name=f"{self.name}/combiner")
        elif self.combine_method == "add":
            self.combiner = Add(name=f"{self.name}/combiner")
        else:
            raise ValueError("The combine method must be concatenate or add.")

        self.predictor_layers = self._prep_predictor(name="predictor")
        self.built = True
        return

    def _prep_conv(self, name="conv"):
        from tensorflow.keras.regularizers import L1, L2, L1L2
        from selectml.tf.layers import (
            AddChannel,
            Flatten1D,
            ConvLinkage,
        )

        layers = []
        if self.conv_nlayers > 0:
            if (self.conv_l1_rate > 0.0) and (self.conv_l2_rate > 0.0):
                reg = L1L2(self.conv_l1_rate, self.conv_l2_rate)
            elif self.conv_l1_rate > 0.0:
                reg = L1(self.conv_l1_rate)
            elif self.conv_l2_rate > 0.0:
                reg = L2(self.conv_l2_rate)
            else:
                reg = None

            layers.extend([
                AddChannel(name=f"{self.name}/{name}/addchannel"),
                ConvLinkage(
                    nlayers=self.conv_nlayers,
                    filters=self.conv_filters,
                    strides=self.conv_strides,
                    kernel_size=self.conv_kernel_size,
                    activation=self.conv_activation,
                    activation_first=self.conv_activation,
                    activation_last=self.conv_activation,
                    kernel_regularizer=reg,
                    use_bn=self.conv_use_batchnorm,
                    name=f"{self.name}/{name}/convlinkage"
                ),
                Flatten1D(name=f"{self.name}/{name}/flatten1d"),
            ])

        return layers

    def _prep_adaptive(self, name="adaptive"):
        from selectml.tf.layers import (
            LocalLasso,
            AddChannel,
            Flatten1D,
        )
        from selectml.tf.regularizers import AdaptiveL1L2Regularizer

        if self.adaptive_l1:
            reg = AdaptiveL1L2Regularizer(
                l1=self.adaptive_l1_rate,
                l2=self.adaptive_l2_rate,
                adapt=True,
            )
            return [
                AddChannel(name=f"{self.name}/{name}/addchannel"),
                LocalLasso(
                    kernel_regularizer=reg,
                    name=f"{self.name}/{name}/locallasso"
                ),
                Flatten1D(name=f"{self.name}/{name}/flatten1d")
            ]
        else:
            return []

    def _prep_embed(self, name="embed"):
        from tensorflow.keras.regularizers import L1, L2, L1L2
        from tensorflow.keras.layers import (
            Dropout,
            Dense,
            BatchNormalization,
            ReLU
        )
        from selectml.tf.layers import (
            ParallelResidualUnit,
            ResidualUnit
        )

        embed_nlayers = getattr(self, f"{name}_nlayers")
        embed_residual = getattr(self, f"{name}_residual")
        embed_nunits = getattr(self, f"{name}_nunits")
        embed_final_nunits = getattr(self, f"{name}_final_nunits")
        embed_0_dropout_rate = getattr(self, f"{name}_0_dropout_rate")
        embed_1_dropout_rate = getattr(self, f"{name}_1_dropout_rate")
        embed_0_l1_rate = getattr(self, f"{name}_0_l1_rate")
        embed_1_l1_rate = getattr(self, f"{name}_1_l1_rate")
        embed_0_l2_rate = getattr(self, f"{name}_0_l2_rate")
        embed_1_l2_rate = getattr(self, f"{name}_1_l2_rate")
        embed_activation = getattr(self, f"{name}_activation")

        if embed_final_nunits is None:
            embed_final_nunits_ = embed_nunits
        else:
            embed_final_nunits_ = embed_final_nunits

        layers = []
        for i in range(embed_nlayers):
            dr = embed_0_dropout_rate if (i == 0) else embed_1_dropout_rate
            layers.append(Dropout(dr, name=f"{self.name}/{name}/dropout.{i}"))

            l1 = embed_0_l1_rate if (i == 0) else embed_1_l1_rate
            l2 = embed_0_l2_rate if (i == 0) else embed_1_l2_rate

            if (l1 > 0.0) and (l2 > 0.0):
                reg = L1L2(l1, l2)
            elif (l1 > 0.0):
                reg = L1(l1)
            elif (l2 > 0.0):
                reg = L2(l2)
            else:
                reg = None

            if i >= (embed_nlayers - 1):
                nunits = embed_final_nunits_
            else:
                nunits = embed_nunits

            if embed_residual:
                if i == 0:
                    type_ = ParallelResidualUnit
                    type_name = "parallel_residual_unit"
                else:
                    type_ = ResidualUnit
                    type_name = "residual_unit"

                layers.append(type_(
                    nunits,
                    dr,
                    activation=embed_activation,
                    use_bias=True,
                    nonlinear_kernel_regularizer=reg,
                    gain_kernel_regularizer=reg,
                    linear_kernel_regularizer=reg,
                    name=f"{self.name}/{name}/{type_name}.{i}"
                ))
            else:
                layers.append(Dense(
                    nunits,
                    activation="linear",
                    kernel_regularizer=reg,
                    use_bias=True,
                    name=f"{self.name}/{name}/dense.{i}"
                ))
                layers.append(BatchNormalization(
                    name=f"{self.name}/{name}/batchnormalization.{i}"
                ))

                # The current trend seems to be to do activation after
                # batch/layer norm. This might change in future.
                if embed_activation == "relu":
                    layers.append(ReLU(
                        name=f"{self.name}/{name}/relu.{i}"
                    ))
        return layers

    def _prep_predictor(self, name="predictor"):
        from tensorflow.keras.regularizers import L1, L2, L1L2
        from tensorflow.keras.layers import (
            Dropout,
            Dense,
            Activation,
        )

        layers = []
        if self.use_predictor and (self.predictor_dropout_rate > 0.0):
            layers.append(Dropout(
                self.predictor_dropout_rate,
                name=f"{self.name}/{name}/dropout"
            ))

        if (self.predictor_l1_rate > 0.0) and (self.predictor_l2_rate > 0.0):
            reg = L1L2(self.predictor_l1_rate, self.predictor_l2_rate)
        elif self.predictor_l1_rate > 0.0:
            reg = L1(self.predictor_l1_rate)
        elif self.predictor_l2_rate > 0.0:
            reg = L2(self.predictor_l2_rate)
        else:
            reg = None

        if self.use_predictor:
            layers.append(
                Dense(
                    self.predictor_nunits,
                    activation="linear",
                    kernel_regularizer=reg,
                    use_bias=self.predictor_use_bias,
                    name=f"{self.name}/{name}/dense"
                )
            )

        if self.predictor_activation != "linear":
            layers.append(Activation(
                self.predictor_activation,
                name=f"{self.name}/{name}/activation"
            ))

        return layers

    @tf.function
    def get_marker_embed_output(self, X, training=False):
        if X is not None:
            for layer in self.marker_layers:
                X = layer(X, training=training)

        return X

    @tf.function
    def get_dist_embed_output(self, X, training=False):
        if X is not None:
            for layer in self.dist_layers:
                X = layer(X, training=training)

        return X

    @tf.function
    def get_group_embed_output(self, X, training=False):
        if X is not None:
            for layer in self.group_layers:
                X = layer(X, training=training)

        return X

    @tf.function
    def get_covariate_embed_output(self, X, training=False):
        if X is not None:
            for layer in self.covariate_layers:
                X = layer(X, training=training)

        return X

    def get_combined_output(
        self,
        markers,
        dists,
        groups,
        covariates,
        training=False
    ):
        to_combine = []
        if markers is not None:
            to_combine.append(markers)

        if dists is not None:
            to_combine.append(dists)

        if groups is not None:
            to_combine.append(groups)

        if covariates is not None:
            to_combine.append(covariates)

        combined = self.combiner(to_combine)
        return combined

    @tf.function
    def get_post_embed_output(self, X, training=False):
        # X should not be None
        for layer in self.post_layers:
            X = layer(X, training=training)

        return X

    @tf.function
    def get_predictor_output(self, X, training=False):
        for layer in self.predictor_layers:
            X = layer(X, training=training)

        return X

    @tf.function
    def long_call(self, inputs, training=False):
        from selectml.higher import fmap

        # Since we call this function directly in train_step,
        # we need to "build" the model by calling __call__
        if not self.built:
            _ = self(inputs)

        assert isinstance(inputs, dict) and len(inputs) > 0

        # I don't love this but it's my only way I think
        markers = inputs.get("markers", None)
        dists = inputs.get("dists", None)
        groups = inputs.get("groups", None)
        covariates = inputs.get("covariates", None)

        if self.drop_markers:
            markers = None

        if self.drop_dists:
            dists = None

        if self.drop_groups:
            groups is None

        if self.drop_covariates:
            covariates is None

        # Fmaps are necessary to avoid converting None to a tensorflow
        # placeholder type.
        markers = fmap(
            self.get_marker_embed_output,
            markers,
            training=training
        )
        dists = fmap(self.get_dist_embed_output, dists, training=training)
        groups = fmap(self.get_group_embed_output, groups, training=training)
        covariates = fmap(
            self.get_covariate_embed_output,
            covariates,
            training=training
        )

        combined = self.get_combined_output(
            markers,
            dists,
            groups,
            covariates,
            training=training
        )

        post = self.get_post_embed_output(combined, training=training)
        predictions = self.get_predictor_output(post, training=training)
        return markers, dists, groups, covariates, combined, post, predictions

    def call(self, inputs, training=False):
        _, _, _, _, _, _, yhat = self.long_call(inputs, training=training)
        return yhat

    def train_step(self, inputs):

        if len(inputs) == 3:
            X, y, sample_weight = inputs
        else:
            sample_weight = None
            X, y = inputs

        with tf.GradientTape() as tape:
            (
                markers,
                dists,
                groups,
                covariates,
                combined,
                post,
                preds
            ) = self.long_call(
                X,
                training=True
            )

            regularization_losses = list(self.losses)
            semihard_loss = self.semihard_triplet_loss(y, post)
            hard_loss = self.hard_triplet_loss(y, post)
            multisurf_loss = self.multisurf_loss(y, post)

            if isinstance(self.semihard_triplet_loss_rate, float):
                regularization_losses.append(
                    self.semihard_triplet_loss_rate *
                    tf.reduce_mean(semihard_loss)
                )

            if isinstance(self.hard_triplet_loss_rate, float):
                regularization_losses.append(
                    self.hard_triplet_loss_rate *
                    tf.reduce_mean(hard_loss)
                )

            if isinstance(self.multisurf_loss_rate, float):
                regularization_losses.append(
                    self.multisurf_loss_rate *
                    tf.reduce_mean(multisurf_loss)
                )

            loss = self.compiled_loss(
                y,
                preds,
                sample_weight=sample_weight,
                regularization_losses=regularization_losses
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(
            y,
            preds,
            sample_weight=sample_weight
        )

        metrics = {m.name: m.result() for m in self.metrics}

        if self.semihard_triplet_loss_tracker is not None:
            self.semihard_triplet_loss_tracker.update_state(
                tf.reduce_mean(semihard_loss)
            )
            metrics["semihard_triplet_loss"] = (
                self.semihard_triplet_loss_tracker.result()
            )

        if self.hard_triplet_loss_tracker is not None:
            self.hard_triplet_loss_tracker.update_state(
                tf.reduce_mean(hard_loss)
            )
            metrics["hard_triplet_loss"] = (
                self.hard_triplet_loss_tracker.result()
            )

        if self.multisurf_loss_tracker is not None:
            self.multisurf_loss_tracker.update_state(
                tf.reduce_mean(multisurf_loss)
            )
            metrics["multisurf_loss"] = (
                self.multisurf_loss_tracker.result()
            )
        return metrics

    @property
    def metrics(self):
        metrics = []
        if self._is_compiled:
            if self.compiled_loss is not None:
                metrics += self.compiled_loss.metrics
            if self.compiled_metrics is not None:
                metrics += self.compiled_metrics.metrics

        for layer in self._flatten_layers():
            metrics.extend(layer._metrics)  # pylint: disable=protected-access

        for attr in [
            "semihard_triplet_loss_tracker",
            "hard_triplet_loss_tracker",
            "multisurf_loss_tracker",
        ]:
            if getattr(self, attr) is not None:
                metrics.append(getattr(self, attr))

        return metrics

    def test_step(self, inputs):
        X, y = inputs

        (
            markers,
            dists,
            groups,
            covariates,
            combined,
            post,
            preds
        ) = self.long_call(
            X,
            training=False
        )

        regularization_losses = list(self.losses)
        semihard_loss = self.semihard_triplet_loss(y, post)
        hard_loss = self.hard_triplet_loss(y, post)
        multisurf_loss = self.multisurf_loss(y, post)

        if isinstance(self.semihard_triplet_loss_rate, float):
            regularization_losses.append(
                self.semihard_triplet_loss_rate *
                tf.reduce_mean(semihard_loss)
            )

        if isinstance(self.hard_triplet_loss_rate, float):
            regularization_losses.append(
                self.hard_triplet_loss_rate *
                tf.reduce_mean(hard_loss)
            )

        if isinstance(self.multisurf_loss_rate, float):
            regularization_losses.append(
                self.multisurf_loss_rate *
                tf.reduce_mean(multisurf_loss)
            )

        _ = self.compiled_loss(
            y,
            preds,
            regularization_losses=regularization_losses
        )

        self.compiled_metrics.update_state(
            y,
            preds,
        )

        metrics = {m.name: m.result() for m in self.metrics}

        if self.semihard_triplet_loss_tracker is not None:
            self.semihard_triplet_loss_tracker.update_state(
                tf.reduce_mean(semihard_loss)
            )
            metrics["semihard_triplet_loss"] = (
                self.semihard_triplet_loss_tracker.result()
            )

        if self.hard_triplet_loss_tracker is not None:
            self.hard_triplet_loss_tracker.update_state(
                tf.reduce_mean(hard_loss)
            )
            metrics["hard_triplet_loss"] = (
                self.hard_triplet_loss_tracker.result()
            )

        if self.multisurf_loss_tracker is not None:
            self.multisurf_loss_tracker.update_state(
                tf.reduce_mean(multisurf_loss)
            )
            metrics["multisurf_loss"] = (
                self.multisurf_loss_tracker.result()
            )
        return metrics


"""
class SSModel(tf.keras.Model):

    def __init__(
        self,
        marker_embedder: tf.keras.Layer,
        dists_embedder: "Optional[tf.keras.Layer]" = None,
        groups_embedder: "Optional[tf.keras.Layer]" = None,
        covariates_embedder: "Optional[tf.keras.Layer]" = None,
        post_embedder: "Optional[tf.keras.Layer]" = None,
        env_hard_loss: "Optional[float]" = None,
        env_semihard_loss: "Optional[float]" = None,
        marker_hard_loss: "Optional[float]" = None,
        marker_semihard_loss: "Optional[float]" = None,
        marker_relief_loss: "Optional[float]" = None,
        rank_loss: "Optional[float]" = None,
        sd: float = 1,
    ):
        super(SSModel, self).__init__()

        self.marker_embedder = marker_embedder
        self.dists_embedder = dists_embedder
        self.groups_embedder = groups_embedder
        self.covariates_embedder = covariates_embedder
        self.post_embedder = post_embedder

        self.marker_hard_loss_rate = marker_hard_loss
        self.marker_semihard_loss_rate = marker_semihard_loss
        self.marker_relief_loss_rate = marker_relief_loss

        self.env_hard_loss_rate = env_hard_loss
        self.env_semihard_loss_rate = env_semihard_loss

        self.rank_loss_rate = rank_loss

        self.marker_semihard_triplet_loss = SemiHardTripletLoss(
            sd=sd,
            name="marker_semihard_triplet"
        )
        self.marker_semihard_triplet_loss_tracker = tf.metrics.Mean(
            name="marker_semihard_triplet_loss"
        )

        self.marker_hard_triplet_loss = HardTripletLoss(
            sd=sd,
            name="marker_hard_triplet"
        )
        self.marker_hard_triplet_loss_tracker = tf.metrics.Mean(
            name="marker_hard_triplet_loss"
        )

        self.marker_relief_loss = MultiSURFTripletLoss(
            sd=sd,
            name="marker_relief"
        )
        self.marker_relief_loss_tracker = tf.metrics.Mean(
            name="marker_relief_loss"
        )

        if self.groups_embedder is None:
            self.groups_semihard_triplet_loss = None
            self.groups_semihard_triplet_loss_tracker = None

            self.groups_hard_triplet_loss = None
            self.groups_hard_triplet_loss_tracker = None
        else:
            self.groups_semihard_triplet_loss = SemiHardTripletLoss(
                sd=sd,
                name="groups_semihard_triplet"
            )
            self.groups_semihard_triplet_loss_tracker = tf.metrics.Mean(
                name="groups_semihard_triplet_loss"
            )

            self.groups_hard_triplet_loss = HardTripletLoss(
                sd=sd,
                name="groups_hard_triplet"
            )
            self.env_hard_triplet_loss_tracker = tf.metrics.Mean(
                name="groups_hard_triplet_loss"
            )

        self.rank_loss = RankLoss(name="rank")
        self.rank_loss_tracker = tf.metrics.Mean(name="rank_loss")

        self.combiner = "concat"
        if self.combiner == "add":
            self.combine_layer = tf.keras.layers.Add()
        if self.combiner == "concat":
            self.combine_layer = tf.keras.layers.Concatenate()

        return

    def train_step(self, data):  # noqa
        if len(data) == 3:
            X, y, sample_weight = data
        else:
            sample_weight = None
            X, y = data

        # Need to filter for semihard tf.reduce_sum(loss) +
        with tf.GradientTape() as tape:
            marker_embed, env_embed, preds, ranks = self.long_call(
                X,
                training=True
            )

            regularization_losses = list(self.losses)
            if marker_embed is not None:
                marker_semihard_triplet_loss = self.marker_semihard_triplet_loss(  # noqa
                    y,
                    marker_embed
                )
                marker_hard_triplet_loss = self.marker_hard_triplet_loss(
                    y,
                    marker_embed
                )
                marker_relief_loss = self.marker_relief_loss(
                    y,
                    marker_embed
                )

                if isinstance(self.marker_semihard_loss_rate, float):
                    regularization_losses.append(
                        self.marker_semihard_loss_rate *
                        tf.reduce_mean(marker_semihard_triplet_loss)
                    )

                if isinstance(self.marker_hard_loss_rate, float):
                    regularization_losses.append(
                        self.marker_hard_loss_rate *
                        tf.reduce_mean(marker_hard_triplet_loss)
                    )

                if isinstance(self.marker_relief_loss_rate, float):
                    regularization_losses.append(
                        self.marker_relief_loss_rate *
                        tf.reduce_mean(marker_relief_loss)
                    )

            if env_embed is not None:
                env_semihard_triplet_loss = self.env_semihard_triplet_loss(
                    y,
                    env_embed
                )
                env_hard_triplet_loss = self.env_hard_triplet_loss(
                    y,
                    env_embed
                )

                if isinstance(self.env_semihard_loss_rate, float):
                    regularization_losses.append(
                        self.env_semihard_loss_rate *
                        tf.reduce_mean(env_semihard_triplet_loss)
                    )

                if isinstance(self.env_hard_loss_rate, float):
                    regularization_losses.append(
                        self.env_hard_loss_rate *
                        tf.reduce_mean(env_hard_triplet_loss)
                    )

            if (ranks is not None) and (self.rank_loss_rate is not None):
                rank_loss = self.rank_loss(y, ranks)
                regularization_losses.append(
                    self.rank_loss_rate *
                    tf.reduce_mean(rank_loss)
                )
            else:
                rank_loss = None

            loss = self.compiled_loss(
                y,
                preds,
                sample_weight=sample_weight,
                regularization_losses=regularization_losses
            )

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(
            y,
            preds,
            sample_weight=sample_weight
        )
        metrics = {m.name: m.result() for m in self.metrics}

        if self.marker_semihard_triplet_loss_tracker is not None:
            self.marker_semihard_triplet_loss_tracker.update_state(
                tf.reduce_mean(marker_semihard_triplet_loss))
            metrics["marker_semihard_triplet_loss"] = (
                self.marker_semihard_triplet_loss_tracker.result()
            )

        if self.marker_hard_triplet_loss_tracker is not None:
            self.marker_hard_triplet_loss_tracker.update_state(
                tf.reduce_mean(marker_hard_triplet_loss))
            metrics["marker_hard_triplet_loss"] = (
                self.marker_hard_triplet_loss_tracker.result()
            )

        if self.marker_relief_loss_tracker is not None:
            self.marker_relief_loss_tracker.update_state(
                tf.reduce_mean(marker_relief_loss))
            metrics["marker_relief_loss"] = (
                self.marker_relief_loss_tracker.result()
            )

        if self.env_semihard_triplet_loss_tracker is not None:
            self.env_semihard_triplet_loss_tracker.update_state(
                tf.reduce_mean(env_semihard_triplet_loss))
            metrics["env_semihard_triplet_loss"] = (
                self.env_semihard_triplet_loss_tracker.result()
            )

        if self.env_hard_triplet_loss_tracker is not None:
            self.env_hard_triplet_loss_tracker.update_state(
                tf.reduce_mean(env_hard_triplet_loss))
            metrics["env_hard_triplet_loss"] = (
                self.env_hard_triplet_loss_tracker.result()
            )

        if rank_loss is not None:
            self.rank_loss_tracker.update_state(tf.reduce_mean(rank_loss))
            metrics["rank_loss"] = self.rank_loss_tracker.result()
        return metrics

    @property
    def metrics(self):
        metrics = []
        if self._is_compiled:
            if self.compiled_loss is not None:
                metrics += self.compiled_loss.metrics
            if self.compiled_metrics is not None:
                metrics += self.compiled_metrics.metrics

        for layer in self._flatten_layers():
            metrics.extend(layer._metrics)  # pylint: disable=protected-access

        for attr in [
            "marker_semihard_triplet_loss_tracker",
            "marker_hard_triplet_loss_tracker",
            "marker_relief_loss_tracker",
            "env_semihard_triplet_loss_tracker",
            "env_hard_triplet_loss_tracker",
            "rank_loss_tracker"
        ]:
            if getattr(self, attr) is not None:
                metrics.append(getattr(self, attr))

        return metrics

    def test_step(self, data):  # noqa
        X, y = data

        marker_embed, env_embed, preds, ranks = self.long_call(
            X,
            training=False
        )

        regularization_losses = list(self.losses)

        if marker_embed is not None:
            marker_semihard_triplet_loss = self.marker_semihard_triplet_loss(
                y,
                marker_embed
            )
            marker_hard_triplet_loss = self.marker_hard_triplet_loss(
                y,
                marker_embed
            )
            marker_relief_loss = self.marker_relief_loss(
                y,
                marker_embed
            )

            if isinstance(self.marker_semihard_loss_rate, float):
                regularization_losses.append(
                    self.marker_semihard_loss_rate *
                    tf.reduce_mean(marker_semihard_triplet_loss)
                )

            if isinstance(self.marker_hard_loss_rate, float):
                regularization_losses.append(
                    self.marker_hard_loss_rate *
                    tf.reduce_mean(marker_hard_triplet_loss)
                )

            if isinstance(self.marker_relief_loss_rate, float):
                regularization_losses.append(
                    self.marker_relief_loss_rate *
                    tf.reduce_mean(marker_relief_loss)
                )

        if env_embed is not None:
            env_semihard_triplet_loss = self.env_semihard_triplet_loss(
                y,
                env_embed
            )
            env_hard_triplet_loss = self.env_hard_triplet_loss(
                y,
                env_embed
            )

            if isinstance(self.env_semihard_loss_rate, float):
                regularization_losses.append(
                    self.env_semihard_loss_rate *
                    tf.reduce_mean(env_semihard_triplet_loss)
                )

            if isinstance(self.env_hard_loss_rate, float):
                regularization_losses.append(
                    self.env_hard_loss_rate *
                    tf.reduce_mean(env_hard_triplet_loss)
                )

        if (ranks is not None) and (self.rank_loss_rate is not None):
            rank_loss = self.rank_loss(y, ranks)
        else:
            rank_loss = None

        _ = self.compiled_loss(
            y,
            preds,
            regularization_losses=regularization_losses
        )

        self.compiled_metrics.update_state(
            y,
            preds,
        )
        metrics = {m.name: m.result() for m in self.metrics}

        if self.marker_semihard_triplet_loss_tracker is not None:
            self.marker_semihard_triplet_loss_tracker.update_state(
                tf.reduce_mean(marker_semihard_triplet_loss))
            metrics["marker_semihard_triplet_loss"] = (
                self.marker_semihard_triplet_loss_tracker.result()
            )

        if self.marker_hard_triplet_loss_tracker is not None:
            self.marker_hard_triplet_loss_tracker.update_state(
                tf.reduce_mean(marker_hard_triplet_loss))
            metrics["marker_hard_triplet_loss"] = (
                self.marker_hard_triplet_loss_tracker.result()
            )

        if self.marker_relief_loss_tracker is not None:
            self.marker_relief_loss_tracker.update_state(
                tf.reduce_mean(marker_relief_loss))
            metrics["marker_relief_loss"] = (
                self.marker_relief_loss_tracker.result()
            )

        if self.env_semihard_triplet_loss_tracker is not None:
            self.env_semihard_triplet_loss_tracker.update_state(
                tf.reduce_mean(env_semihard_triplet_loss))
            metrics["env_semihard_triplet_loss"] = (
                self.env_semihard_triplet_loss_tracker.result()
            )

        if self.env_hard_triplet_loss_tracker is not None:
            self.env_hard_triplet_loss_tracker.update_state(
                tf.reduce_mean(env_hard_triplet_loss))
            metrics["env_hard_triplet_loss"] = (
                self.env_hard_triplet_loss_tracker.result()
            )

        if rank_loss is not None:
            self.rank_loss_tracker.update_state(tf.reduce_mean(rank_loss))
            metrics["rank_loss"] = self.rank_loss_tracker.result()
        return metrics

    @tf.function
    def get_marker_embed_output(self, X, training=False):
        X = self.marker_embedder(X, training=training)
        X = tf.math.l2_normalize(X, axis=1)  # L2 normalize embeddings
        return X

    @tf.function
    def get_env_embed_output(self, X, training=False):
        X = self.env_embedder(X, training=training)
        X = tf.math.l2_normalize(X, axis=1)  # L2 normalize embeddings
        return X

    @tf.function
    def get_embed_output(self, X, training=False):
        if self.env_embedder is None:
            Xmarkers = X
            Xenv = None
        else:
            Xmarkers, Xenv = X
        marker_embed = self.get_marker_embed_output(
            Xmarkers,
            training=training
        )

        if Xenv is None:
            embed = marker_embed
        else:
            env_embed = self.get_env_embed_output(
                Xenv,
                training=training
            )

            embed = self.combine_layer([marker_embed, env_embed])
        return embed

    def call(self, X, training=False):
        _, _, out, _ = self.long_call(X, training=training)
        return out

    def rank(self, X, training=False):
        _, _, _, out = self.long_call(X, training=training)
        assert out is not None
        return out

    @tf.function
    def long_call(self, X, training=False):
        if self.env_embedder is None:
            Xmarkers = X
            Xenv = None
        else:
            Xmarkers, Xenv = X

        marker_embed = self.get_marker_embed_output(
            Xmarkers,
            training=training
        )

        if Xenv is None:
            embed = marker_embed
            env_embed = None
        else:
            env_embed = self.get_env_embed_output(
                Xenv,
                training=training
            )

            embed = self.combine_layer([marker_embed, env_embed])

        if self.post_embedder is not None:
            out = self.post_embedder(embed, training=training)
        else:
            out = embed

        rank_out = out

        return marker_embed, env_embed, out, rank_out
"""
