import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations


class AddChannel(layers.Layer):
    """
    Adds an extra channel after the last dimension of the tensor.
    Useful for converting (nsamples, nfeatures) to
    (nsamples, nfeatures, nchannels).

    The latter is expected by LocallyConnected1D etc.
    """

    def __init__(self, name="add_channel", **kwargs):
        super(AddChannel, self).__init__(name=name, **kwargs)
        return

    def build(self, input_shape):
        self.reshaper = layers.Reshape(
            (input_shape[1], 1),
            name=f"{self.name}/reshape"
        )
        return

    def call(self, inputs):
        return self.reshaper(inputs)


class Flatten1D(layers.Layer):
    """
    Converts (nsamples, nfeatures, nchannels) to (nsamples, nfeatures).
    I.e. drops the channels.

    Output can go into FC layers.
    """

    def __init__(
        self,
        pooling="max",
        name="flatten1d",
        **kwargs
    ):
        super(Flatten1D, self).__init__(name=name, **kwargs)
        assert pooling in ("max", "average")
        self.pooling = pooling
        return

    def pool(self, X):
        if self.pooling == "max":
            return tf.reduce_max(X, axis=-1)
        else:
            return tf.reduce_mean(X, axis=-1)

    def call(self, inputs):
        return self.pool(inputs)

    def get_config(self):
        config = super(Flatten1D, self).get_config()
        config.update({"pooling": self.pooling})
        return config


class LocalResidual(layers.Layer):
    def __init__(
        self,
        implementation=3,
        gain_params=None,
        nonlinear_params=None,
        use_batchnorm=False,
        drop_linear=False,
        name="local_residual",
        **kwargs
    ):
        super(LocalResidual, self).__init__(name=name, **kwargs)
        self.implementation = implementation
        self.use_batchnorm = use_batchnorm
        self.drop_linear = drop_linear

        if nonlinear_params is None:
            nonlinear_params = {
                "activation": "tanh",
                "use_bias": True,
            }
        self.nonlinear_params = nonlinear_params

        self.nonlinear = layers.LocallyConnected1D(
            filters=1,
            kernel_size=1,
            strides=1,
            implementation=implementation,
            name=f"{name}/nonlinear",
            **nonlinear_params,
        )

        if use_batchnorm:
            self.batchnorm = layers.BatchNormalization(
                name=f"{name}/batchnorm"
            )
        else:
            self.batchnorm = None

        if gain_params is None:
            gain_params = {
                "use_bias": True,
            }
        self.gain_params = gain_params

        self.gain = layers.LocallyConnected1D(
            filters=1,
            kernel_size=1,
            strides=1,
            implementation=implementation,
            name=f"{name}/gain",
            **gain_params,
        )

        if not drop_linear:
            self.add = layers.Add()
        return

    def call(self, X, training=False):
        nonlin = self.nonlinear(X)
        if self.use_batchnorm:
            nonlin = self.batchnorm(nonlin, training=training)
        nonlin = self.gain(nonlin)

        if self.drop_linear:
            return nonlin

        combined = self.add([nonlin, X])
        return combined

    def get_config(self):
        config = super(LocalResidual, self).get_config()
        config.update({
            "implementation": self.implementation,
            "use_batchnorm": self.use_batchnorm,
            "nonlinear_params": self.nonlinear_params,
            "gain_params": self.gain_params,
            "drop_linear": self.drop_linear,
        })
        return config


class LocalLinkage(layers.Layer):
    """ Progressively reduce closely co-located markers.

    Having separate activity and bias options for the last and first step
    is intended to be so that you can include a non-linearity, which would
    emulate dominance. I suspect bias would be important if you used a tanh
    activation for dominance. But including bias increases number of parameters
    a fair bit for this application.
    """

    def __init__(
        self,
        nlayers=3,
        filters=1,
        strides=2,
        kernel_size=2,
        implementation=3,
        use_bias_first=True,
        use_bias=False,
        use_bias_last=False,
        activation_first="linear",
        activation="linear",
        activation_last="linear",
        kernel_regularizer=None,
        activity_regularizer=None,
        bias_regularizer=None,
        dropout_rate=None,
        use_bn=False,
        name="local_linkage",
        **kwargs
    ):
        super(LocalLinkage, self).__init__(name=name, **kwargs)

        self.nlayers = nlayers
        self.filters = filters
        self.strides = strides
        self.kernel_size = kernel_size
        self.implementation = implementation
        self.use_bias_first = use_bias_first
        self.use_bias = use_bias
        self.use_bias_last = use_bias_last
        self.activation_first = activation_first
        self.activation = activation
        self.activation_last = activation_last
        self.bias_regularizer = bias_regularizer
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn

        self.layers = []

        for i in range(nlayers):
            if i == 0:
                bias = use_bias_first
                act = activation_first
            elif i == (nlayers - 1):
                bias = use_bias_last
                act = activation_last
            else:
                bias = use_bias
                act = activation

            if (dropout_rate is not None) and (i > 0):
                self.layers.append(layers.Dropout(
                    dropout_rate,
                    name=f"{name}/dropout:{i}"
                ))

            layer = layers.LocallyConnected1D(
                filters=filters,
                use_bias=bias,
                kernel_size=kernel_size,
                strides=strides,
                implementation=implementation,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer,
                bias_regularizer=bias_regularizer,
                activation=act,
                name=f"{name}/locally_connected_1d:{i}"
            )

            self.layers.append(layer)
            if use_bn:
                self.layers.append(layers.BatchNormalization(
                    name=f"{name}/batchnorm:{i}"
                ))

        return

    def call(self, inputs, training=False):
        for layer in self.layers:
            if isinstance(layer, (layers.Dropout, layers.BatchNormalization)):
                inputs = layer(inputs, training=training)
            else:
                inputs = layer(inputs)
        return inputs

    def get_config(self):
        config = super(LocalLinkage, self).get_config()
        config.update(dict(
            nlayers=self.nlayers,
            filters=self.filters,
            strides=self.strides,
            kernel_size=self.kernel_size,
            implementation=self.implementation,
            use_bias_first=self.use_bias_first,
            use_bias=self.use_bias,
            use_bias_last=self.use_bias_last,
            activation_first=self.activation_first,
            activation=activations.serialize(self.activation),
            activation_last=activations.serialize(self.activation_last),
            bias_regularizer=self.bias_regularizer,
            dropout_rate=self.dropout_rate,
            use_bn=self.use_bn,
        ))
        return config


class LocalLasso(tf.keras.layers.Layer):

    def __init__(
        self,
        implementation=3,
        kernel_regularizer=None,
        activity_regularizer=None,
        nonneg=False,
        activation="linear",
        use_bias=False,
        name="local_lasso",
        **kwargs
    ):
        super(LocalLasso, self).__init__(name=name, **kwargs)
        self.implementation = implementation
        self.kernel_regularizer = kernel_regularizer
        self.activity_regularizer = activity_regularizer
        self.nonneg = nonneg
        self.activation = activation
        self.use_bias = use_bias

        if kernel_regularizer is None and (activity_regularizer is not None):
            kernel_regularizer = tf.keras.regularizers.L1(1e-3)

        if nonneg:
            weight_constraint = tf.keras.constraints.NonNeg()
        else:
            weight_constraint = None

        self.lasso = tf.keras.layers.LocallyConnected1D(
            1,
            1,
            activation=activation,
            implementation=implementation,
            use_bias=use_bias,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=weight_constraint,
            name=f"{name}/lasso"
        )
        return

    def call(self, inputs):
        return self.lasso(inputs)

    def get_config(self):
        config = super(LocalLasso, self).get_config()
        config.update(dict(
            implementation=self.implementation,
            kernel_regularizer=self.kernel_regularizer,
            activity_regularizer=self.activity_regularizer,
            nonneg=self.nonneg,
            activation=activations.serialize(self.activation),
            use_bias=self.use_bias,
        ))
        return config


class ConvLinkage(layers.Layer):
    """ Progressively reduce closely co-located markers.

    Having separate activity and bias options for the last and first step
    is intended to be so that you can include a non-linearity, which would
    emulate dominance. I suspect bias would be important if you used a tanh
    activation for dominance. But including bias increases number of parameters
    a fair bit for this application.
    """

    def __init__(
        self,
        nlayers=3,
        filters=4,
        strides=2,
        kernel_size=2,
        use_bias_first=True,
        use_bias=False,
        use_bias_last=False,
        activation_first="linear",
        activation="linear",
        activation_last="linear",
        kernel_regularizer=None,
        activity_regularizer=None,
        bias_regularizer=None,
        dropout_rate=None,
        use_bn=False,
        name="conv_linkage",
        **kwargs
    ):
        super(ConvLinkage, self).__init__(name=name, **kwargs)
        self.nlayers = nlayers
        self.filters = filters
        self.strides = strides
        self.kernel_size = kernel_size
        self.use_bias_first = use_bias_first
        self.use_bias = use_bias
        self.use_bias_last = use_bias_last
        self.activation_first = activation_first
        self.activation = activation
        self.activation_last = activation_last
        self.bias_regularizer = bias_regularizer
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn

        self.layers = []

        for i in range(nlayers):
            if i == 0:
                bias = use_bias_first
                act = activation_first
            elif i == (nlayers - 1):
                bias = use_bias_last
                act = activation_last
            else:
                bias = use_bias
                act = activation

            if (dropout_rate is not None) and (i > 0):
                self.layers.append(layers.Dropout(
                    dropout_rate,
                    name=f"{name}/dropout.{i}"
                ))

            layer = layers.Conv1D(
                filters=filters ** (i + 1),
                use_bias=bias,
                kernel_size=kernel_size,
                strides=strides,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer,
                bias_regularizer=bias_regularizer,
                activation=act,
                name=f"{name}/conv1d.{i}"
            )

            self.layers.append(layer)
            if use_bn:
                self.layers.append(layers.BatchNormalization(
                    name=f"{name}/batchnormalization.{i}"
                ))
        return

    def call(self, inputs, training=False):
        for layer in self.layers:
            if isinstance(layer, (layers.Dropout, layers.BatchNormalization)):
                inputs = layer(inputs, training=training)
            else:
                inputs = layer(inputs)
        return inputs

    def get_config(self):
        config = super(ConvLinkage, self).get_config()
        config.update(dict(
            nlayers=self.nlayers,
            filters=self.filters,
            strides=self.strides,
            kernel_size=self.kernel_size,
            use_bias_first=self.use_bias_first,
            use_bias=self.use_bias,
            use_bias_last=self.use_bias_last,
            activation_first=self.activation_first,
            activation=activations.serialize(self.activation),
            activation_last=activations.serialize(self.activation_last),
            bias_regularizer=self.bias_regularizer,
            dropout_rate=self.dropout_rate,
            use_bn=self.use_bn,
        ))
        return config


class GatedUnit(layers.Layer):
    def __init__(
        self,
        units,
        use_bias=True,
        activation="linear",
        gate_kernel_regularizer=None,
        gate_activity_regularizer=None,
        gate_bias_regularizer=None,
        linear_kernel_regularizer=None,
        linear_activity_regularizer=None,
        linear_bias_regularizer=None,
        name="gated_unit",
        **kwargs
    ):
        super(GatedUnit, self).__init__(name=name, **kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.gate_kernel_regularizer = gate_kernel_regularizer
        self.gate_activity_regularizer = gate_activity_regularizer
        self.gate_bias_regularizer = gate_bias_regularizer
        self.linear_kernel_regularizer = linear_kernel_regularizer
        self.linear_activity_regularizer = linear_activity_regularizer
        self.linear_bias_regularizer = linear_bias_regularizer

        self.linear = layers.Dense(
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_regularizer=linear_kernel_regularizer,
            activity_regularizer=linear_activity_regularizer,
            bias_regularizer=linear_bias_regularizer,
            name=f"{name}/linear"
        )
        self.sigmoid = layers.Dense(
            units,
            activation="sigmoid",
            use_bias=True,
            kernel_regularizer=gate_kernel_regularizer,
            activity_regularizer=gate_activity_regularizer,
            bias_regularizer=gate_bias_regularizer,
            name=f"{name}/sigmoid"
        )

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

    def get_config(self):
        config = super(GatedUnit, self).get_config()
        config.update(dict(
            units=self.units,
            use_bias=self.use_bias,
            activation=activations.serialize(self.activation),
            gate_kernel_regularizer=self.gate_kernel_regularizer,
            gate_activity_regularizer=self.gate_activity_regularizer,
            gate_bias_regularizer=self.gate_bias_regularizer,
            linear_kernel_regularizer=self.linear_kernel_regularizer,
            linear_activity_regularizer=self.linear_activity_regularizer,
            linear_bias_regularizer=self.linear_bias_regularizer,
        ))
        return config


class GatedResidualUnit(layers.Layer):
    def __init__(
        self,
        units,
        activation="relu",
        use_bias=True,
        nonlinear_kernel_regularizer=None,
        nonlinear_activity_regularizer=None,
        nonlinear_bias_regularizer=None,
        gate_kernel_regularizer=None,
        gate_activity_regularizer=None,
        gate_bias_regularizer=None,
        project_kernel_regularizer=None,
        project_activity_regularizer=None,
        project_bias_regularizer=None,
        name="gated_residual_unit",
        **kwargs
    ):
        super(GatedResidualUnit, self).__init__(name=name, **kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self.gate_kernel_regularizer = gate_kernel_regularizer
        self.gate_activity_regularizer = gate_activity_regularizer
        self.gate_bias_regularizer = gate_bias_regularizer
        self.nonlinear_kernel_regularizer = nonlinear_kernel_regularizer
        self.nonlinear_activity_regularizer = nonlinear_activity_regularizer
        self.nonlinear_bias_regularizer = nonlinear_bias_regularizer
        self.project_kernel_regularizer = project_kernel_regularizer
        self.project_activity_regularizer = project_activity_regularizer
        self.project_bias_regularizer = project_bias_regularizer

        self.nonlinear_dense = layers.Dense(
            units,
            use_bias=use_bias,
            activation=activation,
            kernel_regularizer=nonlinear_kernel_regularizer,
            activity_regularizer=nonlinear_activity_regularizer,
            bias_regularizer=nonlinear_bias_regularizer,
            name=f"{name}/nonlinear"
        )

        self.gate = layers.Dense(
            units,
            activation="sigmoid",
            use_bias=True,
            kernel_regularizer=gate_kernel_regularizer,
            activity_regularizer=gate_activity_regularizer,
            bias_regularizer=gate_bias_regularizer,
            bias_initializer=tf.keras.initializers.Constant(-3.),
            name=f"{name}/gate"
        )

        self.layer_norm = layers.BatchNormalization(
            name=f"{name}/batchnormalization"
        )
        self.project = layers.Dense(
            units,
            use_bias=use_bias,
            activation="linear",
            kernel_regularizer=project_kernel_regularizer,
            activity_regularizer=project_activity_regularizer,
            bias_regularizer=project_bias_regularizer,
            name="f{name}/project"
        )
        return

    def call(self, inputs, training=False):
        x = self.nonlinear_dense(inputs)

        gate = self.gate(inputs)

        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)

        x = gate * x
        inputs = (1 - gate) * inputs
        x = inputs + x
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super(GatedResidualUnit, self).get_config()
        config.update(dict(
            units=self.units,
            use_bias=self.use_bias,
            activation=activations.serialize(self.activation),
            gate_kernel_regularizer=self.gate_kernel_regularizer,
            gate_activity_regularizer=self.gate_activity_regularizer,
            gate_bias_regularizer=self.gate_bias_regularizer,
            nonlinear_kernel_regularizer=self.nonlinear_kernel_regularizer,
            nonlinear_activity_regularizer=self.nonlinear_activity_regularizer,
            nonlinear_bias_regularizer=self.nonlinear_bias_regularizer,
            project_kernel_regularizer=self.project_kernel_regularizer,
            project_activity_regularizer=self.project_activity_regularizer,
            project_bias_regularizer=self.project_bias_regularizer,
        ))
        return config


class ResidualUnit(layers.Layer):

    def __init__(
        self,
        units,
        dropout_rate,
        activation="relu",
        use_bias=True,
        nonlinear_kernel_regularizer=None,
        nonlinear_activity_regularizer=None,
        nonlinear_bias_regularizer=None,
        gain_kernel_regularizer=None,
        gain_activity_regularizer=None,
        gain_bias_regularizer=None,
        name="residual_unit",
        **kwargs
    ):
        super(ResidualUnit, self).__init__()
        self.units = units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.nonlinear_kernel_regularizer = nonlinear_kernel_regularizer
        self.nonlinear_activity_regularizer = nonlinear_activity_regularizer
        self.nonlinear_bias_regularizer = nonlinear_bias_regularizer
        self.gain_kernel_regularizer = gain_kernel_regularizer
        self.gain_activity_regularizer = gain_activity_regularizer
        self.gain_bias_regularizer = gain_bias_regularizer

        self.nonlinear_dense = layers.Dense(
            units,
            use_bias=use_bias,
            activation=activation,
            kernel_regularizer=nonlinear_kernel_regularizer,
            activity_regularizer=nonlinear_activity_regularizer,
            bias_regularizer=nonlinear_bias_regularizer,
            name=f"{name}/nonlinear",
        )
        self.dropout = layers.Dropout(
            dropout_rate,
            name=f"{name}/dropout"
        )
        self.gain_dense = layers.Dense(
            units,
            use_bias=use_bias,
            activation="linear",
            kernel_regularizer=gain_kernel_regularizer,
            activity_regularizer=gain_activity_regularizer,
            bias_regularizer=gain_bias_regularizer,
            name=f"{name}/gain",
        )

        self.layer_norm = layers.BatchNormalization(
            name=f"{name}/batchnormalization",
        )
        return

    def call(self, inputs, training=False):
        assert inputs.shape[-1] == self.units, (
            "The input shape must be the same as the output shape. "
            "Consider using the ParallelResidualUnit."
        )

        nl = self.nonlinear_dense(inputs)
        nl = self.dropout(nl, training=training)
        nl = self.gain_dense(nl)

        x = nl + inputs
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super(ResidualUnit, self).get_config()
        config.update(dict(
            units=self.units,
            dropout_rate=self.dropout_rate,
            activation=activations.serialize(self.activation),
            use_bias=self.use_bias,
            nonlinear_kernel_regularizer=self.nonlinear_kernel_regularizer,
            nonlinear_activity_regularizer=self.nonlinear_activity_regularizer,
            nonlinear_bias_regularizer=self.nonlinear_bias_regularizer,
            gain_kernel_regularizer=self.gain_kernel_regularizer,
            gain_activity_regularizer=self.gain_activity_regularizer,
            gain_bias_regularizer=self.gain_bias_regularizer,
        ))


class ParallelResidualUnit(layers.Layer):
    def __init__(
        self,
        units,
        dropout_rate,
        activation="relu",
        use_bias=True,
        nonlinear_kernel_regularizer=None,
        nonlinear_activity_regularizer=None,
        nonlinear_bias_regularizer=None,
        gain_kernel_regularizer=None,
        gain_activity_regularizer=None,
        gain_bias_regularizer=None,
        linear_kernel_regularizer=None,
        linear_activity_regularizer=None,
        linear_bias_regularizer=None,
        name="parallel_residual_unit",
        **kwargs
    ):
        super(ParallelResidualUnit, self).__init__(name=name, **kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.nonlinear_kernel_regularizer = nonlinear_kernel_regularizer
        self.nonlinear_activity_regularizer = nonlinear_activity_regularizer
        self.nonlinear_bias_regularizer = nonlinear_bias_regularizer
        self.linear_kernel_regularizer = linear_kernel_regularizer
        self.linear_activity_regularizer = linear_activity_regularizer
        self.linear_bias_regularizer = linear_bias_regularizer
        self.gain_kernel_regularizer = gain_kernel_regularizer
        self.gain_activity_regularizer = gain_activity_regularizer
        self.gain_bias_regularizer = gain_bias_regularizer

        self.nonlinear_dense = layers.Dense(
            units,
            use_bias=use_bias,
            activation=activation,
            kernel_regularizer=nonlinear_kernel_regularizer,
            activity_regularizer=nonlinear_activity_regularizer,
            bias_regularizer=nonlinear_bias_regularizer,
            name=f"{name}/nonlinear",
        )
        self.dropout = layers.Dropout(
            dropout_rate,
            name=f"{name}/dropout"
        )
        self.gain_dense = layers.Dense(
            units,
            use_bias=use_bias,
            activation="linear",
            kernel_regularizer=gain_kernel_regularizer,
            activity_regularizer=gain_activity_regularizer,
            bias_regularizer=gain_bias_regularizer,
            name=f"{name}/gain",
        )

        self.layer_norm = layers.BatchNormalization(
            name=f"{name}/batchnormalization"
        )
        self.linear_dense = layers.Dense(
            units,
            use_bias=use_bias,
            activation="linear",
            kernel_regularizer=linear_kernel_regularizer,
            activity_regularizer=linear_activity_regularizer,
            bias_regularizer=linear_bias_regularizer,
            name=f"{name}/linear"
        )
        return

    def call(self, inputs, training=False):
        nl = self.nonlinear_dense(inputs)
        nl = self.dropout(nl, training=training)
        nl = self.gain_dense(nl)

        li = self.linear_dense(inputs)
        x = nl + li
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super(ParallelResidualUnit, self).get_config()
        config.update(dict(
            units=self.units,
            dropout_rate=self.dropout_rate,
            activation=activations.serialize(self.activation),
            use_bias=self.use_bias,
            nonlinear_kernel_regularizer=self.nonlinear_kernel_regularizer,
            nonlinear_activity_regularizer=self.nonlinear_activity_regularizer,
            nonlinear_bias_regularizer=self.nonlinear_bias_regularizer,
            linear_kernel_regularizer=self.linear_kernel_regularizer,
            linear_activity_regularizer=self.linear_activity_regularizer,
            linear_bias_regularizer=self.linear_bias_regularizer,
            gain_kernel_regularizer=self.gain_kernel_regularizer,
            gain_activity_regularizer=self.gain_activity_regularizer,
            gain_bias_regularizer=self.gain_bias_regularizer,
        ))


class ParallelUnit(layers.Layer):
    def __init__(
        self,
        units,
        dropout_rate,
        activation="relu",
        use_bias=True,
        nonlinear_kernel_regularizer=None,
        nonlinear_activity_regularizer=None,
        nonlinear_bias_regularizer=None,
        gain_kernel_regularizer=None,
        gain_activity_regularizer=None,
        gain_bias_regularizer=None,
        linear_kernel_regularizer=None,
        linear_activity_regularizer=None,
        linear_bias_regularizer=None,
        name="parallel_unit",
        **kwargs
    ):
        super(ParallelUnit, self).__init__(name=name, **kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_bias = use_bias
        self.nonlinear_kernel_regularizer = nonlinear_kernel_regularizer
        self.nonlinear_activity_regularizer = nonlinear_activity_regularizer
        self.nonlinear_bias_regularizer = nonlinear_bias_regularizer
        self.linear_kernel_regularizer = linear_kernel_regularizer
        self.linear_activity_regularizer = linear_activity_regularizer
        self.linear_bias_regularizer = linear_bias_regularizer
        self.gain_kernel_regularizer = gain_kernel_regularizer
        self.gain_activity_regularizer = gain_activity_regularizer
        self.gain_bias_regularizer = gain_bias_regularizer

        self.nonlinear_dense = layers.Dense(
            units,
            use_bias=use_bias,
            activation=activation,
            kernel_regularizer=nonlinear_kernel_regularizer,
            activity_regularizer=nonlinear_activity_regularizer,
            bias_regularizer=nonlinear_bias_regularizer,
            name=f"{name}/nonlinear"
        )
        self.dropout = layers.Dropout(
            dropout_rate,
            name=f"{name}/dropout"
        )
        self.gain_dense = layers.Dense(
            units,
            use_bias=use_bias,
            activation="linear",
            kernel_regularizer=gain_kernel_regularizer,
            activity_regularizer=gain_activity_regularizer,
            bias_regularizer=gain_bias_regularizer,
            name=f"{name}/gain"
        )

        self.layer_norm = layers.BatchNormalization(
            name=f"{name}/batchnormalization"
        )
        self.linear_dense = layers.Dense(
            units,
            use_bias=use_bias,
            activation="linear",
            kernel_regularizer=linear_kernel_regularizer,
            activity_regularizer=linear_activity_regularizer,
            bias_regularizer=linear_bias_regularizer,
            name=f"{name}/linear"
        )
        self.concat = layers.Concatenate(
            name=f"{name}/concatenate"
        )
        return

    def call(self, inputs, training=False):
        nl = self.nonlinear_dense(inputs)
        nl = self.dropout(nl, training=training)
        nl = self.gain_dense(nl)

        li = self.linear_dense(inputs)
        x = self.concat([nl, li])
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super(ParallelUnit, self).get_config()
        config.update(dict(
            units=self.units,
            dropout_rate=self.dropout_rate,
            activation=activations.serialize(self.activation),
            use_bias=self.use_bias,
            nonlinear_kernel_regularizer=self.nonlinear_kernel_regularizer,
            nonlinear_activity_regularizer=self.nonlinear_activity_regularizer,
            nonlinear_bias_regularizer=self.nonlinear_bias_regularizer,
            linear_kernel_regularizer=self.linear_kernel_regularizer,
            linear_activity_regularizer=self.linear_activity_regularizer,
            linear_bias_regularizer=self.linear_bias_regularizer,
            gain_kernel_regularizer=self.gain_kernel_regularizer,
            gain_activity_regularizer=self.gain_activity_regularizer,
            gain_bias_regularizer=self.gain_bias_regularizer,
        ))
