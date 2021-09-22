import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package='Custom',
    name='AdaptiveL1Regularizer'
)
class AdaptiveL1Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, weights=None, adapt=False, l1=0.):
        self.l1 = l1
        self.weights = weights
        self.adapt = adapt
        return

    def __call__(self, x):
        if self.weights is None:
            self.weights = tf.Variable(tf.ones(tf.shape(x)), trainable=False)

        result = (
            self.l1 *
            tf.math.reduce_sum(tf.math.multiply(tf.math.abs(x), self.weights))
        )

        if self.adapt:
            self.weights.assign(tf.math.divide_no_nan(1., tf.math.abs(x)))
        return result

    def get_config(self):
        return {
            'l1': float(self.l1),
            "weights": [float(f) for f in self.weights]
        }


@tf.keras.utils.register_keras_serializable(
    package='Custom',
    name='AdaptiveL1L2Regularizer'
)
class AdaptiveL1L2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, weights=None, adapt=False, l1=0.0, l2=0.0):
        self.l1 = l1
        self.l2 = l2
        self.weights = weights
        self.adapt = adapt
        return

    def __call__(self, x):
        if self.weights is None:
            self.weights = tf.Variable(tf.ones(tf.shape(x)), trainable=False)

        l1 = (
            self.l1 *
            tf.math.reduce_sum(tf.math.multiply(tf.math.abs(x), self.weights))
        )
        l2 = self.l2 * tf.math.reduce_sum(x ** 2)

        result = l1 + l2

        if self.adapt:
            self.weights.assign(tf.math.divide_no_nan(1., tf.math.abs(x)))

        return result

    def get_config(self):
        return {
            'l1': float(self.l1),
            'l2': float(self.l2),
            "weights": [float(f) for f in self.weights]
        }
