import tensorflow as tf

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional

from .losses import (
    SemiHardTripletLoss,
    HardTripletLoss,
    MultiSURFTripletLoss,
    RankLoss
)


class SSModel(tf.keras.Model):

    def __init__(
        self,
        marker_embedder,
        env_embedder=None,
        combiner: str = "add",
        post_embedder=None,
        env_hard_loss: "Optional[float]" = None,
        env_semihard_loss: "Optional[float]" = None,
        marker_hard_loss: "Optional[float]" = None,
        marker_semihard_loss: "Optional[float]" = None,
        marker_relief_loss: "Optional[float]" = None,
        rank_loss: "Optional[float]" = None,
        sd: float = 1,
    ):
        super(SSModel, self).__init__()

        assert combiner in ("add", "concat")

        self.marker_embedder = marker_embedder
        self.env_embedder = env_embedder
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

        if self.env_embedder is None:
            self.env_semihard_triplet_loss = None
            self.env_semihard_triplet_loss_tracker = None

            self.env_hard_triplet_loss = None
            self.env_hard_triplet_loss_tracker = None
        else:
            self.env_semihard_triplet_loss = SemiHardTripletLoss(
                sd=sd,
                name="env_semihard_triplet"
            )
            self.env_semihard_triplet_loss_tracker = tf.metrics.Mean(
                name="env_semihard_triplet_loss"
            )

            self.env_hard_triplet_loss = HardTripletLoss(
                sd=sd,
                name="env_hard_triplet"
            )
            self.env_hard_triplet_loss_tracker = tf.metrics.Mean(
                name="env_hard_triplet_loss"
            )

        if rank_loss is not None:
            self.rank_out = tf.keras.layers.Dense(1)
            self.rank_loss = RankLoss(name="rank")
            self.rank_loss_tracker = tf.metrics.Mean(name="rank_loss")
        else:
            self.rank_out = None
            self.rank_loss = None
            self.rank_loss_tracker = None

        self.combiner = combiner
        if combiner == "add":
            self.combine_layer = tf.keras.layers.Add()
        if combiner == "concat":
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

        if self.rank_out is not None:
            rank_out = self.rank_out(embed)
        else:
            rank_out = None

        return marker_embed, env_embed, out, rank_out
