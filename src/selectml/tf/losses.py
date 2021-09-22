import tensorflow as tf


@tf.function
@tf.autograph.experimental.do_not_convert
def filter_tfsparse(sp, select):
    select = tf.convert_to_tensor(
        select,
        dtype=tf.dtypes.int64,
        name="select_convert"
    )
    mask0 = tf.reduce_any(
        tf.math.equal(
            tf.reshape(sp.indices[:, 0], [-1, 1], name="mask0_reshape"),
            select
        ),
        -1,
        name="mask0"
    )

    mask1 = tf.reduce_any(
        tf.math.equal(
            tf.reshape(sp.indices[:, 1], [-1, 1], name="mask1_reshape"),
            select
        ),
        -1,
        name="mask1"
    )

    mask = tf.reduce_all(
        tf.concat(
            [
                tf.reshape(mask0, (-1, 1)),
                tf.reshape(mask1, (-1, 1))
            ],
            -1
        ),
        -1
    )

    select_positions = tf.reduce_sum(
        tf.cast(
            tf.math.equal(
                tf.reshape(
                    tf.range(0, tf.size(sp), dtype=tf.dtypes.int64),
                    (-1, 1)
                ),
                select
            ),
            tf.dtypes.int64
        ),
        axis=-1
    )
    select_positions = tf.cumsum(select_positions) - 1

    values = sp.values[mask]
    n = tf.shape(select)[0]

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def mapper(i):
        return select_positions[i]

    new_indices0 = tf.map_fn(mapper, sp.indices[:, 0])
    new_indices1 = tf.map_fn(mapper, sp.indices[:, 1])
    new_indices = tf.concat(
        [
            tf.reshape(new_indices0[mask], (-1, 1)),
            tf.reshape(new_indices1[mask], (-1, 1))
        ],
        -1
    )
    sp1 = tf.SparseTensor(new_indices, values, (n, n))

    return tf.sparse.to_dense(sp1)


@tf.function
def response_diffs(y, sd):
    y = tf.reshape(
        tf.convert_to_tensor(y),
        (-1, 1)
    )

    diffs = tf.math.subtract(y, tf.transpose(y))
    diffs = tf.abs(diffs) > sd
    return tf.sparse.from_dense(diffs)


@tf.function
def dist_to_sparse(dist, threshold):
    dist = tf.convert_to_tensor(dist)
    dist = dist > threshold
    return tf.sparse.from_dense(dist)


@tf.function
def _masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
      data: 2-D float `Tensor` of shape `[n, m]`.
      mask: 2-D Boolean `Tensor` of shape `[n, m]`.
      dim: The dimension over which to compute the maximum.
    Returns:
      masked_maximums: N-D `Tensor`.
        The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = tf.math.reduce_min(data, dim, keepdims=True)
    masked_maximums = (
        tf.math.reduce_max(
            tf.math.multiply(data - axis_minimums, mask), dim, keepdims=True
        )
        + axis_minimums
    )
    return masked_maximums


@tf.function
def _masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
      data: 2-D float `Tensor` of shape `[n, m]`.
      mask: 2-D Boolean `Tensor` of shape `[n, m]`.
      dim: The dimension over which to compute the minimum.
    Returns:
      masked_minimums: N-D `Tensor`.
        The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = tf.math.reduce_max(data, dim, keepdims=True)
    masked_minimums = (
        tf.math.reduce_min(
            tf.math.multiply(data - axis_maximums, mask), dim, keepdims=True
        )
        + axis_maximums
    )
    return masked_minimums


class RankLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        reduction=tf.keras.losses.Reduction.AUTO,
        name=None,
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        return

    def call(self, y_true, y_pred):

        y_true = tf.cast(
            tf.convert_to_tensor(y_true, name="y_true"),
            dtype=tf.dtypes.float32
        )

        if len(y_true.shape) == 1:
            y_true = tf.reshape(y_true, (1, -1))

        y_pred = tf.cast(
            tf.convert_to_tensor(y_pred, name="y_pred"),
            dtype=tf.dtypes.float32
        )
        if len(y_pred.shape) == 1:
            y_pred = tf.reshape(y_pred, (1, -1))

        adjacency = tf.math.greater(
            tf.subtract(y_true, tf.transpose(y_true)),
            0.
        )
        adjacency = tf.cast(adjacency, tf.dtypes.float32)

        pred_diffs = tf.subtract(y_pred, tf.transpose(y_pred))
        pred_diffs = tf.math.multiply(tf.nn.sigmoid(pred_diffs), adjacency)
        pred_diffs = tf.reshape(pred_diffs, (-1, 1))
        adjacency = tf.reshape(adjacency, (-1, 1))

        loss = tf.losses.binary_crossentropy(adjacency, pred_diffs)
        return loss


class MultiSURFTripletLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        sd,
        margin=1.0,
        distance_metric="L2",
        reduction=tf.keras.losses.Reduction.AUTO,
        name=None,
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.sd = sd
        self.margin = margin
        self.distance_metric = distance_metric
        return

    @tf.function
    def response_diffs(self, y):
        diffs = tf.math.subtract(y, tf.transpose(y))
        diffs = tf.abs(diffs) < self.sd
        return diffs  # tf.cast(diffs, tf.dtypes.float32)

    def call(self, y_true, y_pred):
        from tensorflow_addons.losses import metric_learning

        labels = tf.cast(
            tf.convert_to_tensor(y_true, name="labels"),
            dtype=tf.dtypes.float32
        )
        if len(labels.shape) == 1:
            labels = tf.reshape(labels, (1, -1))

        batch_size = tf.shape(labels)[0]

        embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

        convert_to_float32 = (
            (embeddings.dtype == tf.dtypes.float16) or
            (embeddings.dtype == tf.dtypes.bfloat16)
        )
        precise_embeddings = (
            tf.cast(embeddings, tf.dtypes.float32)
            if convert_to_float32
            else embeddings
        )

        # Reshape label tensor to [batch_size, 1].
        # lshape = tf.shape(labels)
        # labels = tf.reshape(labels, [lshape[0], 1])

        # Build pairwise squared distance matrix
        distance_metric = self.distance_metric

        if distance_metric == "L2":
            pdist_matrix = metric_learning.pairwise_distance(
                precise_embeddings, squared=False
            )

        elif distance_metric == "squared-L2":
            pdist_matrix = metric_learning.pairwise_distance(
                precise_embeddings, squared=True
            )

        elif distance_metric == "angular":
            pdist_matrix = metric_learning.angular_distance(precise_embeddings)

        else:
            pdist_matrix = distance_metric(precise_embeddings)

        # Fetch pairwise labels as adjacency matrix.
        adjacency = self.response_diffs(labels)
        # Invert so we can select negatives only.
        adjacency_not = tf.math.logical_not(adjacency)

        radii = (
            tf.reduce_mean(pdist_matrix, axis=1) -
            (tf.math.reduce_std(pdist_matrix, axis=1) / 2.)
        )
        neighbors = tf.math.less(pdist_matrix, tf.reshape(radii, (-1, 1)))

        hits = (
            tf.cast(
                tf.math.logical_and(neighbors, adjacency),
                tf.dtypes.float32
            ) - tf.linalg.diag(tf.ones([batch_size]))
        )

        misses = tf.cast(
            tf.math.logical_and(neighbors, adjacency_not),
            tf.dtypes.float32
        )

        nhits = tf.reduce_sum(hits)
        nmisses = tf.reduce_sum(misses)

        n = tf.cast(batch_size, tf.dtypes.float32)
        hits_dists = tf.multiply(pdist_matrix, hits)
        hits_dists = tf.math.divide_no_nan(
            hits_dists,
            tf.math.multiply(n, nhits)
        )
        misses_dists = tf.multiply(pdist_matrix, misses)
        misses_dists = tf.math.divide_no_nan(
            misses_dists,
            tf.math.multiply(n, nmisses)
        )

        loss = tf.subtract(misses_dists, hits_dists)
        loss = tf.reduce_sum(loss, axis=1)

        if convert_to_float32:
            return tf.cast(loss, embeddings.dtype)
        else:
            return loss


class SemiHardTripletLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        sd,
        margin=1.0,
        distance_metric="L2",
        reduction=tf.keras.losses.Reduction.AUTO,
        name=None,
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.sd = sd
        self.margin = margin
        self.distance_metric = distance_metric
        return

    @tf.function
    def response_diffs(self, y):
        diffs = tf.math.subtract(y, tf.transpose(y))
        diffs = tf.abs(diffs) < self.sd
        return diffs  # tf.cast(diffs, tf.dtypes.float32)

    def call(self, y_true, y_pred):
        from tensorflow_addons.losses import metric_learning

        labels = tf.cast(
            tf.convert_to_tensor(y_true, name="labels"),
            dtype=tf.dtypes.float32
        )
        if len(labels.shape) == 1:
            labels = tf.reshape(labels, (1, -1))

        embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

        convert_to_float32 = (
            (embeddings.dtype == tf.dtypes.float16) or
            (embeddings.dtype == tf.dtypes.bfloat16)
        )
        precise_embeddings = (
            tf.cast(embeddings, tf.dtypes.float32)
            if convert_to_float32
            else embeddings
        )

        # Reshape label tensor to [batch_size, 1].
        # lshape = tf.shape(labels)
        # labels = tf.reshape(labels, [lshape[0], 1])

        # Build pairwise squared distance matrix
        distance_metric = self.distance_metric

        if distance_metric == "L2":
            pdist_matrix = metric_learning.pairwise_distance(
                precise_embeddings, squared=False
            )

        elif distance_metric == "squared-L2":
            pdist_matrix = metric_learning.pairwise_distance(
                precise_embeddings, squared=True
            )

        elif distance_metric == "angular":
            pdist_matrix = metric_learning.angular_distance(precise_embeddings)

        else:
            pdist_matrix = distance_metric(precise_embeddings)

        # Fetch pairwise labels as adjacency matrix.
        adjacency = self.response_diffs(labels)
        # Invert so we can select negatives only.
        adjacency_not = tf.math.logical_not(adjacency)

        batch_size = tf.size(labels)

        # Compute the mask.
        pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
        mask = tf.math.logical_and(
            tf.tile(adjacency_not, [batch_size, 1]),
            tf.math.greater(
                pdist_matrix_tile,
                tf.reshape(tf.transpose(pdist_matrix), [-1, 1])
            ),
        )
        mask_final = tf.reshape(
            tf.math.greater(
                tf.math.reduce_sum(
                    tf.cast(mask, dtype=tf.dtypes.float32),
                    1,
                    keepdims=True
                ),
                0.0,
            ),
            [batch_size, batch_size],
        )
        mask_final = tf.transpose(mask_final)

        adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
        mask = tf.cast(mask, dtype=tf.dtypes.float32)

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = tf.reshape(
            _masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]
        )
        negatives_outside = tf.transpose(negatives_outside)

        # negatives_inside: largest D_an.
        negatives_inside = tf.tile(
            _masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]
        )
        semi_hard_negatives = tf.where(
            mask_final,
            negatives_outside,
            negatives_inside
        )

        loss_mat = tf.math.add(self.margin, pdist_matrix - semi_hard_negatives)

        mask_positives = (
            tf.cast(adjacency, dtype=tf.dtypes.float32) -
            tf.linalg.diag(tf.ones([batch_size]))
        )

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        # Max(n, 1) necessary to stop nan loss, which just stops the whole
        # model from running.
        # Setting to 1 will just mean zero loss, since everything
        # else will be 0.
        num_positives = tf.math.maximum(
            tf.math.reduce_sum(mask_positives),
            1.0
        )

        triplet_loss = tf.math.truediv(
            tf.math.reduce_sum(
                tf.math.maximum(
                    tf.math.multiply(loss_mat, mask_positives),
                    0.0
                )
            ),
            num_positives,
        )

        if convert_to_float32:
            return tf.cast(triplet_loss, embeddings.dtype)
        else:
            return triplet_loss


class HardTripletLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        sd,
        soft=False,
        margin=1.0,
        distance_metric="L2",
        reduction=tf.keras.losses.Reduction.AUTO,
        name=None,
        **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.sd = sd
        self.margin = margin
        self.distance_metric = distance_metric
        self.soft = soft
        return

    @tf.function
    def response_diffs(self, y):
        diffs = tf.math.subtract(y, tf.transpose(y))
        diffs = tf.abs(diffs) < self.sd
        return diffs  # tf.cast(diffs, tf.dtypes.float32)

    def call(self, y_true, y_pred):
        from tensorflow_addons.losses import metric_learning

        labels = tf.cast(
            tf.convert_to_tensor(y_true, name="labels"),
            dtype=tf.dtypes.float32
        )
        if len(labels.shape) == 1:
            labels = tf.reshape(labels, (1, -1))

        embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

        convert_to_float32 = (
            (embeddings.dtype == tf.dtypes.float16) or
            (embeddings.dtype == tf.dtypes.bfloat16)
        )
        precise_embeddings = (
            tf.cast(embeddings, tf.dtypes.float32)
            if convert_to_float32
            else embeddings
        )

        # Reshape label tensor to [batch_size, 1].
        # lshape = tf.shape(labels)
        # labels = tf.reshape(labels, [lshape[0], 1])

        # Build pairwise squared distance matrix
        distance_metric = self.distance_metric

        if distance_metric == "L2":
            pdist_matrix = metric_learning.pairwise_distance(
                precise_embeddings, squared=False
            )

        elif distance_metric == "squared-L2":
            pdist_matrix = metric_learning.pairwise_distance(
                precise_embeddings, squared=True
            )

        elif distance_metric == "angular":
            pdist_matrix = metric_learning.angular_distance(precise_embeddings)

        else:
            pdist_matrix = distance_metric(precise_embeddings)

        # Fetch pairwise labels as adjacency matrix.
        adjacency = self.response_diffs(labels)

        # Invert so we can select negatives only.
        adjacency_not = tf.math.logical_not(adjacency)

        adjacency = tf.cast(adjacency, dtype=tf.dtypes.float32)
        adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
        hard_negatives = _masked_minimum(pdist_matrix, adjacency_not)

        batch_size = tf.size(labels)

        mask_positives = (
            tf.cast(adjacency, dtype=tf.dtypes.float32) -
            tf.linalg.diag(tf.ones([batch_size]))
        )

        # hard positives: largest D_ap.
        hard_positives = _masked_maximum(pdist_matrix, mask_positives)

        if self.soft:
            triplet_loss = tf.math.log1p(
                tf.math.exp(hard_positives - hard_negatives))
        else:
            triplet_loss = tf.maximum(
                hard_positives - hard_negatives + self.margin,
                0.0
            )

        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        if convert_to_float32:
            return tf.cast(triplet_loss, embeddings.dtype)
        else:
            return triplet_loss
