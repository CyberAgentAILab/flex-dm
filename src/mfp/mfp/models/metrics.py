from typing import Dict, Union

import tensorflow as tf
from mfp.data.spec import get_valid_input_columns
from mfp.models.architecture.mask import get_seq_mask

from .tensor_utils import sort_inputs

BIG_CONST = 1000.0


def mae_from_logits(y_true: tf.Tensor, y_pred: tf.Tensor, from_logits: bool = True):
    # tf.debugging.assert_rank(y_true, 2)
    # tf.debugging.assert_rank(y_pred, 3)
    tf.debugging.assert_equal(tf.rank(y_true) + 1, tf.rank(y_pred))

    C = tf.shape(y_pred)[-1]
    div = tf.cast(C - 1, tf.float32)
    target = tf.cast(y_true, tf.float32)
    target = target / div
    output = tf.nn.softmax(y_pred) if from_logits else y_pred
    values = tf.cast(tf.range(C), tf.float32) / div
    if tf.rank(y_true) == 2:
        output *= values[tf.newaxis, tf.newaxis, :]
    elif tf.rank(y_true) == 3:
        output *= values[tf.newaxis, tf.newaxis, tf.newaxis, :]
    else:
        raise NotImplementedError

    output = tf.reduce_sum(output, axis=-1)
    # loss = tf.keras.metrics.mean_absolute_error(target, output)
    loss = tf.math.abs(target - output)
    return loss


def compute_categorical_mfp_metric(
    y_true: tf.Tensor, y_pred: tf.Tensor, from_logits: bool = True
):
    # shape of y_true and y_pred is (..., C, X)
    # shape of loss and score is both (..., C)
    if from_logits:
        y_pred_ = tf.nn.softmax(y_pred)
    else:
        y_pred_ = y_pred

    y_pred_argmax = tf.argmax(y_pred_, axis=-1, output_type=tf.int32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred_)
    score = tf.cast(y_true == y_pred_argmax, tf.float32)
    return loss, score


def compute_continuous_mfp_metric(y_true: tf.Tensor, y_pred: tf.Tensor):
    # shape of y_true and y_pred is (..., C, X)
    # shape of loss and score is both (..., )
    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    score = -0.5 * tf.keras.losses.cosine_similarity(y_true, y_pred) + 0.5
    return loss, score


class BeautyLayer(tf.keras.layers.Layer):
    """
    For definition of each metric, please refer to
    Attribute-conditioned Layout GAN for Automatic Graphic Design
    https://arxiv.org/abs/2009.05284
    """

    def __init__(self, input_columns: Dict, name: str = "beauty_layer", **kwargs):
        super().__init__(name=name, **kwargs)
        assert "left" in input_columns and "width" in input_columns
        self.input_columns = input_columns

    def call(self, inputs, training=False, from_logits: bool = True):
        if from_logits:
            y_pred, masks = inputs
        else:
            y_true, masks = inputs
        mask = masks["left"]  # (B, S)
        B, S = tf.shape(mask)

        mask_float = tf.cast(mask, tf.float32)
        count = tf.reduce_sum(mask_float, axis=-1)
        num_invalid_documents = tf.reduce_sum(tf.cast(count <= 1, tf.float32))
        num_valid_documents = tf.cast(B, tf.float32) - num_invalid_documents

        data = {}
        for key in ["left", "width", "top", "height"]:
            column = self.input_columns[key]
            C = tf.cast(column["input_dim"], tf.float32)
            if from_logits:
                coords = tf.math.argmax(y_pred[key], axis=-1)[..., 0]  # (B, S)
            else:
                coords = y_true[key][..., 0]
            data[key] = tf.cast(coords, tf.float32) / (C - 1)  # (B, S)

        eye = tf.eye(S, batch_shape=[B], dtype=tf.bool)
        valid = tf.math.logical_and(mask[:, tf.newaxis, :], mask[..., tf.newaxis])
        invalid = tf.math.logical_or(eye, tf.logical_not(valid))

        keys = [("left", "width"), ("top", "height")]
        diff = []
        for (start_key, interval_key) in keys:
            for i in range(3):
                s = i / 2
                h = data[start_key] + data[interval_key] * s  # (B, S)
                h = h[:, :, tf.newaxis] - h[:, tf.newaxis, :]  # (B, S, S)
                # Eq. 11
                h = tf.math.abs(h)
                h = tf.where(invalid, tf.ones_like(h), h)
                h = tf.reduce_min(h, axis=-1)  # (B, S)
                h = -1.0 * tf.math.log(1.0 - h)
                diff.append(h)

        # Eq. 10
        diff = tf.stack(diff, axis=-1)  # (B, S, 6)
        diff = tf.reduce_min(diff, axis=-1)  # (B, S)
        diff = tf.where(tf.math.is_finite(diff), diff, tf.zeros_like(diff))
        alignment = tf.reduce_sum(diff, axis=-1) / count  # (B, )
        alignment = tf.where(count > 1, alignment, tf.zeros_like(alignment))

        # Overlap
        right = data["left"] + data["width"]
        bottom = data["top"] + data["height"]
        l1, t1 = data["left"][..., tf.newaxis], data["top"][..., tf.newaxis]
        r1, b1 = right[..., tf.newaxis], bottom[..., tf.newaxis]
        l2, t2 = data["left"][:, tf.newaxis, :], data["top"][:, tf.newaxis, :]
        r2, b2 = right[:, tf.newaxis, :], bottom[:, tf.newaxis, :]

        a1 = (r1 - l1) * (b1 - t1)
        l_max, t_max = tf.math.maximum(l1, l2), tf.math.maximum(t1, t2)
        r_min, b_min = tf.math.minimum(r1, r2), tf.math.minimum(b1, b2)
        cond = (l_max < r_min) & (t_max < b_min)
        ai = (r_min - l_max) * (b_min - t_max)
        ai = tf.where((cond & tf.logical_not(eye)), ai, tf.zeros_like(ai))
        ai = tf.where(a1 > 0.0, ai / a1, tf.zeros_like(ai))
        overlap = tf.reduce_sum(ai, axis=[-2, -1]) / count
        overlap = tf.where(count > 1, overlap, tf.zeros_like(overlap))

        # lb = data["type"]
        # label_match = (lb[..., tf.newaxis] == lb[:, tf.newaxis, :])

        # au = a1 + a2 - ai
        # iou = tf.where(au > 0.0, ai / au, tf.zeros_like(au))
        # cost = tf.fill(tf.shape(ai), 10000.0)
        # cost = tf.where(label_match & valid, 1.0 - iou, cost)
        # # cost = 1.0 - iou  # (0.0 is best, 1.0 is worst)
        # for i in range(B):
        #     score = 0.0
        #     # for (j, k) in linear_sum_assignment(cost[i]):
        #     #     score +=

        scores = {
            "alignment_num": tf.reduce_sum(alignment),
            "alignment_den": num_valid_documents,
            "overlap_num": tf.reduce_sum(overlap),
            "overlap_den": num_valid_documents,
        }
        return scores


class LossLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_columns: Dict,
        name: str = "loss_layer",
        predict_context: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._input_columns = input_columns
        self._valid_input_columns = get_valid_input_columns(input_columns)
        self._predict_context = predict_context

    def call(
        self,
        inputs,
        training=False,
        sort_flag: Union[bool, tf.Tensor] = None,
        ignore_sort: str = None,
    ):
        if tf.is_tensor(sort_flag):
            assert ignore_sort in ["gt", "pred", None]
            y_true_, y_pred_, mfp_masks = inputs
            if ignore_sort == "gt":
                y_true_sort = y_true_
            else:
                y_true_sort = sort_inputs(y_true_, self._valid_input_columns)

            y_pred_["length"] = y_true_["length"]
            if ignore_sort == "pred":
                y_pred_sort = y_pred_
            else:
                y_pred_sort = sort_inputs(
                    y_pred_, self._valid_input_columns, from_logits=True
                )

            y_true, y_pred = {}, {}
            for key in y_true_.keys():
                column = self._input_columns[key]
                if column.get("demo_only", False):
                    continue
                if column["is_sequence"]:
                    flag = sort_flag[:, tf.newaxis, tf.newaxis]
                    y_true[key] = tf.where(flag, y_true_sort[key], y_true_[key])
                    if column["type"] == "categorical":
                        flag = flag[:, tf.newaxis]
                    y_pred[key] = tf.where(flag, y_pred_sort[key], y_pred_[key])
                else:
                    if key in y_true_:
                        y_true[key] = y_true_[key]
                    if key in y_pred_:
                        y_pred[key] = y_pred_[key]
        else:
            y_true, y_pred, mfp_masks = inputs

        seq_mask = get_seq_mask(y_true["length"])

        loss_total = 0
        score_total = 0
        losses = {}
        scores = {}

        for key, column in self._input_columns.items():
            if column.get("demo_only", False):
                continue

            if not column["is_sequence"] and not self._predict_context:
                continue

            prediction = y_pred[key]
            # Cut extra elements in prediction.
            prediction = prediction[:, : tf.shape(seq_mask)[1]]

            if column["type"] == "categorical":
                # check if the labels are in intended range of values
                C = tf.cast(column["input_dim"], tf.int32)
                tf.debugging.assert_less_equal(tf.reduce_max(y_true[key]), C - 1)
                tf.debugging.assert_greater_equal(tf.reduce_min(y_true[key]), 0)

                y_true[key] = tf.cast(y_true[key], tf.int32)
                loss, score = compute_categorical_mfp_metric(
                    y_true[key], prediction, from_logits=True
                )
                # if key == 'font_size':
                #     score = mae_from_logits(y_true[key], prediction, from_logits=True)
            else:
                loss, score = compute_continuous_mfp_metric(y_true[key], prediction)
                loss = tf.expand_dims(loss, -1)
                loss = loss * tf.cast(column["shape"][-1], tf.float32)
                score = tf.expand_dims(score, -1)

            mfp_weight = tf.cast(mfp_masks[key][..., tf.newaxis], tf.float32)
            loss *= mfp_weight
            score *= mfp_weight
            den = tf.cast(tf.ones(tf.shape(loss)), tf.float32) * mfp_weight

            if "loss_condition" in column:
                cond = column["loss_condition"]
                weight = tf.gather(cond["mask"], y_true[cond["key"]])
                loss *= tf.cast(weight, tf.float32)
                score *= tf.cast(weight, tf.float32)
                den *= tf.cast(weight, tf.float32)

            if column["is_sequence"]:
                weight = tf.cast(seq_mask[:, :, tf.newaxis], tf.float32)
                loss = tf.reduce_sum(loss * weight, axis=1)  # sum timesteps
                score = tf.reduce_sum(score * weight, axis=1)
                den = tf.reduce_sum(den * weight, axis=1)

            loss = tf.reduce_sum(loss, axis=1)  # sum features
            score = tf.reduce_sum(score, axis=1)
            den = tf.reduce_sum(den, axis=1)

            tf.debugging.assert_rank(loss, 1)
            tf.debugging.assert_rank(score, 1)
            tf.debugging.assert_rank(den, 1)

            loss = tf.reduce_mean(loss)  # average batch

            score = tf.reduce_sum(score)
            den = tf.reduce_sum(den)
            normalized_score = tf.where(den == 0.0, 1.0, score / den)

            score_total += normalized_score

            self.add_metric(normalized_score, name=key + "_score")

            scores[key + "_score_num"] = score
            scores[key + "_score_den"] = den
            losses[key] = loss

        losses_normalized = losses  # currently no reweight operation

        for key, loss in losses_normalized.items():
            self.add_metric(loss, name=key + "_loss")
            loss_total += loss

        self.add_loss(loss_total)
        self.add_metric(score_total / len(self._input_columns), name="total_score")
        return [scores]


class LayoutMetricLayer(tf.keras.layers.Layer):
    """Compute Accuracy and mean IoU of the layout map."""

    def __init__(self, input_columns, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self._xsize = tf.cast(input_columns["left"]["input_dim"], tf.int32)
        self._ysize = tf.cast(input_columns["top"]["input_dim"], tf.int32)
        self._label_name = next(
            key for key, c in input_columns.items() if c["primary_label"] is not None
        )
        self._default_label = tf.cast(
            input_columns[self._label_name]["primary_label"], tf.int32
        )
        self._label_size = tf.cast(
            input_columns[self._label_name]["input_dim"], tf.int32
        )
        self._from_logits = from_logits
        assert input_columns["left"]["input_dim"] == input_columns["width"]["input_dim"]
        assert input_columns["top"]["input_dim"] == input_columns["height"]["input_dim"]

    def call(self, inputs, training=False):
        y_true, y_pred = inputs
        mask_true, mask_pred = self._get_seq_masks(y_true, y_pred, training)
        map_true = _compute_gridmaps(
            y_true,
            mask_true,
            from_logits=False,
            label_name=self._label_name,
            xsize=self._xsize,
            ysize=self._ysize,
            default_label=self._default_label,
        )
        map_pred = _compute_gridmaps(
            y_pred,
            mask_pred,
            from_logits=self._from_logits,
            label_name=self._label_name,
            xsize=self._xsize,
            ysize=self._ysize,
            default_label=self._default_label,
        )
        acc, miou = _compute_acc_miou(map_true, map_pred, self._label_size)
        self.add_metric(acc, name="layout_acc")
        self.add_metric(miou, name="layout_miou")
        return {"layout_acc": acc, "layout_miou": miou}

    def _get_seq_masks(self, y_true, y_pred, training):
        maxlen = tf.shape(y_true[self._label_name])[1]
        seq_mask_true = get_seq_mask(y_true["length"], maxlen=maxlen)
        if training:
            seq_mask_pred = seq_mask_true
        else:
            maxlen = tf.shape(y_pred[self._label_name])[1]
            seq_mask_pred = get_seq_mask(
                y_pred["length"],
                from_logits=self._from_logits,
                maxlen=maxlen,
            )
        tf.debugging.assert_rank(seq_mask_true, 2)
        tf.debugging.assert_rank(seq_mask_pred, 2)
        return seq_mask_true, seq_mask_pred


# @tf.function(experimental_relax_shapes=True)
def _compute_gridmaps(
    example,
    mask,
    from_logits,
    label_name,
    xsize,
    ysize,
    default_label,
):
    if from_logits:
        # Assume all categorical here.
        example = {
            key: tf.cast(
                tf.argmax(tf.stop_gradient(example[key]), axis=-1),
                tf.int32,
            )
            for key in ("left", "top", "width", "height", label_name)
        }
    else:
        example = {
            key: tf.cast(tf.stop_gradient(example[key]), tf.int32)
            for key in ("left", "top", "width", "height", label_name)
        }

    batch_size = tf.shape(mask)[0]
    gridmaps = tf.TensorArray(tf.int32, size=batch_size)
    for i in tf.range(batch_size):
        left = tf.reshape(example["left"][i][mask[i]], (-1,))
        top = tf.reshape(example["top"][i][mask[i]], (-1,))
        width = tf.reshape(example["width"][i][mask[i]], (-1,))
        height = tf.reshape(example["height"][i][mask[i]], (-1,))

        label = tf.cast(
            tf.reshape(example[label_name][i][mask[i]], (-1,)),
            tf.int32,
        )
        tf.assert_rank(left, 1)

        right = tf.minimum(xsize - 1, left + width)
        bottom = tf.minimum(ysize - 1, top + height)

        gridmap = _make_gridmap(
            left,
            top,
            right,
            bottom,
            label,
            ysize,
            xsize,
            default_label,
        )
        gridmaps = gridmaps.write(i, gridmap)
    return gridmaps.stack()


# @tf.function(experimental_relax_shapes=True)
def _make_gridmap(left, top, right, bottom, label, ysize, xsize, default_label):
    # Fill bbox region with the specified label.
    canvas = tf.fill((ysize, xsize), default_label)
    for j in tf.range(tf.shape(label)[0]):
        if top[j] >= bottom[j] or left[j] >= right[j]:
            continue
        y, x = tf.meshgrid(
            tf.range(top[j], bottom[j] + 1),
            tf.range(left[j], right[j] + 1),
        )
        indices = tf.stack([tf.reshape(y, (-1,)), tf.reshape(x, (-1,))], axis=1)
        updates = tf.fill((tf.shape(indices)[0],), label[j])
        canvas = tf.tensor_scatter_nd_update(canvas, indices, updates)
    return canvas


# @tf.function(experimental_relax_shapes=True)
def _compute_acc_miou(map_true, map_pred, label_size):
    batch_size = tf.shape(map_pred)[0]
    batch_index = tf.reshape(
        tf.tile(tf.range(batch_size)[:, tf.newaxis], [1, tf.size(map_pred[0])]),
        (-1,),
    )
    indices = tf.stack(
        [
            tf.cast(batch_index, tf.int32),
            tf.reshape(map_pred, (-1,)),
            tf.reshape(map_true, (-1,)),
        ],
        axis=1,
    )
    updates = tf.ones((tf.shape(indices)[0],), dtype=tf.int32)
    confusion = tf.cast(
        tf.scatter_nd(indices, updates, (batch_size, label_size, label_size)),
        tf.float32,
    )

    inter = tf.linalg.diag_part(confusion)
    union = tf.reduce_sum(confusion, axis=1) + tf.reduce_sum(confusion, axis=2) - inter

    # Compute accuracy
    acc = tf.math.truediv(
        tf.reduce_sum(inter, axis=1), tf.reduce_sum(confusion, axis=(1, 2))
    )

    # Compute nanmean of iou.
    weight = tf.cast(union > 0, tf.float32)
    iou = inter / (union + 1e-9)
    miou = tf.reduce_sum(weight * iou, axis=1) / tf.reduce_sum(weight, axis=1)
    return acc, miou
