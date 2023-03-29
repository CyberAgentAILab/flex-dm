import logging
import random
from typing import Dict, List, Union

import tensorflow as tf
from mfp.models.architecture.mask import get_seq_mask

logger = logging.getLogger(__name__)


KEYS = ["type", "left", "top", "width", "height"]


def sort_inputs(inputs: Dict, input_columns: Dict, from_logits: bool = False):
    CONST = 100
    # assert set(inputs.keys()) == (set(input_columns.keys()))
    assert "length" in inputs
    assert tf.executing_eagerly()
    for key in KEYS:
        assert key in inputs
        assert input_columns[key]["input_dim"] < CONST

    data = {k: tf.identity(v) for (k, v) in inputs.items()}
    for key, column in input_columns.items():
        if column["is_sequence"] and column["type"] == "categorical":
            if from_logits:
                data[key] = tf.argmax(data[key], axis=-1)
            data[key] = tf.cast(data[key], tf.int64)

    invalid = tf.logical_not(get_seq_mask(data["length"]))
    priority = 0  # use int64 to avoid overflow
    for key in KEYS:
        priority = priority * CONST + data[key][..., 0]  # (B, S)
    priority += tf.cast(invalid, tf.int64) * (CONST ** len(KEYS))
    indices = tf.argsort(priority, axis=-1)

    new_inputs = {}
    for key in inputs:
        val = tf.identity(inputs[key])
        if key in input_columns and input_columns[key]["is_sequence"]:
            new_inputs[key] = tf.gather(val, indices, batch_dims=1)
        else:
            new_inputs[key] = val
    return new_inputs


def shuffle_inputs(inputs: Dict):
    """
    Used to shuffle input sets for training
    - auto-regressive models
    - models that take shuffled sets as inputs
    """
    assert "length" in inputs and "left" in inputs
    if tf.executing_eagerly():
        shape = tf.shape(inputs["left"])
        B = shape[0]
        S = shape[1]
        data = []
        for i in range(B):
            N = inputs["length"][i, 0] + 1
            x = list(range(N))
            random.shuffle(x)
            x = x + list(range(N, S))
            data.append(x)
        indices = tf.convert_to_tensor(data)

        new_inputs = {}
        for key in inputs.keys():
            val = tf.identity(inputs[key])
            if val.shape[1] == S:
                new_inputs[key] = tf.gather(val, indices, batch_dims=1)
            else:
                new_inputs[key] = val
        return new_inputs
    else:
        logger.info("Shuffling sequences in order not to feed order for autoreg models")
        # backdoor for model._make() (done in graph mode)
        return inputs


def reorganize_indices(
    from_inds: tf.Tensor, n_elems: tf.Tensor, maxlen: Union[int, None] = None
):
    """
    Used to reorganize the element order (for element-wise masking)
    """
    if tf.executing_eagerly():
        tf.debugging.assert_rank(from_inds, 2)  # (B, 1)
        tf.debugging.assert_rank(n_elems, 2)  # (B, 1)
        # tf.debugging.assert_less_equal(from_inds, n_elems)
        B = tf.shape(from_inds)[0]
        if not maxlen:
            maxlen = tf.reduce_max(n_elems).numpy() + 1
        data = []
        for i in range(B):
            from_ind = from_inds[i, 0].numpy()
            n_elem = n_elems[i, 0].numpy()
            ids = list(range(maxlen))
            del ids[from_ind]
            ids = ids[:n_elem] + [from_ind] + ids[n_elem:]
            data.append(ids)
        return tf.convert_to_tensor(data)
    else:
        # backdoor for model._make() (done in graph mode)
        B = tf.shape(n_elems)[0]
        maxlen = tf.reduce_max(n_elems) + 1
        indices = tf.tile(tf.range(maxlen)[tf.newaxis, :], (B, 1))
        return indices


def merge_list_of_dict_of_tensors(
    inputs: List[Dict[str, tf.Tensor]], axis: int = 0
) -> Dict[str, tf.Tensor]:
    result = {}
    for k in inputs[0].keys():
        result[k] = tf.concat([x[k] for x in inputs], axis=axis)
    return result


def split_dict_of_tensors(
    inputs: Dict[str, tf.Tensor], num_splits: int = 1, axis: int = 0
) -> List[Dict[str, tf.Tensor]]:
    result = [{} for _ in range(num_splits)]
    for (k, v) in inputs.items():
        for i, x in enumerate(tf.split(v, num_splits, axis=axis)):
            result[i][k] = x
            if i >= 1:  # num of dim. along axis should be divisible
                tf.debugging.assert_equal(tf.shape(x), tf.shape(result[0][k]))
    return result


if __name__ == "__main__":
    x = [
        {"a": tf.reshape(tf.range(6), (2, 3)), "b": tf.zeros((3, 2))},
        {"a": 10 + tf.reshape(tf.range(6), (2, 3)), "b": tf.ones((3, 2))},
    ]
    h = merge_list_of_dict_of_tensors(x, axis=0)
