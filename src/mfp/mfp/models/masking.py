from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from mfp.data.spec import get_attribute_groups

MASK_VALUE = 10.0
NULL_VALUE = 0.0

MASK_PROB = 0.15
REPLACE_PROB = 0.1
UNCHANGE_PROB = 0.1
CHANGE_PROB = 1.0 - UNCHANGE_PROB
THRESH = REPLACE_PROB / CHANGE_PROB


def get_task_names(input_columns):
    task_names = ["random", "elem"]
    task_names += list(get_attribute_groups(input_columns.keys()).keys())
    return task_names


def filter_padding(
    inputs: Dict[str, tf.Tensor], input_columns: Dict, mask: tf.Tensor
) -> Dict[str, tf.Tensor]:
    modified_inputs = {}

    # to set [NULL] for padding caused by making minibatch
    # from variable-length elements
    unused_mask = tf.logical_not(mask)

    for key, column in input_columns.items():
        input_ = inputs[key]
        if column["is_sequence"]:
            # to set [NULL] for invalid data
            # (e.g., TextElement does not have image_embedding)
            if "loss_condition" in column:
                cond = column["loss_condition"]
                mask_ = tf.fill(tf.shape(mask), False)
                for i, flag in enumerate(cond["mask"]):
                    if not flag:
                        mask_ = tf.math.logical_or(
                            mask_, (inputs[cond["key"]] == i)[..., 0]
                        )
                mask_ = tf.logical_or(mask_, unused_mask)
            else:
                mask_ = unused_mask
            modified_inputs[key] = apply_token(input_, column, mask_, "unused")
        else:
            modified_inputs[key] = input_

    return modified_inputs


def get_initial_masks(input_columns: Dict, mask: tf.Tensor) -> Dict[str, tf.Tensor]:
    # returning masks with all False
    masks = {}
    for key, column in input_columns.items():
        # don't mask variables defined for canvas
        if not column["is_sequence"]:
            masks[key] = tf.fill(tf.shape(mask)[:1], True)
        else:
            masks[key] = tf.fill(tf.shape(mask), False)
    return masks


def apply_token(
    input_: tf.Tensor, column: Dict[str, Any], mask: tf.Tensor, token_type: str
) -> tf.Tensor:
    assert token_type in ["masked", "unused", "random"]
    tf.debugging.assert_equal(tf.rank(mask), 2)
    tf.debugging.assert_equal(tf.rank(input_), 3)

    mask = mask[..., tf.newaxis]
    shape = tf.shape(input_)

    if column["type"] == "categorical":
        x = tf.cast(mask, dtype=tf.int32)
        data = {
            "masked": column["input_dim"],
            "unused": column["input_dim"] + 1,
            "random": tf.random.uniform(shape, 0, column["input_dim"], dtype=tf.int32),
        }
        output = input_ * (1 - x) + data[token_type] * x
    else:
        x = tf.cast(mask, dtype=tf.float32)
        data = {
            "masked": MASK_VALUE,
            "unused": NULL_VALUE,
            "random": tf.random.normal(shape, stddev=0.1),
        }
        output = input_ * (1.0 - x) + data[token_type] * x

    return output


def select_single_element(mask: tf.Tensor, select_last: bool = False) -> tf.Tensor:
    """
    Select a single element for each sample.
    If mask is all False, then return an array filled with False
    For autoregressive models, always return the last valid element
    """
    tf.debugging.assert_rank(mask, 2)  # (B, S)

    length = tf.cast(tf.reduce_sum(tf.cast(mask, tf.int64), axis=1), tf.float32)
    if select_last:
        arr = tf.cast(length - 1, tf.int32)
    else:
        arr = tf.cast(tf.random.uniform(tf.shape(mask)[:1]) * length, tf.int32)
    new_mask = tf.cast(tf.one_hot(arr, depth=tf.shape(mask)[1]), tf.bool)
    new_mask = new_mask & (length > 0.0)[:, tf.newaxis]
    return new_mask


def feat_masking(
    inputs: Dict[str, tf.Tensor],
    input_columns: Dict,
    mask: tf.Tensor,
    feat_group: List[str],
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    modified_inputs = {}
    for key in inputs.keys():
        modified_inputs[key] = tf.identity(inputs[key])

    masks = get_initial_masks(input_columns, mask)

    for key in feat_group:
        column = input_columns[key]
        modified_inputs[key] = apply_token(modified_inputs[key], column, mask, "masked")
        masks[key] = mask

    return modified_inputs, masks


def elem_masking(
    inputs: Dict[str, tf.Tensor],
    input_columns: Dict,
    mask: tf.Tensor,
    is_autoreg: bool = False,
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    # modifing a specific element of all the features in a sequence
    masks = get_initial_masks(input_columns, mask)
    selected_mask = select_single_element(mask, is_autoreg)

    modified_inputs = {}
    for key, column in input_columns.items():
        if not column["is_sequence"]:
            modified_inputs[key] = inputs[key]
        else:
            modified_inputs[key] = apply_token(
                inputs[key], column, selected_mask, "masked"
            )
            masks[key] = selected_mask
    return modified_inputs, masks


def unused_masking(
    inputs: Dict[str, tf.Tensor],
    input_columns: Dict,
    masks: Dict[str, tf.Tensor],
    drop_ratio: float = 0.1,
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    dist = tfp.distributions.Bernoulli(probs=drop_ratio)

    modified_inputs = {}
    modified_masks = {}
    for key, column in input_columns.items():
        if not column["is_sequence"]:
            modified_masks[key] = masks[key]
            modified_inputs[key] = inputs[key]
            continue

        is_masked = masks[key]  # (B, S)
        is_unused = tf.cast(dist.sample(tf.shape(is_masked)[:1]), tf.bool)
        is_unused = is_unused[:, tf.newaxis, tf.newaxis]
        modified_masks[key] = tf.logical_and(is_masked, tf.logical_not(is_unused))
        modified_inputs[key] = apply_token(inputs[key], column, is_unused, "unused")

    return modified_inputs, masks


def rowcol_random_masking(
    inputs: Dict[str, tf.Tensor],
    input_columns: Dict,
    mask: tf.Tensor,
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    modified_inputs = {}
    masks = {}

    B = tf.shape(inputs["left"])[0]
    S = tf.shape(inputs["left"])[1]
    F = len(input_columns.values())
    p = MASK_PROB / 2.0
    col_mask = tf.random.uniform((B, S), minval=0.0, maxval=1.0) < p
    row_mask = tf.random.uniform((B, F), minval=0.0, maxval=1.0) < p

    for i, (key, column) in enumerate(input_columns.items()):
        # don't mask variables defined for canvas
        if not column["is_sequence"]:
            modified_inputs[key] = inputs[key]
            masks[key] = tf.fill(tf.shape(inputs[key]), True)
            continue

        # merge X-wise mask, and the latter steps are exactly the same as random
        mfp_mask = mask & (col_mask | row_mask[:, i : i + 1])

        # 80% mask, 10% random token, 10% unchanged
        chg_mask = mfp_mask & (
            tf.random.uniform(tf.shape(mfp_mask), minval=0.0, maxval=1.0) < CHANGE_PROB
        )
        rand_arr = tf.random.uniform(tf.shape(chg_mask), minval=0.0, maxval=1.0)
        masked_input = apply_token(
            inputs[key], column, chg_mask & (rand_arr >= THRESH), "masked"
        )
        masked_input = apply_token(
            masked_input, column, chg_mask & (rand_arr < THRESH), "random"
        )

        # update input
        modified_inputs[key] = masked_input
        masks[key] = mfp_mask

    return modified_inputs, masks


def random_masking(
    inputs: Dict[str, tf.Tensor],
    input_columns: Dict,
    mask: tf.Tensor,
) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """
    Like standard MLM training, do some operations for 15% of the tokens
    # 80% mask, 10% random token, 10% unchanged
    """

    # run in eager mode because of random sampling outside tf function
    modified_inputs = {}
    masks = {}

    # for random masking
    for key, column in input_columns.items():
        # don't mask variables defined for canvas
        if not column["is_sequence"]:
            modified_inputs[key] = inputs[key]
            masks[key] = tf.fill(tf.shape(inputs[key]), True)
            continue

        # create mask with shape (B, S) while ignoring padded region
        rand_arr = tf.random.uniform(tf.shape(inputs[key])[:-1], minval=0.0, maxval=1.0)
        mfp_mask = mask & (rand_arr < MASK_PROB)

        # 80% mask, 10% random token, 10% unchanged
        chg_mask = mfp_mask & (
            tf.random.uniform(tf.shape(mfp_mask), minval=0.0, maxval=1.0) < CHANGE_PROB
        )
        rand_arr = tf.random.uniform(tf.shape(chg_mask), minval=0.0, maxval=1.0)
        masked_input = apply_token(
            inputs[key], column, chg_mask & (rand_arr >= THRESH), "masked"
        )
        masked_input = apply_token(
            masked_input, column, chg_mask & (rand_arr < THRESH), "random"
        )

        # update input
        modified_inputs[key] = masked_input
        masks[key] = mfp_mask

    return modified_inputs, masks
