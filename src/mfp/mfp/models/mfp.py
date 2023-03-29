import logging
from typing import Dict, List, Optional

import tensorflow as tf
import tensorflow_probability as tfp
from mfp.data.spec import get_attribute_groups, get_dataset_name
from mfp.models.architecture.mask import get_seq_mask
from mfp.models.canvasvae import CanvasVAE
from mfp.models.layoutvae import LayoutVAE
from mfp.models.masking import (
    apply_token,
    elem_masking,
    feat_masking,
    filter_padding,
    get_task_names,
    random_masking,
)
from mfp.models.model import BART, AutoReg, Model, VanillaTransformer

from .metrics import LossLayer
from .tensor_utils import shuffle_inputs, sort_inputs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# def load_weights(model: tf.keras.Model, weight_path: str):
#     model.compile(optimizer="adam")
#     logger.info(f"Loading: {weight_path}")
#     model.load_weights(weight_path)
#     return model


def get_task_cat_dist_sampler(task_names: List[str], masking_method: str):
    used_names = masking_method.split("_")
    probs = [1.0 if name in used_names else 0.0 for name in task_names]
    probs_total = sum(probs)
    assert probs_total > 0.0

    probs = [p / probs_total for p in probs]
    logger.info([item for item in zip(task_names, probs)])
    sampler = tfp.distributions.Categorical(logits=tf.math.log(probs))
    return sampler


def merge_inputs_and_prediction(inputs, input_columns, masks, prediction):
    for key, column in input_columns.items():
        if not column["is_sequence"]:
            # keep canvas attributes
            prediction[key] = inputs[key]
        elif key not in masks.keys():
            # demo only attributes
            continue
        elif column["type"] == "numerical":
            cond = masks[key][..., tf.newaxis]
            cond = tf.repeat(cond, tf.shape(prediction[key])[-1], axis=-1)
            prediction[key] = tf.where(cond, prediction[key], inputs[key])
        else:
            gt = tf.one_hot(inputs[key], depth=column["input_dim"])
            cond = masks[key][..., tf.newaxis, tf.newaxis]
            cond = tf.repeat(cond, tf.shape(gt)[-2], axis=-2)
            cond = tf.repeat(cond, tf.shape(gt)[-1], axis=-1)
            prediction[key] = tf.where(cond, prediction[key], gt)

    # copy unpredicted items for visualization
    for key, column in input_columns.items():
        if column.get("demo_only", False):
            prediction[key] = inputs[key]
    return prediction


def preprocess_for_test(inputs, input_columns, masks, tasks=None):
    seq_mask = get_seq_mask(inputs["length"])
    filtered_inputs = filter_padding(inputs, input_columns, seq_mask)

    modified_inputs = {}
    for key, column in input_columns.items():
        # don't mask variables defined for canvas
        if not column["is_sequence"]:
            modified_inputs[key] = filtered_inputs[key]
            continue

        modified_inputs[key] = apply_token(
            filtered_inputs[key], column, masks[key], "masked"
        )

    if tasks is None:
        # add dummy tensor
        tasks = tf.zeros(tf.shape(inputs["left"])[0])
    modified_inputs["task"] = tasks[..., tf.newaxis]

    return modified_inputs


def preprocess_for_train(
    inputs: Dict[str, tf.Tensor],
    input_columns: Dict,
    tasks: tf.Tensor,
    is_autoreg: bool = False,
    input_dtype: str = "set",
):
    tf.debugging.assert_rank(tasks, 1)
    attribute_groups = get_attribute_groups(input_columns.keys())

    if is_autoreg or input_dtype == "shuffled_set":
        inputs = shuffle_inputs(inputs)
    elif input_dtype == "sorted_set":
        inputs = sort_inputs(inputs, input_columns)

    seq_mask = get_seq_mask(inputs["length"])
    filtered_inputs = filter_padding(inputs, input_columns, seq_mask)

    data = []
    modified_inputs, masks = random_masking(filtered_inputs, input_columns, seq_mask)
    data.append(elem_masking(filtered_inputs, input_columns, seq_mask, is_autoreg))
    for attribute_group in attribute_groups.values():
        x = feat_masking(filtered_inputs, input_columns, seq_mask, attribute_group)
        data.append(x)

    for key in modified_inputs.keys():
        for i, (modified_inputs_tmp, masks_tmp) in enumerate(data):
            # cond = (method_probs_onehot[:, i + 1] == 1.0)
            cond = tasks == (i + 1)
            if input_columns[key]["is_sequence"]:
                cond = cond[..., tf.newaxis]

            modified_inputs[key] = tf.where(
                cond[..., tf.newaxis],
                modified_inputs_tmp[key],
                modified_inputs[key],
            )

            if input_columns[key]["is_sequence"]:
                masks[key] = tf.where(cond, masks_tmp[key], masks[key])

    # add task info.
    modified_inputs["task"] = tasks[..., tf.newaxis]
    return inputs, modified_inputs, masks


def iterative_decode(model, masks, inputs, input_columns, modified_inputs, num_iter):
    # MaskGIT-like decoding
    # NOTE: not optimal implementation, could be faster
    masks = masks.copy()
    seq_mask = get_seq_mask(inputs["length"])
    filtered_inputs = filter_padding(inputs, input_columns, seq_mask)
    categorical_keys = [
        k
        for k, v in input_columns.items()
        if v["is_sequence"] and v.get("type", None) == "categorical"
    ]
    num_masked = sum(masks[k].numpy().astype("int").sum(-1) for k in categorical_keys)
    num_update_per_iter = (num_masked / num_iter).round().astype("int")
    for i in range(num_iter):
        # predict masked fields
        outputs = model(modified_inputs, training=False)
        if i == 0:
            final_outputs = outputs

        # use top-k confident prediction
        confidence = {
            k: tf.where(
                masks[k],
                tf.reduce_mean(
                    tf.reduce_max(tf.nn.softmax(outputs[k], axis=-1), axis=-1),
                    axis=-1,
                ),  # mean(max_prob, -1); mean for "color" field
                0.0,
            )
            for k in categorical_keys
        }
        confidence_sorted = tf.sort(
            tf.concat([confidence[k] for k in categorical_keys], axis=-1),
            axis=-1,
            direction="DESCENDING",
        )
        threshold = tf.stack(
            [confidence_sorted[i, k] for i, k in enumerate(num_update_per_iter)]
        )

        # update filtered_inputs and mask
        for key in categorical_keys:
            pred = tf.argmax(outputs[key], axis=-1, output_type=tf.int32)
            update_field = (confidence[key] >= threshold) & (confidence[key] > 0)
            filtered_inputs[key] = tf.where(
                update_field[:, :, None], pred, filtered_inputs[key]
            )
            masks[key] = tf.where(masks[key] == update_field, False, masks[key])
            if i > 0:
                final_outputs[key] = tf.where(
                    update_field[:, :, None, None],
                    outputs[key],
                    final_outputs[key],
                )

        # update model input
        for key, column in input_columns.items():
            if column["is_sequence"]:
                modified_inputs[key] = apply_token(
                    filtered_inputs[key], column, masks[key], "masked"
                )

    # use last prediction for numerical fields
    for key in ["image_embedding", "text_embedding"]:
        final_outputs[key] = outputs[key]

    return final_outputs


class MFP(tf.keras.Model):
    """
    MFP trainer.
    """

    def __init__(
        self,
        input_columns: Dict,
        num_blocks: int = 4,
        block_type: str = "deepsvg",
        masking_method: str = "random",
        seq_type: str = "default",
        arch_type: str = "oneshot",
        context: Optional[str] = None,
        input_dtype: str = "set",
        name: str = "mfp",
        use_elemwise_noise: bool = False,
        **kwargs,  # keys are latent_dim, dropout, l2
    ):
        super().__init__(name=name)
        assert arch_type == "oneshot"
        self.arch_type = arch_type
        self.context = context
        self.input_dtype = input_dtype

        self.input_columns = {
            k: v for (k, v) in input_columns.items() if not v.get("demo_only", False)
        }

        self.is_autoreg = False if arch_type in ["oneshot", "canvasvae"] else True

        if arch_type.endswith("vae") and "kl" in kwargs:
            del kwargs["kl"]  # won't use it

        if arch_type == "oneshot":
            model_class = {
                "default": Model,
                "flat": VanillaTransformer,
            }[seq_type]
            self.model = model_class(
                input_columns=input_columns,
                num_blocks=num_blocks,
                block_type=block_type,
                context=context,
                input_dtype=input_dtype,
                use_elemwise_noise=use_elemwise_noise,
                **kwargs,
            )
        elif "vae" in arch_type:
            kl = kwargs.pop("kl") if "kl" in kwargs else 1.0
            model_class = {
                "layoutvae": LayoutVAE,
                "canvasvae": CanvasVAE,
            }[arch_type]

            self.model = model_class(
                input_columns=input_columns,
                num_blocks=num_blocks,
                block_type=block_type,
                input_dtype=input_dtype,
                kl=kl,
                **kwargs,
            )
        elif "autoreg" in arch_type:
            model_class = {
                "autoreg": AutoReg,
                "bart_autoreg": BART,
            }[arch_type]
            self.model = model_class(
                input_columns=input_columns,
                num_blocks=num_blocks,
                block_type=block_type,
                context=context,
                input_dtype=input_dtype,
                **kwargs,
            )
        else:
            raise NotImplementedError

        self.loss_layer = LossLayer(input_columns)

        self.task_names = get_task_names(input_columns)
        self.task_cat_dist = get_task_cat_dist_sampler(self.task_names, masking_method)
        if get_dataset_name(input_columns.keys()) == "rico":
            self.sort_pos = True
        else:
            self.sort_pos = False

    def call(self, inputs, training=False, demo_args=None):
        is_demo = True if demo_args else False
        B = tf.shape(inputs["left"])[0]
        tasks = self.task_cat_dist.sample(B)

        if is_demo:
            targets = inputs
            masks = demo_args["masks"]
            modified_inputs = preprocess_for_test(
                inputs,
                self.input_columns,
                masks,
                demo_args.get("tasks", tasks),
            )
        else:
            targets, modified_inputs, masks = preprocess_for_train(
                inputs,
                self.input_columns,
                tasks,
                is_autoreg=self.is_autoreg,
                input_dtype=self.input_dtype,
            )

        iter_decode = False
        if is_demo:
            num_iter = demo_args.get("num_iter", 1)
            iter_decode = num_iter > 1

        if iter_decode:
            outputs = iterative_decode(
                self.model, masks, inputs, self.input_columns, modified_inputs, num_iter
            )
        elif self.is_autoreg:
            outputs = self.model(modified_inputs, targets, masks, training)
        else:
            outputs = self.model(modified_inputs, training)

        if not is_demo:
            if self.sort_pos:
                ind = self.task_names.index("pos")
                self.loss_layer((targets, outputs, masks), training, (tasks == ind))
            else:
                self.loss_layer((targets, outputs, masks), training)

        outputs = merge_inputs_and_prediction(
            inputs, self.input_columns, masks, outputs
        )

        outputs["tasks"] = tasks
        return outputs
