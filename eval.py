import argparse
import csv
import itertools
import json
import logging
import os
import random
from collections import defaultdict

import numpy as np
import tensorflow as tf
from fsspec.core import url_to_fs
from mfp.data import DataSpec
from mfp.data.spec import get_attribute_groups, get_dataset_name
from mfp.models.architecture.mask import get_seq_mask
from mfp.models.masking import get_initial_masks, get_task_names, random_masking
from mfp.models.metrics import LossLayer
from mfp.models.mfp import MFP
from mfp.models.tensor_utils import reorganize_indices
from omegaconf import OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)
# logging.basicConfig(level=logging.INFO)

# fix seeds for reproducibility and stable validation
seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)


def evaluate(args, model, dataset, input_columns, group):
    if group:
        group_name, group_keys = group
    logger.info(f"Test on mode: {args.task_mode} feat: {group}")

    dataset_name = get_dataset_name(input_columns.keys())
    sort_pos = True if dataset_name == "rico" else False

    # define losses
    loss_layer = LossLayer(input_columns)
    total = defaultdict(float)

    iterator = iter(dataset)
    for _ in tqdm(range(args.steps_per_epoch), dynamic_ncols=True):
        example = iterator.get_next()

        B, S = example["left"].shape[:2]
        if S == 0:
            continue

        seq_mask = get_seq_mask(example["length"])
        masks = get_initial_masks(input_columns, seq_mask)

        if args.task_mode == "random":
            _, masks = random_masking(
                example,
                input_columns,
                seq_mask,
                replace_prob=0.0,
                unchange_prob=0.0,
            )
        elif args.task_mode == "elem":
            if model.arch_type in ["oneshot", "canvasvae"]:
                mask = tf.cast(tf.eye(S), tf.bool)
                for key, column in input_columns.items():
                    example[key] = tf.repeat(example[key], S, axis=0)
                    if column["is_sequence"]:
                        masks[key] = mask
            elif model.arch_type in ["autoreg", "bart_autoreg", "layoutvae"]:
                mask = tf.cast(tf.eye(S), tf.bool)
                indices = tf.range(S)[:, tf.newaxis]
                length = tf.tile(example["length"], (S, 1))
                indices = reorganize_indices(indices, length)

                # https://www.tensorflow.org/api_docs/python/tf/gather
                mask = tf.gather(mask, indices, axis=1, batch_dims=1)

                for key, column in input_columns.items():
                    example[key] = tf.repeat(example[key], S, axis=0)
                    if column["is_sequence"]:
                        example[key] = tf.gather(
                            example[key], indices, axis=1, batch_dims=1
                        )
                        masks[key] = mask
            else:
                raise NotImplementedError
        else:
            for key in group_keys:
                masks[key] = seq_mask
        demo_args = {"masks": masks}

        # set for MaskGIT-like decoding
        demo_args["num_iter"] = args.num_iter

        id_ = get_task_names(input_columns).index(group_name)
        if model.context == "id":
            demo_args["tasks"] = tf.fill(tf.shape(example["left"])[:1], id_)

        prediction = model(example, training=False, demo_args=demo_args)
        if sort_pos and args.task_mode == "pos":
            sort_flag = tf.fill((B,), True)
            (scores_tmp,) = loss_layer((example, prediction, masks), False, sort_flag)
        else:
            (scores_tmp,) = loss_layer((example, prediction, masks))
        for k, v in scores_tmp.items():
            total[k] += v.numpy()

    ans = {}
    for k in input_columns:
        num_key, den_key = f"{k}_score_num", f"{k}_score_den"
        if num_key in total.keys():
            val = total[num_key] / total[den_key]
            ans[k] = val
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-dir",
        required=True,
        help="The GCS or local path of logs and saved models.",
    )
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--task_mode", type=str, default="attr")
    parser.add_argument("--feature", type=str, default="all")
    parser.add_argument("--model", type=str, default="mfp")
    parser.add_argument("--num_iter", type=int, default=1)
    parser.add_argument("--result_csv", type=str, default="")
    args = parser.parse_args()

    fs, _ = url_to_fs(args.job_dir)
    json_path = os.path.join(args.job_dir, "args.json")
    with fs.open(json_path, "r") as file_obj:
        train_args = OmegaConf.create(json.load(file_obj))

    if args.task_mode in ["elem"]:
        if args.batch_size != 1:
            args.batch_size = 1
    logger.info(args)

    dataspec = DataSpec(
        train_args.dataset_name, train_args.data_dir, batch_size=args.batch_size
    )

    input_columns = dataspec.make_input_columns()
    dataset = dataspec.make_dataset("test", shuffle=False)
    args.steps_per_epoch = dataspec.steps_per_epoch("test", args.batch_size)

    if args.model == "mfp":
        model = MFP(
            input_columns,
            latent_dim=train_args.latent_dim,
            num_blocks=train_args.num_blocks,
            block_type=train_args.block_type,
            context=train_args.context,
            masking_method=train_args.masking_method,
            seq_type=train_args.seq_type,
            arch_type=train_args.arch_type,
            input_dtype=train_args.input_dtype,
        )
    else:
        raise NotImplementedError

    weight_path = os.path.join(args.job_dir, "checkpoints", "best.ckpt")
    model.compile(optimizer="adam")  # dummy but necessary for loading weights
    logger.info(f"Loading: {weight_path}")
    model.load_weights(weight_path)
    attribute_groups = get_attribute_groups(input_columns.keys())

    ans_all = {}
    if args.task_mode in ["elem", "random"]:
        ans_all["all"] = evaluate(args, model, dataset, input_columns, None)
    elif args.task_mode == "all_feat":
        for group in attribute_groups.items():
            if group[0] == "type":
                continue
            ans_all[group[0]] = evaluate(args, model, dataset, input_columns, group)
    else:  # assume testing a single group
        group = (args.task_mode, attribute_groups[args.task_mode])
        ans_all[args.task_mode] = evaluate(args, model, dataset, input_columns, group)

    # merge answers
    final_results = {}
    for ans in ans_all.values():
        for k, v in ans.items():
            if v == v:  # avoid nan
                final_results[k] = round(v, 4)
    print(final_results)

    if args.result_csv:
        with open(args.result_csv, "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(list(final_results.keys()))
            writer.writerow(list(final_results.values()))


if __name__ == "__main__":
    main()
