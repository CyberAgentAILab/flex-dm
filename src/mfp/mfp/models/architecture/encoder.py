from typing import Dict, Union

import tensorflow as tf
from einops import rearrange
from mfp.data.spec import get_valid_input_columns
from mfp.models.architecture.mask import get_seq_mask
from mfp.models.architecture.transformer import PositionEmbedding
from mfp.models.architecture.utils import make_dense_options, make_emb_options
from mfp.models.masking import MASK_VALUE, NULL_VALUE, get_task_names

CONTEXT_NAMES = [None, "id", "canvas", "length", "canvas_add"]


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        input_columns: Dict,
        context: Union[str, None] = None,
        input_dtype: str = "set",
        use_elemwise_noise: bool = False,
        fusion: str = "add",
        latent_dim: int = 128,
        dropout: float = 0.1,
        l2: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert context in CONTEXT_NAMES
        # canvas_add: for CTX experiments, add canvas info. to each element in seq.

        self.input_columns = input_columns

        # to encode and aggregate canvas-level attributes
        self.use_canvas = context is not None and "canvas" in context
        self.use_elemwise_noise = use_elemwise_noise
        self.valid_input_columns = get_valid_input_columns(
            input_columns, self.use_canvas
        )

        self.context = context
        self.use_pos_token = True if input_dtype != "set" else False
        self.fusion = fusion
        self.latent_dim = latent_dim

        self.input_layer = {}

        # for shuffled sets or sequence
        if self.use_pos_token:
            self.input_layer["const"] = PositionEmbedding(
                latent_dim,
                self.input_columns["length"]["input_dim"],
                dropout=dropout,
                emb_options=make_emb_options(l2),
                name="input_const",
            )

        if self.use_elemwise_noise:
            self.noise_size = 4
            self.input_layer["noise_fc"] = tf.keras.layers.Dense(
                units=latent_dim,
                name="input_noise",
                **make_dense_options(l2),
            )

        # for global context features
        # initialize <CLS> token
        # w_init = tf.zeros_initializer()(
        #     shape=[1, 1, latent_dim], dtype=tf.float32
        # )
        # self.cls_token[key] = tf.Variable(initial_value=w_init, trainable=True)

        for key, column in self.valid_input_columns.items():
            if column["type"] == "categorical":
                self.input_layer[key] = tf.keras.layers.Embedding(
                    input_dim=column["input_dim"] + 2,
                    output_dim=latent_dim,
                    name="input_%s" % key,
                    **make_emb_options(l2),
                )
            elif column["type"] == "numerical":
                # used to embed <MASK> and <UNUSED> token
                self.input_layer["%s_special" % key] = tf.keras.layers.Embedding(
                    input_dim=2,
                    output_dim=latent_dim,
                    name="input_%s_special" % key,
                    **make_emb_options(l2),
                )
                self.input_layer[key] = tf.keras.layers.Dense(
                    units=latent_dim,
                    name="input_%s" % key,
                    **make_dense_options(l2),
                )
            else:
                raise ValueError("Invalid column: %s" % column)

        if self.context == "id":
            task_len = len(get_task_names(input_columns))
            self.input_layer["task"] = tf.keras.layers.Embedding(
                input_dim=task_len,
                output_dim=latent_dim,
                name="input_task",
                **make_emb_options(l2),
            )
        elif self.context == "length":
            self.input_layer["length"] = tf.keras.layers.Embedding(
                input_dim=input_columns["length"]["input_dim"],
                output_dim=latent_dim,
                name="input_task",
                **make_emb_options(l2),
            )

        assert fusion in ["add", "concat", "flat", "none"]
        if self.fusion == "concat":
            self.fusion = tf.keras.layers.Sequential(
                [
                    tf.keras.layers.Dense(
                        units=latent_dim,
                        name="fusion_fc",
                        **make_dense_options(l2),
                    ),
                    tf.keras.layers.LayerNormalization(),
                    tf.keras.layers.Dropout(dropout),
                ]
            )
        elif self.fusion == "flat":
            valid_feats = self.valid_input_columns.keys()
            maxlen = len(valid_feats)
            maxlen *= self.input_columns["length"]["input_dim"] + 1
            self.input_layer["emb_seq_pos"] = PositionEmbedding(
                latent_dim,
                self.input_columns["length"]["input_dim"] + 1,
                dropout=dropout,
                emb_options=make_emb_options(l2),
                name="input_emb_elem",
            )

            # valid_feats = self.valid_input_columns.keys()
            # self.input_layer["emb_feat"] = PositionEmbedding(
            #     latent_dim,
            #     len(valid_feats) + 1,
            #     # len(valid_feats),
            #     dropout=dropout,
            #     emb_options=make_emb_options(l2),
            #     name="input_emb_feat",
            # )

    def call(self, inputs: Dict, training: bool = False):
        B = tf.shape(inputs["length"])[0]
        # Sequence inputs.
        # Note that length is zero-based
        seq_mask = get_seq_mask(inputs["length"])

        # aggregate info for both canvas and sequence
        data_c, data_s, keys_c, keys_s = [], [], [], []
        for key, column in self.valid_input_columns.items():
            if column["type"] == "categorical":
                x = self.input_layer[key](inputs[key])
                # sum across multiple prediction targets (e.g., RGB)
                axis = 2 if column["is_sequence"] else 1
                x = tf.reduce_sum(x, axis=axis)
            else:
                # find vector corresponding to <MASK> and <UNUSED>,
                # and then retrieve dense embedding for both
                # see apply_token in mfp.models.masking
                is_masked = tf.math.reduce_all(inputs[key] == MASK_VALUE, axis=2)
                is_unused = tf.math.reduce_all(inputs[key] == NULL_VALUE, axis=2)
                masked_emb = self.input_layer["%s_special" % key](
                    tf.zeros(tf.shape(seq_mask))
                )
                unused_emb = self.input_layer["%s_special" % key](
                    tf.ones(tf.shape(seq_mask))
                )
                x = self.input_layer[key](inputs[key])
                x = tf.where(is_masked[..., tf.newaxis], masked_emb, x)
                x = tf.where(is_unused[..., tf.newaxis], unused_emb, x)

            # for global context features
            # cls_token = tf.tile(self.cls_token[key], [batch, 1, 1])
            # x = tf.concat([cls_token, x], axis=1)
            if column["is_sequence"]:
                data_s.append(x)
                keys_s.append(key)
            else:
                data_c.append(x)
                keys_c.append(key)

        if self.use_canvas:
            assert len(keys_c) > 0, (keys_s, keys_c)

        if self.fusion != "add":
            # did not implement unusual cases
            assert len(data_c) == 0

        if self.fusion == "add":
            seq, canvas = 0.0, 0.0
            for d in data_s:
                seq += d
            for d in data_c:
                canvas += d
        elif self.fusion == "flat":
            shape = tf.shape(inputs["left"])

            S = shape[1]
            F = len(data_s)
            D = self.latent_dim

            seq_mask = tf.repeat(seq_mask, F, axis=1)  # (B, S * F)
            seq = tf.concat([tf.expand_dims(d, axis=2) for d in data_s], axis=2)
            seq = tf.reshape(seq, (B, -1, D))  # (B, S * F, D)

            seq_ids = rearrange(tf.range(S * F), "s -> 1 s")
            seq += self.input_layer["emb_seq_pos"](seq_ids)  # (B, S * F, D)

            # elem_ids = (tf.range(S * F) // F)[:, tf.newaxis]  # (S * F, 1)
            # elem_emb = self.input_layer["emb_elem"](elem_ids)  # (S * F, 1, D)
            # elem_emb = tf.transpose(elem_emb, [1, 0, 2])  # (1, S * F, D)
            # feat_ids = (tf.range(S * F) % F)[:, tf.newaxis]  # (S * F, 1)
            # feat_emb = self.input_layer["emb_feat"](feat_ids)  # (S * F, 1, D)
            # feat_emb = tf.transpose(feat_emb, [1, 0, 2])  # (1, S * F, D)
            # seq = seq + elem_emb + feat_emb
        elif self.fusion == "none":
            seq = {}
            for k, v in zip(keys_s, data_s):
                seq[k] = v
        else:
            raise NotImplementedError

        if self.context == "canvas_add":
            canvas = rearrange(canvas, "b c -> b 1 c")
            seq += canvas
        elif self.context is not None:
            assert self.fusion == "add", self.fusion
            # add special token (currently including task information if available)
            if self.context == "id":
                task = inputs["task"]
                task = task[:, 0] if tf.rank(task).numpy() == 2 else task
                canvas = self.input_layer["task"](task)
            elif self.context == "length":
                length = inputs["length"]
                if tf.rank(length) == 2:
                    length = length[:, 0]
                canvas = self.input_layer["length"](length)
            elif self.context == "canvas":
                pass
            else:
                raise NotImplementedError
            canvas = rearrange(canvas, "b c -> b 1 c")
            seq = tf.concat([canvas, seq], axis=1)
            seq_mask = get_seq_mask(inputs["length"] + 1)

        if self.use_pos_token and not self.fusion == "flat":
            seq += self.input_layer["const"](seq_mask, training=training)

        if self.use_elemwise_noise:
            assert self.fusion == "add"
            shape = seq.shape[:2] + (self.noise_size,)
            noise = tf.random.normal(shape)
            seq += self.input_layer["noise_fc"](noise)

        if self.fusion == "none":
            for v in seq.values():
                tf.debugging.assert_rank(v, 3)
        else:
            tf.debugging.assert_rank(seq, 3)
        return seq, seq_mask
