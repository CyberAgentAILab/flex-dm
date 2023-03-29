from typing import Dict, Union

import tensorflow as tf
from mfp.data.spec import get_valid_input_columns
from mfp.models.architecture.mask import get_seq_mask
from mfp.models.architecture.utils import make_dense_options


class Decoder(tf.keras.layers.Layer):
    """Multi-way head for decoders."""

    def __init__(
        self,
        input_columns: Dict,
        context: Union[str, None] = None,
        detachment: str = "default",
        latent_dim: int = 256,
        dropout: float = 0.1,
        l2: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_columns = input_columns
        self.context = context
        self.use_canvas = context == "canvas"
        self.detachment = detachment
        self.latent_dim = latent_dim
        self.valid_input_columns = get_valid_input_columns(
            input_columns, self.use_canvas
        )

        self.decoders = {}
        for key, column in self.valid_input_columns.items():
            if column["type"] == "categorical":
                units = column["shape"][-1] * column["input_dim"]
            else:
                units = column["shape"][-1]

            self.decoders[key] = tf.keras.layers.Dense(
                units,
                name="decoder_%s" % key,
                **make_dense_options(l2),
            )

        # def compute_mask(self, z, mask=None):
        #     """Compute mask according to Keras specification."""
        #     if isinstance(z, tuple):
        #         _, z = z
        #         seq_mask = get_seq_mask(inputs['length'])
        #     else:
        #         seq_mask = self.predict_mask(z)
        #     tf.debugging.assert_rank(seq_mask, 2)

        #     outputs = {}
        #     for key, column in self.input_columns.items():
        #         if column['is_sequence']:
        #             outputs[key] = seq_mask
        #         else:
        #             outputs[key] = None
        #     return outputs

        assert detachment in ["default", "flat", "none"]
        if self.context is not None:
            assert detachment == "default"
        if self.detachment == "flat":
            self.valid_keys = self.valid_input_columns.keys()

    def predict_mask(self, z):
        length_logit = self.decoders["length"](z)
        return get_seq_mask(length_logit, from_logits=True)

    def call(self, inputs: tf.Tensor, training: bool = False):
        """Take a sequence of transformed embeddings and compute outputs."""
        if self.context in ["id", "length", "canvas"]:
            canvas = inputs[:, :1]  # for global tasks (e.g., classification)
            seq = inputs[:, 1:]
        else:
            seq = inputs

        if self.use_canvas:
            # raise NotImplementedError
            pass

        if self.detachment == "flat":
            keys = self.valid_keys
            B = tf.shape(seq)[0]
            seq = tf.reshape(seq, (B, -1, len(keys), self.latent_dim))
            seq = tf.split(seq, len(keys), axis=2)
            seq = {k: tf.squeeze(v, axis=2) for (k, v) in zip(keys, seq)}
        elif self.detachment == "none":
            B = tf.shape(inputs["left"])[0]
        else:
            B = tf.shape(seq)[0]

        # Predict output for each head.
        outputs = {}
        for key, column in self.valid_input_columns.items():
            if column["type"] == "categorical":
                shape = (column["shape"][-1], column["input_dim"])
            else:
                shape = (column["shape"][-1],)

            if column["is_sequence"]:
                input_ = seq if self.detachment == "default" else seq[key]
                outputs[key] = tf.reshape(self.decoders[key](input_), (B, -1) + shape)
                tf.debugging.assert_rank_at_least(outputs[key], 3)
            else:
                input_ = canvas
                outputs[key] = tf.reshape(self.decoders[key](input_), (B,) + shape)
                tf.debugging.assert_rank_at_least(outputs[key], 2)
        return outputs
