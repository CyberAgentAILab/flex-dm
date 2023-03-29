from typing import Dict, Optional

import tensorflow as tf
import tensorflow_probability as tfp
from einops import rearrange
from mfp.data.spec import get_valid_input_columns
from mfp.models.architecture.cvae import Head
from mfp.models.architecture.decoder import Decoder
from mfp.models.architecture.encoder import Encoder
from mfp.models.architecture.mask import Unmask, get_seq_mask
from mfp.models.architecture.transformer import Blocks, PositionEmbedding
from mfp.models.architecture.utils import make_dense_options, make_emb_options

MND = tfp.distributions.MultivariateNormalDiag


class CanvasVAE(tf.keras.layers.Layer):
    def __init__(
        self,
        input_columns: Dict,
        num_blocks: int = 4,
        block_type: str = "deepsvg",
        context: Optional[str] = "length",
        input_dtype: str = "set",
        kl: float = 1e-0,
        **kwargs,  # keys are latent_dim, dropout, l2
    ):
        super().__init__()
        assert context == "length"
        assert input_dtype == "sorted_set"

        # assert "l2" in kwargs and "latent_dim" in kwargs
        l2 = kwargs.get("l2", None)
        dropout = kwargs.get("dropout", 0.0)

        self.kl = kl
        self.input_columns = input_columns
        self.valid_input_columns = get_valid_input_columns(input_columns, False)

        self.encoder = Encoder(
            input_columns, context=context, input_dtype=input_dtype, **kwargs
        )
        self.decoder = Decoder(input_columns, **kwargs)

        self.enc_blocks = Blocks(
            num_blocks=num_blocks // 2,
            block_type=block_type,
            lookahead=True,
            conditional=True,
            **kwargs,
        )
        self.prior_head = Head(**kwargs, compute_kl=True)
        self.norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation("relu")
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.unmask = Unmask()

        self.blocks = Blocks(
            num_blocks=num_blocks // 2,
            block_type=block_type,
            lookahead=True,
            conditional=True,
            **kwargs,
        )
        self.length_fc = tf.keras.layers.Dense(
            input_columns["length"]["input_dim"], **make_dense_options(l2)
        )
        self.length_loss_func = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
        )

        self.embedding_const = PositionEmbedding(
            kwargs["latent_dim"],
            self.input_columns["length"]["input_dim"],
            dropout=dropout,
            emb_options=make_emb_options(l2),
            name="embedding_const",
        )

    def call(
        self,
        inputs: Dict,
        training: bool,
    ):
        # note that the element in seq. is for canvas attributes
        h_masked, enc_mask = self.encoder(inputs, training=training)
        canvas = h_masked[:, 0]
        sequence = h_masked[:, 1:]
        enc_mask = enc_mask[:, 1:]
        B = enc_mask.shape[0]
        h = self.enc_blocks((sequence, canvas), enc_mask, training=training)

        # aggregate latent codes and sample
        pooled = self.norm(sequence, training=training)
        pooled = self.pooling(self.relu(pooled))  # (B, S, D) -> (B, D)
        pooled = self.unmask(pooled)
        z = self.prior_head(pooled, training=training)["z"]

        # get the length of sequence at first
        if training:
            length_logits = self.length_fc(z)
            length_loss = self.length_loss_func(inputs["length"], length_logits)
            self.add_loss(length_loss)
            self.add_metric(length_loss, name="length_loss")

            # At training, use the supplied GT mask.
            mask = get_seq_mask(inputs["length"])
        else:
            length_pred = tf.argmax(self.length_fc(z), axis=1)
            maxlen = tf.reduce_max(inputs["length"]) + 1
            mask = get_seq_mask(rearrange(length_pred, "b -> b 1"), maxlen=maxlen)

        sequence = self.embedding_const(mask, training=training)
        h = self.blocks((sequence, z), mask, training=training)
        outputs = self.decoder(h, training=training)
        return outputs
