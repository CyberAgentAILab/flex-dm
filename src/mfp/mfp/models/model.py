from typing import Dict, Union

import tensorflow as tf
from mfp.models.architecture.decoder import Decoder
from mfp.models.architecture.encoder import Encoder
from mfp.models.architecture.transformer import Blocks, CrossBlocks


class _OneShot(tf.keras.layers.Layer):
    def __init__(
        self,
        input_columns: Dict,
        num_blocks: int = 4,
        block_type: str = "deepsvg",
        **kwargs,  # keys are latent_dim, dropout, l2
    ):
        super().__init__()
        self.arch_type = "oneshot"
        self.encoder, self.decoder = None, None
        self.blocks = Blocks(
            num_blocks=num_blocks,
            block_type=block_type,
            **kwargs,
        )

    def call(self, inputs, training):
        h, mask = self.encoder(inputs, training=training)
        h = self.blocks(h, mask, training=training)
        outputs = self.decoder(h, training=training)
        return outputs


class Model(_OneShot):
    def __init__(
        self,
        input_columns: Dict,
        num_blocks: int = 4,
        block_type: str = "deepsvg",
        context: Union[str, None] = None,
        input_dtype: str = "set",
        use_elemwise_noise: bool = False,
        **kwargs,
    ):
        super().__init__(input_columns, num_blocks, block_type, **kwargs)
        self.encoder = Encoder(
            input_columns,
            context=context,
            input_dtype=input_dtype,
            use_elemwise_noise=use_elemwise_noise,
            **kwargs,
        )
        self.decoder = Decoder(input_columns, context=context, **kwargs)


class VanillaTransformer(_OneShot):
    def __init__(
        self,
        input_columns: Dict,
        num_blocks: int = 4,
        block_type: str = "deepsvg",
        context: Union[str, None] = None,
        input_dtype: str = "set",
        use_elemwise_noise: bool = False,
        **kwargs,
    ):
        super().__init__(input_columns, num_blocks, block_type, **kwargs)
        assert input_dtype == "shuffled_set"
        self.encoder = Encoder(
            input_columns, fusion="flat", input_dtype=input_dtype, **kwargs
        )
        self.decoder = Decoder(input_columns, detachment="flat", **kwargs)


class _AutoReg(tf.keras.layers.Layer):
    def __init__(
        self,
        input_columns: Dict,
        num_blocks: int = 4,
        block_type: str = "deepsvg",
        context: Union[str, None] = None,
        input_dtype: str = "set",
        **kwargs,  # keys are latent_dim, dropout, l2
    ):
        super().__init__()
        self.add_masked_input = False
        # self.add_masked_input = True

        self.lookahead = False
        self.latent_dim = kwargs["latent_dim"]
        dim = self.latent_dim // 2 if self.add_masked_input else self.latent_dim
        self.input_columns = get_valid_input_columns(input_columns)

        self.encoder = Encoder(input_columns, input_dtype=input_dtype, **kwargs)
        self.decoder = Decoder(input_columns, **kwargs)

        initializer = tf.random_normal_initializer()
        self.bos = tf.Variable(
            initial_value=initializer(shape=(1, 1, dim), dtype=tf.float32),
            trainable=True,
        )
        if self.add_masked_input:
            self.dimred = tf.keras.layers.Dense(
                units=dim,
                name="dimred",
                **make_dense_options(kwargs.get("l2", None)),
            )

    def _compute_next(self, h, mask, inputs, mfp_masks):
        # Transform sequence and get the last element.
        if isinstance(mask, tuple):
            # (tgt_mask, memory_mask)
            B = tf.shape(mask[0])[0]
            S = tf.shape(mask[0])[1]
        else:
            B = tf.shape(mask)[0]
            S = tf.shape(mask)[1]

        h = self.blocks(h, mask, training=False)
        h_t = h[:, S - 1 : S]

        # Get output (=next input) at step t.
        outputs_t = self.decoder(h_t, training=False)
        new_inputs = {}

        for key, column in self.input_columns.items():
            if column["is_sequence"]:
                if column["type"] == "categorical":
                    outputs_t[key] = tf.argmax(
                        outputs_t[key], axis=-1, output_type=tf.int32
                    )
                new_inputs[key] = tf.where(
                    mfp_masks[key][:, S - 1 : S, tf.newaxis],
                    outputs_t[key],
                    inputs[key][:, S - 1 : S],
                )

        new_inputs["length"] = tf.zeros((B, 1))
        next_elem, _ = self.encoder(new_inputs)

        tf.debugging.assert_rank(next_elem, 3)
        return next_elem


class AutoReg(_AutoReg):
    def __init__(
        self,
        input_columns: Dict,
        num_blocks: int = 4,
        block_type: str = "deepsvg",
        context: Union[str, None] = None,
        input_dtype: str = "set",
        **kwargs,  # keys are latent_dim, dropout, l2
    ):
        super().__init__(
            input_columns=input_columns,
            num_blocks=num_blocks,
            block_type=block_type,
            context=context,
            input_dtype=input_dtype,
            **kwargs,
        )
        self.blocks = Blocks(
            num_blocks=num_blocks,
            block_type=block_type,
            lookahead=False,
            **kwargs,
        )

    def call(self, inputs, targets, mfp_masks, training):
        """
        If add_masked_input, each point in a sequence will be concat. of
        [previous_emb, current_emb(masked)] instead of previous_emb
        """
        if training:
            h_masked, mask = self.encoder(inputs, training=training)
            h_tgt, _ = self.encoder(targets, training=training)
            B = tf.shape(h_masked)[0]

            if self.add_masked_input:
                h_masked = self.dimred(h_masked)
                h_tgt = self.dimred(h_tgt)

            # Prepend the beginning-of-seq embedding, and drop the last.
            bos = tf.tile(self.bos, (B, 1, 1))
            h = tf.concat([bos, h_tgt[:, 0:-1, :]], axis=1)
            if self.add_masked_input:
                h = tf.concat([h, h_masked], axis=-1)  # (B, 1, 2*D)

            h = self.blocks(h, mask, training=training)
            outputs = self.decoder(h, training=training)
        else:
            h_masked, mask = self.encoder(inputs, training=training)

            B = tf.shape(mask)[0]
            S = tf.shape(mask)[1]
            h = tf.tile(self.bos, (B, 1, 1))

            if self.add_masked_input:
                h_masked = self.dimred(h_masked)
                h = tf.concat([h, h_masked[:, 0:1]], axis=-1)  # (B, 1, 2*D)

            for t in range(S - 1):
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[
                        (h, tf.TensorShape([None, None, self.latent_dim])),
                    ]
                )

                next_elem = self._compute_next(h, mask[:, : t + 1], inputs, mfp_masks)
                if self.add_masked_input:
                    next_elem = tf.concat(
                        [self.dimred(next_elem), h_masked[:, t + 1 : t + 2]],
                        axis=-1,
                    )  # (B, 1, 2*D)
                h = tf.concat([h, next_elem], axis=1)  # (B, (t+1)+1, ?)

            # [<BOS>, T_{1}, ..., T_{t-1}] -> [T_{1}, ..., T_{t}]
            h = self.blocks(h, mask, training=training)
            outputs = self.decoder(h, training=training)
        return outputs


# class OneShotAutoReg(_AutoReg):
#     def __init__(
#         self,
#         input_columns: Dict,
#         num_blocks: int = 4,
#         block_type: str = "deepsvg",
#         context: Union[str, None] = None,
#         input_dtype: str = "set",
#         **kwargs,  # keys are latent_dim, dropout, l2
#     ):
#         super().__init__(
#             input_columns=input_columns,
#             num_blocks=num_blocks,
#             block_type=block_type,
#             context=context,
#             input_dtype=input_dtype,
#             **kwargs,
#         )
#         self.enc_blocks = Blocks(
#             num_blocks=num_blocks // 2,
#             block_type=block_type,
#             lookahead=True,
#             **kwargs,
#         )
#         self.blocks = Blocks(
#             num_blocks=num_blocks // 2,
#             block_type=block_type,
#             lookahead=False,
#             conditional=True,
#             **kwargs,
#         )
#         initializer = tf.random_normal_initializer()
#         self.cls = tf.Variable(
#             initial_value=initializer(
#                 shape=(1, 1, kwargs["latent_dim"]), dtype=tf.float32
#             ),
#             trainable=True,
#         )

#     def call(self, inputs, targets, mfp_masks, training):
#         training = False
#         if training:
#             h_masked, mask = self.encoder(inputs, training=training)
#             h_tgt, _ = self.encoder(targets, training=training)
#             B, S, _ = tf.shape(h_masked)

#             # add [CLS] token
#             new_mask = get_seq_mask(inputs["length"] + 1)
#             new_h_masked = tf.concat(
#                 [tf.tile(self.cls, (B, 1, 1)), h_masked], axis=1
#             )

#             z = self.enc_blocks(new_h_masked, new_mask, training=training)[:, 0]
#             if self.add_masked_input:
#                 h_masked = self.dimred(h_masked)
#                 h_tgt = self.dimred(h_tgt)

#             # Prepend the beginning-of-seq embedding, and drop the last.
#             bos = tf.tile(self.bos, (B, 1, 1))
#             h = tf.concat([bos, h_tgt[:, 1:, :]], axis=1)
#             if self.add_masked_input:
#                 h = tf.concat([h, h_masked], axis=-1)  # (B, 1, 2*D)

#             h = self.blocks((h, z), mask, training=training)
#             outputs = self.decoder(h, training=training)
#         else:
#             # add [CLS] token
#             h_masked, mask = self.encoder(inputs, training=training)
#             B = tf.shape(h_masked)[0]
#             new_mask = get_seq_mask(inputs["length"] + 1)
#             new_h_masked = tf.concat(
#                 [tf.tile(self.cls, (B, 1, 1)), h_masked], axis=1
#             )

#             z = self.enc_blocks(new_h_masked, new_mask, training=training)[:, 0]

#             # make sure to ignore first element (only for z)

#             bos = tf.tile(self.bos, (B, 1, 1))
#             if self.add_masked_input:
#                 h_masked = self.dimred(h_masked)
#                 h = tf.concat([bos, h_masked[:, 0:1]], axis=-1)  # (B, 1, 2*D)
#             else:
#                 h = bos

#             S = tf.shape(mask)[1]
#             for t in range(S - 1):
#                 tf.autograph.experimental.set_loop_options(
#                     shape_invariants=[
#                         (h, tf.TensorShape([None, None, self.latent_dim])),
#                     ]
#                 )
#                 next_elem = self._compute_next(
#                     (h, z), mask[:, : t + 1], inputs, mfp_masks
#                 )
#                 if self.add_masked_input:
#                     next_elem = tf.concat(
#                         [self.dimred(next_elem), h_masked[:, t + 1 : t + 2]],
#                         axis=-1,
#                     )  # (B, 1, 2*D)
#                 h = tf.concat([h, next_elem], axis=1)  # (B, (t+1)+1, 2*D)

#             # [<BOS>, T_{1}, ..., T_{t-1}] -> [T_{1}, ..., T_{t}]
#             h = self.blocks((h, z), mask, training=training)
#             outputs = self.decoder(h, training=training)

#         return outputs


class BART(_AutoReg):
    def __init__(
        self,
        input_columns: Dict,
        num_blocks: int = 4,
        block_type: str = "deepsvg",
        context: Union[str, None] = None,
        input_dtype: str = "set",
        **kwargs,  # keys are latent_dim, dropout, l2
    ):
        assert input_dtype == "shuffled_set"
        super().__init__(
            input_columns=input_columns,
            num_blocks=num_blocks,
            block_type=block_type,
            context=context,
            input_dtype=input_dtype,
            **kwargs,
        )
        self.enc_blocks = Blocks(
            num_blocks=num_blocks // 2,
            block_type=block_type,
            lookahead=True,
            **kwargs,
        )
        self.blocks = CrossBlocks(
            num_blocks=num_blocks // 2,
            block_type=f"{block_type}_cross",
            lookahead=False,
            **kwargs,
        )
        # initializer = tf.random_normal_initializer()

    def call(self, inputs, targets, mfp_masks, training):
        if training:
            h_masked, mask = self.encoder(inputs, training=training)
            h_tgt, _ = self.encoder(targets, training=training)

            B, _, _ = tf.shape(h_masked)
            z = self.enc_blocks(h_masked, mask, training=training)
            tgt_mask = mask

            # Prepend the beginning-of-seq embedding, and drop the last.
            bos = tf.tile(self.bos, (B, 1, 1))
            h = tf.concat([bos, h_tgt[:, :-1, :]], axis=1)
            h = self.blocks((h, z), (tgt_mask, mask), training=training)
            outputs = self.decoder(h, training=training)
        else:
            # add [CLS] token
            h_masked, mask = self.encoder(inputs, training=training)
            B, S, _ = tf.shape(h_masked)
            z = self.enc_blocks(h_masked, mask, training=training)

            h = tf.tile(self.bos, (B, 1, 1))
            for t in range(S - 1):
                tf.autograph.experimental.set_loop_options(
                    shape_invariants=[
                        (h, tf.TensorShape([None, None, self.latent_dim])),
                    ]
                )
                next_elem = self._compute_next(
                    (h, z), (mask[:, : t + 1], mask), inputs, mfp_masks
                )
                h = tf.concat([h, next_elem], axis=1)  # (B, (t+1)+1, 2*D)

            # [<BOS>, T_{1}, ..., T_{t-1}] -> [T_{1}, ..., T_{t}]
            h = self.blocks((h, z), (mask, mask), training=training)
            outputs = self.decoder(h, training=training)

        return outputs
