from typing import Dict

import tensorflow as tf
import tensorflow_probability as tfp
from mfp.data.spec import get_valid_input_columns
from mfp.models.architecture.cvae import MACVAEDecoder, MACVAEEncoder, MAPrior
from mfp.models.architecture.decoder import Decoder
from mfp.models.architecture.encoder import Encoder
from mfp.models.architecture.transformer import Blocks

MND = tfp.distributions.MultivariateNormalDiag


class LayoutVAE(tf.keras.layers.Layer):
    def __init__(
        self,
        input_columns: Dict,
        num_blocks: int = 4,
        block_type: str = "deepsvg",
        input_dtype: str = "set",
        kl: float = 1e-0,
        **kwargs,  # keys are latent_dim, dropout, l2
    ):
        super().__init__()
        l2 = kwargs.get("l2", None)
        self.kl = kl  # kl search range: [1e0, 1e1, 1e2]
        self.arch_type = "autoreg"
        self.input_columns = input_columns
        self.valid_input_columns = get_valid_input_columns(input_columns, False)
        self.lookahead = False
        self.encoder = Encoder(input_columns, **kwargs)
        self.decoder = Decoder(input_columns, detachment="none", **kwargs)

        # separately encode each attribute
        self.encoder_gt = Encoder(input_columns, fusion="none", **kwargs)
        self.encoder_cvae = MACVAEEncoder(self.valid_input_columns, l2=l2)
        self.decoder_cvae = MACVAEDecoder(self.valid_input_columns, l2=l2)
        self.prior = MAPrior(self.valid_input_columns, l2=l2)

        self.blocks = Blocks(
            num_blocks=num_blocks,
            block_type=block_type,
            **kwargs,
        )

    def call(
        self,
        inputs: Dict,
        targets: Dict,
        mfp_masks: Dict,
        training: bool,
        add_masked_input: bool = True,
    ):
        S = tf.shape(inputs["left"])[1]
        h_inputs, mask = self.encoder(inputs, training=training)
        if training:
            h_targets, _ = self.encoder(targets, training=training)

        stack = {k: None for k in self.valid_input_columns}
        buffer = {}

        h_pred = None  # should be the result of Encoder(x)
        for i in range(S):
            if i == 0:
                h_fused = h_inputs
            else:
                # use GT in 0~i-1 th, use masked inputs in i~S-1 th in training
                h_fused = h_targets[:, 0:i] if training else h_pred[:, 0:i]
                h_fused = tf.concat([h_fused, h_inputs[:, i:]], axis=1)

            c = self.blocks(h_fused, mask, training=training)[:, i : i + 1]
            if training:
                h, _ = self.encoder_gt(targets, training=training)
                h = {k: v[:, i : i + 1] for (k, v) in h.items()}
                zs = self.encoder_cvae(h, c, training=training)
                zs_p = self.prior(c, training=training)
            else:
                zs = self.prior(c, training=training)
            z = {k: v["z"] for (k, v) in zs.items()}

            for (k, v) in self.decoder_cvae(z, c, training=training).items():
                if i == 0:
                    stack[k] = v
                    if training:
                        for name in ["mean", "log_sigma"]:
                            buffer[f"{k}_{name}"] = zs[k][f"z_{name}"]
                            buffer[f"{k}_{name}_p"] = zs_p[k][f"z_{name}"]
                else:
                    stack[k] = tf.concat([stack[k], v], axis=1)
                    if training:
                        for name in ["mean", "log_sigma"]:
                            buffer[f"{k}_{name}"] = tf.concat(
                                [buffer[f"{k}_{name}"], zs[k][f"z_{name}"]],
                                axis=1,
                            )
                            buffer[f"{k}_{name}_p"] = tf.concat(
                                [buffer[f"{k}_{name}_p"], zs_p[k][f"z_{name}"]],
                                axis=1,
                            )

            if not training:
                elem = self._compute_next(i, stack, mask, inputs, mfp_masks)
                h_pred = (
                    tf.concat([h_pred, elem], axis=1) if tf.is_tensor(h_pred) else elem
                )

        if training:
            self._compute_kl(buffer, mfp_masks)

        outputs = self.decoder(stack, training=training)
        return outputs

    def _compute_kl(self, x: Dict[str, tf.Tensor], mfp_masks: Dict[str, tf.Tensor]):
        loss_total = 0.0
        for k in self.valid_input_columns:
            dist = MND(x[f"{k}_mean"], tf.exp(0.5 * x[f"{k}_log_sigma"]))
            dist_p = MND(x[f"{k}_mean_p"], tf.exp(0.5 * x[f"{k}_log_sigma_p"]))
            loss = dist.kl_divergence(dist_p)
            weight = tf.cast(mfp_masks[k], tf.float32)
            loss = loss * self.kl * weight
            loss = tf.reduce_mean(loss)
            self.add_metric(loss, name=k + "_loss")
            loss_total += loss

        self.add_metric(loss_total, name="kl_loss_total")
        self.add_loss(loss_total)

    def _compute_next(
        self,
        i: int,
        h: Dict[str, tf.Tensor],
        mask: tf.Tensor,
        inputs: Dict[str, tf.Tensor],
        mfp_masks: Dict[str, tf.Tensor],
    ) -> tf.Tensor:
        B = tf.shape(mask)[0]
        h_i = {}
        for (k, v) in h.items():
            h_i[k] = v[:, i : i + 1]
        outputs_i = self.decoder(h_i, training=False)

        new_inputs = {}
        for key, column in self.input_columns.items():
            if column["is_sequence"] and not column.get("demo_only", False):
                if column["type"] == "categorical":
                    outputs_i[key] = tf.argmax(
                        outputs_i[key], axis=-1, output_type=tf.int32
                    )
                new_inputs[key] = tf.where(
                    mfp_masks[key][:, i : i + 1, tf.newaxis],
                    outputs_i[key],
                    inputs[key][:, i : i + 1],
                )
        new_inputs["length"] = tf.zeros((B, 1))
        next_elem, _ = self.encoder(new_inputs)  # (B, 1, D)
        return next_elem
