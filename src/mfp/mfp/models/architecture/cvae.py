from typing import Dict, Tuple

import tensorflow as tf
from mfp.models.architecture.utils import make_dense_options


class Head(tf.keras.layers.Layer):
    def __init__(
        self,
        latent_dim: int = 32,
        kl: float = 1.0,
        l2: float = None,
        compute_kl: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.fc_mean = tf.keras.layers.Dense(
            units=latent_dim,
            **make_dense_options(l2),
        )
        self.fc_log_sigma = tf.keras.layers.Dense(
            units=latent_dim,
            **make_dense_options(l2),
        )
        self.kl = kl
        self.compute_kl = compute_kl

    def reparameterize(self, z_mean: tf.Tensor, z_log_sigma: tf.Tensor):
        epsilon = tf.random.normal(shape=tf.shape(z_log_sigma))
        return z_mean + tf.exp(0.5 * z_log_sigma) * epsilon

    def call(self, h: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
        z_mean = self.fc_mean(h)
        z_log_sigma = self.fc_log_sigma(h)
        if training:
            z = self.reparameterize(z_mean, z_log_sigma)
        else:
            z = z_mean

        if training and self.compute_kl:
            # Compute KL divergence to normal distribution.
            kl_div = -0.5 * tf.reduce_mean(
                1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma)
            )
            self.add_loss(self.kl * kl_div)
            self.add_metric(kl_div, name="kl_divergence")

        return {"z": z, "z_mean": z_mean, "z_log_sigma": z_log_sigma}


class Prior(tf.keras.layers.Layer):
    def __init__(self, l2: float = None):
        super().__init__()
        latent_dim = 32
        self.fc = tf.keras.layers.Dense(
            units=latent_dim,
            activation="relu",
            **make_dense_options(l2),
        )
        self.head = Head(l2=l2)

    def call(self, h: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
        return self.head(self.fc(h), training=training)


class MAPrior(tf.keras.layers.Layer):
    """
    It has separate models for each attribute
    """

    def __init__(
        self,
        input_columns: Dict,
        l2: float = None,
    ):
        super().__init__()
        self.input_columns = input_columns

        self.layers = {}
        for key in input_columns:
            self.layers[key] = Prior(l2=l2)

    def call(
        self,
        context: tf.Tensor,
        training: bool = False,
    ) -> Dict[str, Dict[str, tf.Tensor]]:
        outputs = {}
        for key, layer in self.layers.items():
            outputs[key] = layer(context, training=training)
        return outputs


class VAEEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        l2: float = None,
    ):
        super().__init__()
        dim_in, dim_out = 128, 32
        self.fc1 = tf.keras.layers.Dense(
            units=dim_in,
            **make_dense_options(l2),
        )
        self.fc2 = tf.keras.layers.Dense(
            units=dim_out,
            activation="relu",
            **make_dense_options(l2),
        )
        self.head = Head(l2=l2)

    def call(
        self, hidden: tf.Tensor, context: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        h = self.fc1(hidden)
        h = tf.concat([h, context], axis=-1)
        h = self.fc2(h)
        return self.head(h, training=training)


class MACVAEEncoder(tf.keras.layers.Layer):
    """
    It has separate models for each attribute
    """

    def __init__(
        self,
        input_columns: Dict,
        l2: float = None,
    ):
        super().__init__()
        self.input_columns = input_columns

        self.layers = {}
        for key in input_columns:
            self.layers[key] = VAEEncoder(l2=l2)

    def call(
        self,
        h_gts: Dict[str, tf.Tensor],
        context: tf.Tensor,
        training: bool = False,
    ) -> Dict[str, Dict[str, tf.Tensor]]:
        outputs = {}
        for key, layer in self.layers.items():
            outputs[key] = layer(h_gts[key], context, training=training)
        return outputs


class VAEDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        l2: float = None,
    ):
        super().__init__()
        latent_dim, dim_out = 128, 64
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    units=latent_dim,
                    activation="relu",
                    **make_dense_options(l2),
                ),
                tf.keras.layers.Dense(
                    units=dim_out,
                    activation="relu",
                    **make_dense_options(l2),
                ),
            ]
        )

    def call(
        self, z: tf.Tensor, context: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        h = tf.concat([z, context], axis=-1)
        return self.model(h)


class MACVAEDecoder(tf.keras.layers.Layer):
    """
    It has separate models for each attribute
    """

    def __init__(
        self,
        input_columns: Dict,
        l2: float = None,
    ):
        super().__init__()
        self.input_columns = input_columns
        self.layers = {}
        for key in input_columns:
            self.layers[key] = VAEDecoder(l2=l2)

    def call(
        self,
        zs: Dict[str, tf.Tensor],
        context: tf.Tensor,
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        outputs = {}
        for key, layer in self.layers.items():
            outputs[key] = layer(zs[key], context, training=training)
        return outputs
