import tensorflow as tf
from mfp.models.architecture.utils import make_dense_options


class PositionEmbedding(tf.keras.layers.Layer):
    """Returns positional const embeddings."""

    def __init__(
        self,
        output_dim,
        maxlen,
        dropout=0.1,
        emb_options=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embeddings = tf.keras.layers.Embedding(
            maxlen + 1,
            output_dim,
            **(emb_options or {}),
        )
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        B = tf.shape(inputs)[0]
        positions = tf.range(tf.shape(inputs)[1])
        embeddings = self.embeddings(positions[tf.newaxis, :])
        embeddings = tf.tile(embeddings, [B, 1, 1])
        embeddings = self.dropout(embeddings, training=training)
        return embeddings


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    Taken from
    https://keras.io/examples/nlp/text_classification_with_transformer/

    :param emb_size: Size of the embedding.
    :param num_heads: Number of heads.
    :param lookahead: Allow attention to future tokens.
    """

    def __init__(self, emb_size, num_heads=8, lookahead=True, **dense_options):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.lookahead = lookahead
        if emb_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {emb_size} should be divisible by "
                f"number of heads = {num_heads}."
            )
        self.projection_dim = emb_size // num_heads
        self.dense_query = tf.keras.layers.Dense(emb_size, **dense_options)
        self.dense_key = tf.keras.layers.Dense(emb_size, **dense_options)
        self.dense_value = tf.keras.layers.Dense(emb_size, **dense_options)
        self.combine_heads = tf.keras.layers.Dense(emb_size, **dense_options)
        self.supports_masking = True

    def attention(self, query, key, value, mask=None):
        score = tf.matmul(query, key, transpose_b=True)  # (B, H, S, projection_dim)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)  # (B, H, S, S)
        scaled_score = score / tf.math.sqrt(dim_key)  # (B, H, S, S)
        if mask is not None:
            # padding mask (B, 1, 1, S)
            mask = tf.cast(mask, tf.float32)[:, tf.newaxis, tf.newaxis, :]
            if not self.lookahead:
                size = tf.shape(mask)[-1]
                mask *= tf.linalg.band_part(tf.ones((size, size)), -1, 0)[
                    tf.newaxis, tf.newaxis, :, :
                ]
            # Force large negative for masks: (B, H, S, S).
            scaled_score += -1e9 * (1.0 - mask)
        weights = tf.nn.softmax(scaled_score, axis=-1)  # (B, H, S, S)
        output = tf.matmul(weights, value)  # (B, H, S, projection_dim)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        # inputs.shape = [B, S, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.dense_query(inputs)  # (B, S, emb_size)
        query = self.separate_heads(query, batch_size)  # (B, H, S, projection_dim)
        key = self.dense_key(inputs)  # (B, S, emb_size)
        key = self.separate_heads(key, batch_size)  # (B, H, S, projection_dim)
        value = self.dense_value(inputs)  # (B, S, emb_size)
        value = self.separate_heads(value, batch_size)  # (B, H, S, projection_dim)
        attention, _ = self.attention(query, key, value, mask)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (B, S, H, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.emb_size)
        )  # (B, S, emb_size)
        output = self.combine_heads(concat_attention)  # (B, S, emb_size)
        return output


class MultiHeadCrossAttention(MultiHeadSelfAttention):
    """
    Taken from
    https://keras.io/examples/nlp/text_classification_with_transformer/

    :param emb_size: Size of the embedding.
    :param num_heads: Number of heads.
    :param lookahead: Allow attention to future tokens.
    """

    def __init__(self, emb_size, num_heads=8, lookahead=True, **dense_options):
        super().__init__(emb_size, num_heads, lookahead, **dense_options)
        assert self.lookahead

    def call(self, inputs, mask=None):
        # inputs.shape = [B, S, embedding_dim]
        x, z = inputs
        # tgt_mask, memory_mask = masks

        batch_size = tf.shape(x)[0]
        query = self.dense_query(x)  # (B, S, emb_size)
        query = self.separate_heads(query, batch_size)  # (B, H, S, projection_dim)
        key = self.dense_key(z)  # (B, S, emb_size)
        key = self.separate_heads(key, batch_size)  # (B, H, S, projection_dim)
        value = self.dense_value(z)  # (B, S, emb_size)
        value = self.separate_heads(value, batch_size)  # (B, H, S, projection_dim)

        attention, _ = self.attention(query, key, value, mask)

        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (B, S, H, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.emb_size)
        )  # (B, S, emb_size)
        output = self.combine_heads(concat_attention)  # (B, S, emb_size)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with optional global conditional."""

    def __init__(
        self,
        emb_size=64,
        num_heads=8,
        ff_dim=None,
        dropout=0.1,
        conditional=None,
        pooling=None,
        dense_options=None,
        lookahead=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        dense_options = dense_options or {}
        self.attn = MultiHeadSelfAttention(
            emb_size, num_heads, lookahead=lookahead, **dense_options
        )
        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    ff_dim or (2 * emb_size),
                    activation="relu",
                    **dense_options,
                ),
                # tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(emb_size, **dense_options),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True
        self.conditional = None
        if conditional:
            self.norm3 = tf.keras.layers.LayerNormalization()
            self.conditional = tf.keras.layers.Dense(emb_size, **dense_options)

        self.pooling = None
        if pooling:
            self.relu = tf.keras.layers.Activation("relu")
            self.pooling = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, training=False, mask=None):
        if self.conditional is not None:
            x = inputs[0]
            z = inputs[1]
        else:
            x = inputs
        y = self.attn(x, mask=mask)
        y = self.dropout1(y, training=training)
        x = self.norm1(x + y, training=training)
        if self.conditional is not None:
            z = tf.expand_dims(self.conditional(z), 1)
            x = self.norm3(x + z, training=training)
        y = self.mlp(x)
        y = self.dropout2(y, training=training)
        x = self.norm2(x + y, training=training)
        if self.pooling is not None:
            x = self.relu(x)
            return self.pooling(x, mask=mask)
        return x


class DeepSVGBlock(TransformerBlock):
    """DeepSVG transformer block."""

    def call(self, inputs, training=False, mask=None):
        if self.conditional is not None:
            x, z = inputs
        else:
            x = inputs
        y = self.norm1(x, training=training)
        y = self.attn(y, mask=mask)
        y = self.dropout1(y, training=training)
        x += y
        if self.conditional is not None:
            x += tf.expand_dims(self.conditional(z), 1)
        y = self.norm2(x, training=training)
        y = self.mlp(y)
        y = self.dropout2(y, training=training)
        x = x + y
        if self.pooling is not None:
            x = self.relu(x)
            return self.pooling(x, mask=mask)
        return x


def get_seq_block(layer_type):
    return {
        "transformer": TransformerBlock,
        "deepsvg": DeepSVGBlock,
    }[layer_type]


class Blocks(tf.keras.layers.Layer):
    """
    Stack of transformer layers implementation.
    """

    def __init__(
        self,
        latent_dim=128,
        num_blocks=1,
        block_type="deepsvg",
        conditional=None,
        lookahead=True,  # False if using auto-regressive models
        dropout=0.1,
        l2=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq2seq = {}
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.conditional = conditional

        layer_fn = get_seq_block(block_type)
        for i in range(num_blocks):
            self.seq2seq["seq2seq_%d" % i] = layer_fn(
                latent_dim,
                dropout=dropout,
                conditional=conditional,
                dense_options=make_dense_options(l2),
                lookahead=lookahead,
                name="seq2seq_%d" % i,
            )

    def __call__(self, seq, mask, training=False):
        if self.conditional:
            seq, z = seq[0], seq[1]
            for layer in self.seq2seq.values():
                seq = layer((seq, z), training=training, mask=mask)
        else:
            for layer in self.seq2seq.values():
                seq = layer(seq, training=training, mask=mask)
        return seq


class CrossBlocks(Blocks):
    """
    Stack of transformer layers implementation.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def __call__(self, inputs, masks, training=False):
        tgt, memory = inputs
        for layer in self.seq2seq.values():
            tgt = layer((tgt, memory), training=training, masks=masks)
        return tgt
