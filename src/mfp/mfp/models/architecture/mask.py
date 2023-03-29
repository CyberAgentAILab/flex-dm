import tensorflow as tf


class Unmask(tf.keras.layers.Layer):
    """Layer to stop mask propagation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        if hasattr(inputs, "_keras_mask"):
            delattr(inputs, "_keras_mask")
        return inputs


# @tf.function(experimental_relax_shapes=True)
def get_seq_mask(inputs, from_logits: bool = False, maxlen: int = None):
    """Generate mask from length."""
    if from_logits:
        length = tf.reshape(tf.argmax(inputs, axis=-1), (-1,))
    else:
        length = tf.reshape(inputs, (-1,))

    # Fix zero-based index.
    length += 1

    seq_mask = tf.sequence_mask(length, maxlen=maxlen)
    tf.debugging.assert_rank(seq_mask, 2)
    return seq_mask
