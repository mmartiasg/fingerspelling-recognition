import tensorflow as tf
import string
import re


@tf.keras.utils.register_keras_serializable(name="source_standarizer")
class DefaultSourceEstandarizer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
    
    def call(self, inputs, *args, **kwargs):
        strip_chars = string.punctuation
        lowercase = tf.strings.lower(inputs)
        return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")

    def get_config(self):
        return super().get_config()
    
@tf.keras.utils.register_keras_serializable(name="target_standarizer")
class DefaultTargetEstandarizer(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
    
    def call(self, inputs, *args, **kwargs):
        strip_chars = string.punctuation
        strip_chars = strip_chars.replace("[", "")
        strip_chars = strip_chars.replace("]", "")

        lowercase = tf.strings.lower(inputs)
        return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")

    def get_config(self):
        return super().get_config()
