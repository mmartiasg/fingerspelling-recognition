from src.data_utils.dataset import char_to_num, num_to_char
import tensorflow as tf
from src.data_utils.dataset import VOCAB_SIZE, start_token_idx, end_token_idx, FEATURE_COLUMNS, pre_process


class TFLiteModel(tf.Module):
    def __init__(self, model):
        super(TFLiteModel, self).__init__()
        self.target_start_token_idx = start_token_idx
        self.target_end_token_idx = end_token_idx
        # Load the feature generation and main models
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, FEATURE_COLUMNS.shape[0]], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs, training=False):
        # Preprocess Data
        x = tf.cast(inputs, tf.float32)

        x = x[None]

        x = tf.cond(tf.shape(x)[1] == 0, lambda: tf.zeros((1, 1, FEATURE_COLUMNS.shape[0])), lambda: tf.identity(x))

        x = x[0]

        x = pre_process(x)
        #shape after [MAX_LENGHT_SOURCE, FEATURE_SIZE/2(52)]

        x = x[None]

        x = self.model.generate(x)

        x = x[0]

        # Remove start, end and unknown tokens from the generated sentence
        # Remove the end token
        idx = tf.argmax(tf.cast(tf.equal(x, self.target_end_token_idx), tf.int32))
        # Remove unknown tokens
        idx = tf.where(tf.math.less(idx, 1), tf.constant(2, dtype=tf.int64), idx)
        # Remove start token
        x = x[1:idx]

        # 59 is the number of characters after removing the start, end and unknown tokens
        x = tf.one_hot(x, 59)
        return {"outputs": x}

