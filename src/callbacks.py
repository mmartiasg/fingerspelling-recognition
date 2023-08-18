import tensorflow as tf


def get_predefine_callbacks(model_name):
    assert model_name is not None

    early_stop = tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss")
    save_best_model = tf.keras.callbacks.ModelCheckpoint(filepath=f"../best_model/prototype/{model_name}", save_best_only=True, save_weights_only=True)

    return [early_stop, save_best_model]
