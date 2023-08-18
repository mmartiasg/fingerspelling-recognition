import tensorflow as tf
from Transformers.Seq2SeqModels import TransformerBasedTranslator
from Transformers.DataLoader import Source2Target, SampledTestatN
from Transformers.preprocess import DefaultSourceEstandarizer, DefaultTargetEstandarizer


class TransformerTranslatorBuilder:
    def __init__(self, path, num_heads=3, dim_embeddings=256, dim_dense=1024, source_max_tokens=15000,
                 target_max_tokens=30000, source_seq_length=20, target_seq_length=20, dropout_rate=0.4,
                 source_standarizer=DefaultSourceEstandarizer,
                 target_standarizer=DefaultTargetEstandarizer):
        
        self.path = path
        self.num_heads = num_heads
        self.dim_embeddings = dim_embeddings
        self.dim_dense = dim_dense
        self.source_max_tokens = source_max_tokens
        self.target_max_tokens = target_max_tokens
        self.source_seq_length = source_seq_length
        self.target_seq_length = target_seq_length
        self.dropout_rate = dropout_rate
        self.source_standarizer = source_standarizer
        self.target_standarizer = target_standarizer

        # I cant pass a preprocessing function until the stupid people fix the stupid TextVectorizer layer to fucking works
        self.loader = Source2Target(path=self.path,
                                    max_tokens_source=self.source_max_tokens,
                                    max_tokens_target=self.target_max_tokens,
                                    seq_length_source=source_seq_length,
                                    seq_length_target=target_seq_length,
                                    source_standarizer=self.source_standarizer,
                                    target_standarizer=self.target_standarizer)
        self.model = None

    def build(self, learning_rate=1e-3, *args, **kwargs):
        translator_model = TransformerBasedTranslator(num_heads=self.num_heads,
                                                      dim_embeddings=self.dim_embeddings,
                                                      dim_dense=self.dim_dense,
                                                      source_max_seq_length=self.source_seq_length,
                                                      target_max_seq_length=self.target_seq_length,
                                                      source_max_tokens=self.source_max_tokens,
                                                      target_max_tokens=self.target_max_tokens,
                                                      dropout_rate=self.dropout_rate)

        translator_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                 optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
                                 metrics=["accuracy"])

        self.model = translator_model

    def __call__(self, epochs=50, batch_size=512, output_model_path=None, *args, **kwargs):
        self.output_model_path = output_model_path
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=5)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="temp_best_model",
                                                        save_best_only=True,
                                                        save_weights_only=False)

        self.build()

        self.loader(batch_size=batch_size)

        self.model.fit(self.loader.train_dataset,
                       validation_data=self.loader.val_dataset,
                       epochs=epochs,
                       callbacks=[early_stop, checkpoint])

        self.model = tf.keras.models.load_model("temp_best_model")

        self.model.evaluate(self.loader.test_dataset)

        if self.output_model_path:
            x_input1 = tf.keras.Input(shape=(), dtype="string", name="source")
            x_input2 = tf.keras.Input(shape=(), dtype="string", name="target")

            standarized_x1 = self.source_standarizer(x_input1)
            standarized_x2 = self.target_standarizer(x_input2)

            source_vec = self.loader.source_tokenizer(standarized_x1)
            target_vec = self.loader.source_tokenizer(standarized_x2)

            x = self.model({"source": source_vec, "target": target_vec})

            new_model = tf.keras.Model(inputs=[x_input1, x_input2], outputs=x)

            new_model.save(self.output_model_path+"_"+"FULL")

            self.full_model = new_model


