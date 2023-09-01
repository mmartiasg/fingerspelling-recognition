import tensorflow as tf
import math
import numpy as np


class DataLoaderAbstract:
    def __init__(self, path) -> None:
        self.path = path
        self.eng_ita = []

    def read_file(self):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class Source2Target(DataLoaderAbstract):

    def __init__(self, path,
                 max_tokens_source,
                 max_tokens_target,
                 seq_length_source,
                 seq_length_target,
                 source_standarizer,
                 target_standarizer):

        super().__init__(path)

        self.source_standarizer = source_standarizer
        self.target_standarizer = target_standarizer

        self.source_tokenizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens_source,
                                                                  output_sequence_length=seq_length_source,
                                                                  output_mode="int",
                                                                  standardize=None,
                                                                  name="source_vectorizer")

        # This sequence is one step ahead to the source
        self.target_tokenizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens_target,
                                                                  output_sequence_length=seq_length_target + 1,
                                                                  output_mode="int",
                                                                  standardize=None,
                                                                  name="target_vectorizer")

        self.train = None
        self.val = None
        self.test = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def read_file(self):
        self.eng_ita = []
        with open(self.path) as f:
            for line in f:
                eng, ita, _ = line.split(sep="\t", maxsplit=2)
                # end and start tokens
                self.eng_ita.append((eng, "[start] " + ita + " [end]"))

    def split_data(self):
        n = len(self.eng_ita)
        print(f"Total records: {n}")

        n_train = math.ceil(n * 0.75)
        n_val = math.ceil((n * 0.15))

        np.random.shuffle(self.eng_ita)

        self.train = self.eng_ita[:n_train]
        self.val = self.eng_ita[n_train:n_train + n_val]
        self.test = self.eng_ita[n_train + n_val:]

    def tokenize_data(self):
        self.source_tokenizer.adapt([pair[0] for pair in self.train])
        self.target_tokenizer.adapt([pair[1] for pair in self.train])

    def format_dataset(self, source_sequence, target_sequence):
        # "I need to do it separated because of the stupid TextVectorizer not able to save the stupid function"

        source_sequence = self.source_standarizer(source_sequence)
        target_sequence = self.target_standarizer(target_sequence)

        source_vect = self.source_tokenizer(source_sequence)
        target_vect = self.target_tokenizer(target_sequence)

        return ({
                    "source": source_vect,
                    # the -1 is to remove the [end] token
                    "target": target_vect[:, :-1]
                }, target_vect[:, 1:])

    def __call__(self, batch_size, **kwargs):
        print("Reading file... (1/4)")
        self.read_file()
        print("Splitting data... (2/4)")
        self.split_data()
        print("Building vocabulary... (3/4")
        self.tokenize_data()

        print("Building tensorflow datasets... (4/4)")

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            ([pair[0] for pair in self.train], [pair[1] for pair in self.train])).batch(batch_size).map(
            self.format_dataset, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size).prefetch(tf.data.AUTOTUNE)
        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            ([pair[0] for pair in self.val], [pair[1] for pair in self.val])).batch(math.ceil(batch_size/2)).map(
            self.format_dataset, num_parallel_calls=tf.data.AUTOTUNE).shuffle(batch_size).prefetch(tf.data.AUTOTUNE)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            ([pair[0] for pair in self.test], [pair[1] for pair in self.test])).batch(batch_size).map(
            self.format_dataset, num_parallel_calls=tf.data.AUTOTUNE).shuffle(math.ceil(batch_size/2)).prefetch(tf.data.AUTOTUNE)
        
        print("Done!")

        return self.train_dataset, self.val_dataset, self.test_dataset

    def inspect_from_dataset(self, dataset, max_samples=10):
        """
        :param dataset: returns a tuple the first one has a dictionary with 2 key source and target
        :param max_samples: the max amount of samples that will be shown
        :return: returns the decoded phrase from the dataset
        """

        source_token_2_word = dict(
            [(index, word) for index, word in enumerate(self.source_tokenizer.get_vocabulary())])
        target_token_2_word = dict(
            [(index, word) for index, word in enumerate(self.target_tokenizer.get_vocabulary())])

        i = 0
        for source_target, target_1_step_ahead in dataset:
            print("======================================================")
            print("source: ", " ".join([source_token_2_word[token] for token in source_target["source"][0].numpy()]))
            print("target: ", " ".join([target_token_2_word[token] for token in source_target["target"][0].numpy()]))
            if i > max_samples:
                break
            i += 1

    def from_source_index_to_words(self, sentences):
        source_token_2_word = dict(
            [(index, word) for index, word in enumerate(self.source_tokenizer.get_vocabulary())])

        for sentence in sentences:
            print(" ".join([source_token_2_word[token] for token in sentence]))

    def from_target_index_to_words(self, sentences):
        target_token_2_word = dict(
            [(index, word) for index, word in enumerate(self.target_tokenizer.get_vocabulary())])

        for sentence in sentences:
            print(" ".join([target_token_2_word[token] for token in sentence]))


class SampledTestatN(Source2Target):
    def read_file(self):
        super(SampledTestatN, self).read_file()
        self.eng_ita = self.eng_ita[:1000]