import tensorflow as tf
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

import pickle
import json
import os


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, data_dir, word2vec_path, max_sentence_length):
        self.data_dir = data_dir
        self.word2vec_path = word2vec_path
        self.max_sentence_length = max_sentence_length
        self.labels = set()
        self.num_class = 0
        self.label_map = dict()
        self.tokenizer = None

    def _build_vocabulary(self, train_texts, oov_token='UNK', filters='', lower=True):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            oov_token=oov_token,
            filters=filters,
            lower=lower)
        self.tokenizer.fit_on_texts(train_texts)
        # add PAD
        self.tokenizer.word_index['<PAD>'] = 0
        self.tokenizer.index_word[0] = '<PAD>'
        self.tokenizer.word_counts['<PAD>'] = 0
        self.tokenizer.word_docs['<PAD>'] = 0

        # get word embedding
        self.dump_word_embedding(self.tokenizer.word_index)
        print("Build the vocabulary done")

    def build_label_map(self, train_labels_name, valid_labels_name, test_labels_name):
        train_labels_path = os.path.join(self.data_dir, train_labels_name)
        valid_labels_path = os.path.join(self.data_dir, valid_labels_name)
        test_labels_path = os.path.join(self.data_dir, test_labels_name)
        with open(train_labels_path, 'rt') as fin:
            train_labels = json.load(fin)
        with open(valid_labels_path, 'rt') as fin:
            valid_labels = json.load(fin)
        with open(test_labels_path, 'rt') as fin:
            test_labels = json.load(fin)
        for train_label in train_labels+valid_labels+test_labels:
            self.labels = self.labels.union(train_label)
        self.num_class = len(self.labels)
        self.label_map = dict(zip(self.labels, range(self.num_class)))

    def _transform_label(self, label):
        label_id = np.zeros(self.num_class, dtype=np.int64)
        for item in label:
            if item in self.label_map:
                label_id[self.label_map[item]] = 1
            else:
                return None
        return label_id

    def dump_train_features(self, text_name, label_name):
        text_path = os.path.join(self.data_dir, text_name)
        label_path = os.path.join(self.data_dir, label_name)

        texts, labels = self._get_data_from_json(text_path, label_path)
        self._build_vocabulary(texts)
        # self._build_label_map(labels)
        texts_ids = self.tokenizer.texts_to_sequences(texts)
        max_sentence_length = max(len(x) for x in texts_ids)
        if max_sentence_length < self.max_sentence_length:
            self.max_sentence_length = max_sentence_length
        print("max sentence length is {}".format(self.max_sentence_length))
        # padding
        texts_ids = tf.keras.preprocessing.sequence.pad_sequences(texts_ids,
                                                                  maxlen=self.max_sentence_length,
                                                                  padding='post',
                                                                  truncating='post')
        labels_ids = np.array([self._transform_label(label) for label in labels])
        with open(os.path.join(self.data_dir, 'train_texts_ids.dat'), 'wb') as fout:
            pickle.dump(texts_ids, fout)
        with open(os.path.join(self.data_dir, 'train_labels_ids.dat'), 'wb') as fout:
            pickle.dump(labels_ids, fout)
        print("Train Data Done {}".format(len(labels_ids)))

    def dump_eval_features(self, text_name, label_name):
        text_path = os.path.join(self.data_dir, text_name)
        label_path = os.path.join(self.data_dir, label_name)
        texts, labels = self._get_data_from_json(text_path, label_path)
        texts_ids = self.tokenizer.texts_to_sequences(texts)
        # padding
        texts_ids = tf.keras.preprocessing.sequence.pad_sequences(texts_ids,
                                                                  maxlen=self.max_sentence_length,
                                                                  padding='post',
                                                                  truncating='post')
        labels_ids = np.array([self._transform_label(label) for label in labels])
        # texts_ids, labels_ids = self._filter_examples(texts_ids, labels_ids)
        with open(os.path.join(self.data_dir, 'valid_texts_ids.dat'), 'wb') as fout:
            pickle.dump(texts_ids, fout)
        with open(os.path.join(self.data_dir, 'valid_labels_ids.dat'), 'wb') as fout:
            pickle.dump(labels_ids, fout)
        print("Valid Data Done {}".format(len(labels_ids)))

    def dump_test_features(self, text_name, label_name):
        text_path = os.path.join(self.data_dir, text_name)
        label_path = os.path.join(self.data_dir, label_name)
        texts, labels = self._get_data_from_json(text_path, label_path)
        texts_ids = self.tokenizer.texts_to_sequences(texts)
        # padding
        texts_ids = tf.keras.preprocessing.sequence.pad_sequences(texts_ids,
                                                                  maxlen=self.max_sentence_length,
                                                                  padding='post',
                                                                  truncating='post')
        labels_ids = np.array([self._transform_label(label) for label in labels])
        # texts_ids, labels_ids = self._filter_examples(texts_ids, labels_ids)
        with open(os.path.join(self.data_dir, 'test_texts_ids.dat'), 'wb') as fout:
            pickle.dump(texts_ids, fout)
        with open(os.path.join(self.data_dir, 'test_labels_ids.dat'), 'wb') as fout:
            pickle.dump(labels_ids, fout)
        print("Test Data Done {}".format(len(labels_ids)))

    def dump_word_embedding(self, vocabulary):
        vocab_size = len(vocabulary)
        print("vocabulary size is {}".format(vocab_size))
        word_vectors = KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)
        embed_size = word_vectors.vector_size
        bound = np.sqrt(6.0 / embed_size)
        vocab_size = len(vocabulary)
        word_embeddings = np.random.uniform(-bound, bound, [vocab_size+1, embed_size])
        for word in vocabulary:
            # print(word)
            if word in word_vectors:
                word_embeddings[vocabulary[word], :] = word_vectors[word]

        with open(os.path.join(self.data_dir, 'word_embeddings.dat'), 'wb') as fout:
            pickle.dump(word_embeddings, fout)

    def dump_meta_data(self):
        with open(os.path.join(self.data_dir, "tokenizer.dat"), 'wb') as fout:
            pickle.dump(self.tokenizer, fout)
        with open(os.path.join(self.data_dir, "label_map.dat"), 'wb') as fout:
            pickle.dump(self.label_map, fout)
        with open(os.path.join(self.data_dir, "max_sentence_length.dat"), 'wb') as fout:
            pickle.dump(self.max_sentence_length, fout)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise self.labels

    @classmethod
    def _get_data_from_json(cls, text_path, label_path):
        with open(text_path, 'rt') as fin:
            texts = json.load(fin)
        with open(label_path, 'rt') as fin:
            labels = json.load(fin)
        return texts, labels

    @classmethod
    def _filter_examples(cls, text_ids, label_ids):
        output_text_ids = list()
        output_label_ids = list()
        count = 0
        for text_id, label_id in zip(text_ids, label_ids):
            if label_id is not None:
                output_label_ids.append(label_id)
                output_text_ids.append(text_id)
            else:
                count += 1
        print("Filter {} examples".format(count))
        return np.array(output_text_ids), np.array(output_label_ids)


if __name__ == "__main__":
    data_dir = r'/home/yaojq/data/text/reuters'
    word2vec_path = r'/home/yaojq/data/word_embedding/GoogleNews-vectors-negative300.bin'
    print(data_dir)
    max_seq_length = 512
    processor = DataProcessor(data_dir, word2vec_path, max_seq_length)
    processor.build_label_map("train_labels.txt", "valid_labels.txt", "test_labels.txt")
    processor.dump_train_features("train_texts.txt", "train_labels.txt")
    processor.dump_eval_features("valid_texts.txt", "valid_labels.txt")
    processor.dump_test_features("test_texts.txt", "test_labels.txt")
    processor.dump_meta_data()

