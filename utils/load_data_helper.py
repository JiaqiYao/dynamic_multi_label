from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf
from scipy import io as sio
import numpy as np
import pickle
import os


def file_based_input(input_file,
                     seq_length,
                     num_labels,
                     batch_size,
                     drop_remainder):
    name_to_features = {
        "text_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([num_labels], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        return example

    d = tf.data.TFRecordDataset(input_file)
    d = d.map(lambda record: _decode_record(record, name_to_features))
    d = d.shuffle(buffer_size=100)
    d = d.batch(batch_size, drop_remainder)

    return d


def load_meta_data(data_dir):
    with open(os.path.join(data_dir, 'label_map.dat'), 'rb') as fin:
        label_map = pickle.load(fin)
    with open(os.path.join(data_dir, 'max_sentence_length.dat'), 'rb') as fin:
        max_sentence_length = pickle.load(fin)
    with open(os.path.join(data_dir, 'tokenizer.dat'), 'rb') as fin:
        tokenizer = pickle.load(fin)
    return tokenizer, max_sentence_length, label_map


def load_word_embeddings(FLAGS, vocabulary):
    vocab_size = len(vocabulary)
    print("vocabulary size is {}".format(vocab_size))
    bound = np.sqrt(6.0 / FLAGS.word_embed_size)
    # plus 1 for unknown words
    if not FLAGS.use_embedding:
        return np.random.uniform(-bound, bound, (vocab_size+1, FLAGS.word_embed_size))
    else:
        with open(os.path.join(FLAGS.data_dir, 'word_embeddings.dat'), 'rb') as fin:
            word_embeddings = pickle.load(fin)
    return word_embeddings


def load_label_embeddings(data_path):
    with open(data_path, 'rb') as fin:
        w = pickle.load(fin)
    return w


# linear drop
def drop_word(word_data, drop_rate):
    shape = word_data.shape

    output = list()
    mask_rate = np.linspace(0, drop_rate, shape[1])
    for i in range(shape[0]):
        mask = np.random.rand(shape[1]) > mask_rate
        output.append(word_data[i] * mask)
    output = np.array(output)
    return output


def load_pickle_data(data_dir, mode):
    labels_ids_path = os.path.join(data_dir, mode+"_labels_ids.dat")
    texts_ids_path = os.path.join(data_dir, mode + "_texts_ids.dat")
    with open(labels_ids_path, 'rb') as fin:
        labels_ids = pickle.load(fin)
    with open(texts_ids_path, 'rb') as fin:
        texts_ids = pickle.load(fin)
    return texts_ids, labels_ids