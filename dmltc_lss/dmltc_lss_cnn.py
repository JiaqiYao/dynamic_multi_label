from xmlCNN.xmlCNN_model import xmlCNN
from xmlCNN.xmlCNN_config import FLAGS
from utils.load_data_helper import load_pickle_data, load_meta_data, load_word_embeddings
from metric.measure import measure_multi_label, measure_ex
import tensorflow as tf
from collections import OrderedDict
import numpy as np
from pprint import pprint
from tqdm import tqdm
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def load_date(mode):
    with open(os.path.join(FLAGS.data_dir, mode+'_dates.dat'), 'rb') as fin:
        dates = pickle.load(fin)
    return dates


def build_model(sess):
    FLAGS.use_embedding = False
    FLAGS.is_training = False
    tokenizer, max_sentence_length, label_map = load_meta_data(FLAGS.data_dir)
    num_labels = len(label_map)
    word_embeddings = load_word_embeddings(FLAGS, tokenizer.word_index)

    model = xmlCNN(
        word_embeddings,
        max_sentence_length,
        FLAGS.hidden_dim,
        num_labels
        )
    print('Initializing Variables')
    sess.run(tf.global_variables_initializer())
    # Save and restore model
    checkpoint_dir = os.path.abspath(os.path.join(FLAGS.log_dir, "checkpoints"))
    checkPoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkPoint is not None:
        saver = tf.train.Saver(max_to_keep=100)
        saver.restore(sess, checkPoint.model_checkpoint_path)
        print("restored %s" % checkPoint.model_checkpoint_path)
    else:
        print("There is no checkpoint!!!")
        exit()
    return sess, model


def get_sentence_feature(sess, model, texts_ids):
    number_of_data = np.size(texts_ids, axis=0)
    batch_size = FLAGS.batch_size
    num_batches_per_epoch = number_of_data // batch_size + 1
    sentence_features = []
    for batch_num in range(num_batches_per_epoch):
        start = batch_num * batch_size
        end = min((batch_num + 1) * batch_size, number_of_data)
        feed_dict = {
            model.sentence: texts_ids[start:end],
            model.keep_prob: 1.0,
            model.is_training: False,
        }
        curr_sentence_features = sess.run(model.hidden, feed_dict=feed_dict)
        sentence_features.append(curr_sentence_features)
    sentence_features = np.concatenate(sentence_features, axis=0)
    sentence_features = sentence_features / np.linalg.norm(sentence_features, axis=1, keepdims=True)
    return sentence_features


def query_similarity(sess, model):
    train_texts_ids, train_labels_ids = load_pickle_data(FLAGS.data_dir, 'train')
    eval_texts_ids, eval_labels_ids = load_pickle_data(FLAGS.data_dir, 'valid')
    test_texts_ids, test_labels_ids = load_pickle_data(FLAGS.data_dir, 'test')
    train_sentence_features = get_sentence_feature(sess, model, train_texts_ids)
    eval_sentence_features = get_sentence_feature(sess, model, eval_texts_ids)
    test_sentence_features = get_sentence_feature(sess, model, test_texts_ids)
    sentence_features = np.concatenate([train_sentence_features, eval_sentence_features], axis=0)
    labels_ids = list(np.concatenate([train_labels_ids, eval_labels_ids], axis=0))
    sim_values = np.matmul(test_sentence_features, sentence_features.T)
    # add time decay
    train_dates = load_date('train')
    eval_dates = load_date('valid')
    test_dates = load_date('test')
    dates = train_dates + eval_dates
    pred_label_ids = []

    for test_date, test_sentence_feature, test_label_ids in tqdm(zip(test_dates, test_sentence_features, test_labels_ids)):
        test_sentence_feature.shape = 1, -1
        delta_time = np.array([(test_date-date).days for date in dates])
        sim_values = np.dot(test_sentence_feature, sentence_features.T)
        decay_sim_values = sim_values * np.exp(-delta_time/90)
        sim_index = np.argmax(decay_sim_values)
        pred_label_ids.append(labels_ids[sim_index])
        # update label_ids, sentence_features, dates
        sentence_features = np.concatenate([sentence_features, test_sentence_feature], axis=0)
        dates.append(test_date)
        labels_ids.append(test_label_ids)
    # add time decay end
    # sim_index = np.max(sim_values, axis=1)
    # pred_label_ids = label_ids[sim_index]
    pred_label_ids = np.array(pred_label_ids)
    result = measure_multi_label(pred_label_ids, test_labels_ids, 'test')
    pprint(result)


def main(_):
    with tf.Session() as sess:
        sess, model = build_model(sess)
        query_similarity(sess, model)


if __name__ == "__main__":
    tf.app.run()