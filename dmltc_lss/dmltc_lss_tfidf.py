from utils.load_data_helper import load_pickle_data
from metric.measure import measure_multi_label, measure_ex
from utils.text2tfidf import load_tfidf_data
from collections import OrderedDict
import numpy as np
from pprint import pprint
from tqdm import tqdm
import pickle
import os


def load_date(data_dir, mode):
    with open(os.path.join(data_dir, mode+'_dates.dat'), 'rb') as fin:
        dates = pickle.load(fin)
    return dates


def query_similarity(data_dir):
    _, train_labels_ids = load_pickle_data(data_dir, 'train')
    _, eval_labels_ids = load_pickle_data(data_dir, 'valid')
    _, test_labels_ids = load_pickle_data(data_dir, 'test')
    train_sentence_features, eval_sentence_features, test_sentence_features = load_tfidf_data(data_dir)
    train_sentence_features = train_sentence_features.toarray()
    eval_sentence_features = eval_sentence_features.toarray()
    test_sentence_features = test_sentence_features.toarray()
    sentence_features = np.concatenate([train_sentence_features, eval_sentence_features], axis=0)
    labels_ids = list(np.concatenate([train_labels_ids, eval_labels_ids], axis=0))
    sim_values = np.matmul(test_sentence_features, sentence_features.T)
    # add time decay
    train_dates = load_date(data_dir, 'train')
    eval_dates = load_date(data_dir, 'valid')
    test_dates = load_date(data_dir, 'test')
    dates = train_dates + eval_dates
    pred_label_ids = []

    for test_date, test_sentence_feature, test_label_ids in tqdm(zip(test_dates, test_sentence_features, test_labels_ids)):
        test_sentence_feature.shape = 1, -1
        delta_time = np.array([(test_date-date).days for date in dates])
        sim_values = np.dot(test_sentence_feature, sentence_features.T)
        decay_sim_values = sim_values #* np.exp(-delta_time/365000)
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


if __name__ == "__main__":
    data_dir = r"/home/yaojq/data/text/reuters"
    query_similarity(data_dir)