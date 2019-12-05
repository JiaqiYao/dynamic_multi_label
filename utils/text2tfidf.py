# coding:utf-8
import os
from scipy.sparse import csr_matrix
import pickle
import json
import warnings
from gensim.summarization import textcleaner
from tqdm import tqdm
import unicodedata

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore')
    from gensim import corpora, models


class tfidf_text:
    """ tfidf_text: a text model of linear_model
    """

    def __init__(self):
        self.stop_word = []
        self.corpus = []
        self.dictionary = corpora.Dictionary()
        self.tfidf = None

    def load_corpus_build_dict(self, train_texts):
        # load corpus and build dictionary
        # load stop word
        print("load corpus and build dictionary ing...")
        # load corpus and build dictionary
        for line in train_texts:
            tokens = [word for word in line
                      if word not in self.stop_word]
            self.corpus.append(tokens)
            self.dictionary.add_documents([tokens])

    def filter_dictionary(self, no_below=5, no_above=0.7, keep_n=50000, keep_tokens=None):
        """
        Filter out tokens that appear in

        1. less than `no_below` documents (absolute number) or
        2. more than `no_above` documents (fraction of total corpus size, *not*
           absolute number).
        3. if tokens are given in keep_tokens (list of strings), they will be kept regardless of
           the `no_below` and `no_above` settings
        4. after (1), (2) and (3), keep only the first `keep_n` most frequent tokens (or
           keep all if `None`).

        After the pruning, shrink resulting gaps in word ids.

        **Note**: Due to the gap shrinking, the same word may have a different
        word id before and after the call to this function!
        """
        self.dictionary.filter_extremes(no_below=no_below,
                                        no_above=no_above,
                                        keep_n=keep_n,
                                        keep_tokens=keep_tokens)

    def to_csr(self, bows):
        data = []
        rows = []
        cols = []
        count = 0
        for line in bows:
            for elem in line:
                rows.append(count)
                cols.append(elem[0])
                data.append(elem[1])
            count += 1
        bow_sparse_matrix = csr_matrix((data, (rows, cols)), shape=[count, self.dictionary.__len__()])
        return bow_sparse_matrix

    def load_train_data(self):
        print("load train data ...")
        train_bows = []
        for tokens in self.corpus:
            train_bows.append(self.dictionary.doc2bow(tokens))
        # Transforming bows to tfidfs
        self.tfidf = models.TfidfModel(train_bows)
        train_tfidfs = [self.tfidf[bow] for bow in train_bows]
        train_tfidfs = self.to_csr(train_tfidfs)
        return train_tfidfs

    def load_test_data(self, test_texts):
        print("load test data...")
        test_bows = []
        for line in test_texts:
            tokens = [word for word in line if word not in self.stop_word]
            test_bows.append(self.dictionary.doc2bow(tokens))
        # pred_labels = self.clf.predict(test_bows)
        test_tfidfs = [self.tfidf[bow] for bow in test_bows]
        test_tfidfs = self.to_csr(test_tfidfs)
        return test_tfidfs


def load_tfidf_data(data_dir):
    if os.path.isfile(os.path.join(data_dir, 'tfidf_trainX')):
        print("load data from " + data_dir)
        with(open(os.path.join(data_dir, 'tfidf_trainX'), 'rb')) as fin:
            trainX = pickle.load(fin)
        with(open(os.path.join(data_dir, 'tfidf_validX'), 'rb')) as fin:
            validX = pickle.load(fin)
        with(open(os.path.join(data_dir, 'tfidf_testX'), 'rb')) as fin:
            testX = pickle.load(fin)
        return [trainX, validX, testX]
    else:
        return None


def dump_tfidf_data(data_dir, data):
    trainX, validX, testX = data
    with(open(os.path.join(data_dir, 'tfidf_trainX'), 'wb')) as fout:
        pickle.dump(trainX, fout)

    with(open(os.path.join(data_dir, 'tfidf_validX'), 'wb')) as fout:
        pickle.dump(validX, fout)

    with(open(os.path.join(data_dir, 'tfidf_testX'), 'wb')) as fout:
        pickle.dump(testX, fout)



def cut_word(texts):
    output = []
    for text in texts:
        output.append(list(textcleaner.tokenize_by_word(text)))
    return output


def load_text_data(data_dir):
    with open(os.path.join(data_dir, 'train_texts.txt'), 'rt') as fin:
        train_texts = json.load(fin)
        trainX = cut_word(train_texts)
    with open(os.path.join(data_dir, 'valid_texts.txt'), 'rt') as fin:
        valid_texts = json.load(fin)
        validX = cut_word(valid_texts)
    with open(os.path.join(data_dir, 'test_texts.txt'), 'rt') as fin:
        test_texts = json.load(fin)
        testX = cut_word(test_texts)

    return trainX, validX, testX


if __name__ == "__main__":
    data_dir = r"/home/yaojq/data/text/reuters"
    trainX, validX, testX = load_text_data(data_dir)
    tfidf_text_obj = tfidf_text()
    tfidf_text_obj.load_corpus_build_dict(trainX)
    tfidf_text_obj.filter_dictionary()
    trainX = tfidf_text_obj.load_train_data()
    validX = tfidf_text_obj.load_test_data(validX)
    testX = tfidf_text_obj.load_test_data(testX)
    data = [trainX, validX, testX]
    dump_tfidf_data(data_dir, data)