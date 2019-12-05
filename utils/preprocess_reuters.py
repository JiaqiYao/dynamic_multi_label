from gensim.summarization import textcleaner
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
import json
import pickle
import glob
import re
import os


data_dir = r'/home/yaojq/data/text/reuters'
month_dict = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
              'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}


def handle_time_str(time_str):
    ydm, hms = time_str.split()
    day, month, year = ydm.split('-')
    day = int(day)
    month = month_dict[month]
    year = int(year)
    hour, min, sec = hms.split(':')
    hour = int(hour)
    min = int(min)
    sec = int(float(sec))
    return datetime(year=year, month=month, day=day, hour=hour, minute=min, second=sec)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


train_labels = list()
test_labels = list()
train_texts = list()
test_texts = list()
train_dates = list()
test_dates = list()
total_labels = set()

filenames = glob.glob(os.path.join(data_dir, '*.sgm'))
for filename in filenames:
    # print(filename)
    with open(filename, 'r', errors='ignore') as data:
        soup = BeautifulSoup(data)
        docs = soup.body.find_all('reuters')
        for doc in docs:
            topics = doc.topics
            try:
                date = handle_time_str(doc.date.text)
                if len(topics.contents) > 0:
                    labels = list()
                    for topic in topics:
                        labels.extend(topic.contents)
                    total_labels = total_labels.union(labels)
                    text = ''
                    for content in doc.find('text').contents:
                        text = text + str(content)
                    # clean the punctuation and escape character
                    # print(textcleaner.clean_text_by_sentences(text))
                    text = clean_str(text)
                    if doc['lewissplit'] == 'TRAIN':
                        train_labels.append(labels)
                        train_texts.append(text)
                        train_dates.append(date)
                    elif doc['lewissplit'] == 'TEST':
                        test_labels.append(labels)
                        test_texts.append(text)
                        test_dates.append(date)
            except Exception as e:
                print(e)
                print(filename)
                print(doc['newid'])

# train valid split
train_data = zip(train_texts, train_labels, train_dates)
train_data = sorted(train_data, key=lambda x:x[-1])
number_of_data = len(train_data)
percent = 0.7
split_index = int(number_of_data*percent)
valid_data = train_data[split_index:]
train_data = train_data[:split_index]
train_texts, train_labels, train_dates = zip(*train_data)
valid_texts, valid_labels, valid_dates = zip(*valid_data)
print('The number of train data is {}'.format(len(train_labels)))
print('The number of valid data is {}'.format(len(valid_labels)))

# sort test data
test_data = sorted(zip(test_texts, test_labels, test_dates), key=lambda x:x[-1])
test_texts, test_labels, test_dates = zip(*test_data)

with open(os.path.join(data_dir, 'train_labels.txt'), 'wt') as fout:
    json.dump(train_labels, fout)
with open(os.path.join(data_dir, 'valid_labels.txt'), 'wt') as fout:
    json.dump(valid_labels, fout)
with open(os.path.join(data_dir, 'test_labels.txt'), 'wt') as fout:
    json.dump(test_labels, fout)

with open(os.path.join(data_dir, 'train_texts.txt'), 'wt') as fout:
    json.dump(train_texts, fout)
with open(os.path.join(data_dir, 'valid_texts.txt'), 'wt') as fout:
    json.dump(valid_texts, fout)
with open(os.path.join(data_dir, 'test_texts.txt'), 'wt') as fout:
    json.dump(test_texts, fout)

with open(os.path.join(data_dir, 'train_dates.dat'), 'wb') as fout:
    pickle.dump(list(train_dates), fout)
with open(os.path.join(data_dir, 'valid_dates.dat'), 'wb') as fout:
    pickle.dump(list(valid_dates), fout)
with open(os.path.join(data_dir, 'test_dates.dat'), 'wb') as fout:
    pickle.dump(list(test_dates), fout)

with open(os.path.join(data_dir, 'total_labels.txt'), 'wt') as fout:
    json.dump(list(total_labels), fout)

print('over')