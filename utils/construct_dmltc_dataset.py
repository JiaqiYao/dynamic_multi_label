from gensim.summarization import textcleaner
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
from collections import Counter
import json
import pickle
import glob
import re
import os


data_dir = os.path.join(os.path.expanduser('~'), r'data/text/reuters')
month_dict = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
              'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}


def flat_labels(labels):
    output = []
    for label in labels:
        output = output + label
    return output


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

# find top 10 labels
train_labels_flat = flat_labels(train_labels)
test_labels_flat = flat_labels(test_labels)
train_labels_counter = Counter(train_labels_flat)
test_labels_counter = Counter(test_labels_flat)
labels_counter = train_labels_counter+test_labels_counter
sorted_labels_counter = sorted(labels_counter.items(), key=lambda x:x[1])
top_labels = set([x[0] for x in sorted_labels_counter[-10:]])
# fliter by top 10 labels


def filter_dataset(texts, dates, labels, top_labels):
    output_labels = []
    output_texts = []
    output_dates = []
    for label, text, date in zip(labels, texts, dates):
        if all([x in top_labels for x in label]):
            output_labels.append(label)
            output_texts.append(text)
            output_dates.append(date)
    return output_texts, output_dates, output_labels


train_texts, train_dates, train_labels = filter_dataset(train_texts, train_dates, train_labels, top_labels)
test_texts, test_dates, test_labels = filter_dataset(test_texts, test_dates, test_labels, top_labels)
number_of_train = len(train_labels)
# sorted by date
train_data = zip(train_texts, train_labels, train_dates)
train_data = sorted(train_data, key=lambda x: x[-1])
train_texts, train_labels, train_dates = zip(*train_data)
test_data = sorted(zip(test_texts, test_labels, test_dates), key=lambda x:x[-1])
test_texts, test_labels, test_dates = zip(*test_data)
# construct dynamic multi label text classification dataset
top_labels = list(top_labels)
relations = [{0: top_labels[0:4], 1: top_labels[1:5], 2: top_labels[2:6], 3: top_labels[3:7], 4: top_labels[7:]},
             {0: top_labels[0:3], 1: top_labels[3:4], 2: top_labels[2:5], 3: top_labels[3:6], 4: top_labels[6:]},
             {0: top_labels[2:5], 1: top_labels[4:6], 2: top_labels[5:7], 3: top_labels[4:8], 4: top_labels[5:]}]

period = 2000


def construct_labels(labels, relations, period):
    output = []
    for num, label in enumerate(labels):
        output_label = []
        for one in label:
            num_period = num // period
            relation = relations[num_period%3]
            for i in range(5):
                if one in relation[i]:
                    #concept drift
                    output_label.append(i)
        output.append(list(set(output_label)))
    return output


labels = construct_labels(train_labels+test_labels, relations, period)
train_labels = labels[:number_of_train]
test_labels = labels[number_of_train:]
# train valid split
train_data = list(zip(train_texts, train_labels, train_dates))
number_of_data = len(train_data)
percent = 0.9
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
print('The number of test data is {}'.format(len(test_labels)))
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