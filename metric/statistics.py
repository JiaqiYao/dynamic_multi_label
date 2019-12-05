import os
import json
data_dir = r'/home/yaojq/data/text/reuters'

total_labels = set()
num_labels = 0
with open(os.path.join(data_dir, 'train_labels.txt')) as fin:
    train_labels = json.load(fin)
    num_train = len(train_labels)
    for train_label in train_labels:
        num_labels += len(train_label)
        total_labels = total_labels.union(train_label)
    print('The number of train examples is {}'.format(num_train))

with open(os.path.join(data_dir, 'valid_labels.txt')) as fin:
    valid_labels = json.load(fin)
    num_valid = len(valid_labels)
    for valid_label in valid_labels:
        num_labels += len(valid_label)
        total_labels = total_labels.union(valid_label)
    print('The number of train examples is {}'.format(num_valid))

with open(os.path.join(data_dir, 'test_labels.txt')) as fin:
    test_labels = json.load(fin)
    num_test = len(test_labels)
    for test_label in test_labels:
        num_labels += len(test_label)
        total_labels = total_labels.union(test_label)
    print('The number of test examples is {}'.format(num_test))

print('The number of total labels is {}'.format(len(total_labels)))
print("The average number of labels is {}".format(num_labels/(num_train + num_test + num_valid)))
