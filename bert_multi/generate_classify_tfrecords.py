# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from tqdm import tqdm
from bert_multi import tokenization
from bert_multi.bert_multi_config import FLAGS
import tensorflow as tf
import pickle
import json



class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id



class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class TeleProcessor():
    def __init__(self, data_dir):
        self.language = "zh"
        self.data_dir = data_dir

    def get_train_examples(self):
        """See base class."""
        with open(os.path.join(self.data_dir, "train_titles.dat"), 'rb') as fin:
            train_titles = pickle.load(fin)
        with open(os.path.join(self.data_dir, "train_labels.dat"), 'rb') as fin:
            train_labels = pickle.load(fin)
        examples = []
        for i in range(len(train_labels)):
            guid = "train-%d" % (i)
            text_a = tokenization.convert_to_unicode(train_titles[i])
            text_b = None
            label = [tokenization.convert_to_unicode(train_label_one) for train_label_one in train_labels[i]]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self):
        """See base class."""
        with open(os.path.join(self.data_dir, "test_titles.dat"), 'rb') as fin:
            valid_titles = pickle.load(fin)
        with open(os.path.join(self.data_dir, "test_labels.dat"), 'rb') as fin:
            valid_labels = pickle.load(fin)
        examples = []
        for i in range(len(valid_labels)):
            guid = "valid-%d" % (i)
            text_a = tokenization.convert_to_unicode(valid_titles[i])
            text_b = None
            label = [tokenization.convert_to_unicode(valid_label_one) for valid_label_one in valid_labels[i]]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for prediction."""
        with open(os.path.join(self.data_dir, "test_titles.dat"), 'rb') as fin:
            test_titles = pickle.load(fin)
        with open(os.path.join(self.data_dir, "test_labels.dat"), 'rb') as fin:
            test_labels = pickle.load(fin)
        examples = []
        for i in range(len(test_labels)):
            guid = "test-%d" % (i)
            text_a = tokenization.convert_to_unicode(test_titles[i])
            text_b = None
            label = [tokenization.convert_to_unicode(test_label_one) for test_label_one in test_labels[i]]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""

        return ['014', '021', '031', '032', '033', '034', '041', '042', '043', '044', '061', '062', '063', '071', '072', '073', '081', '082', '083', '084']


class TextProcessor():
    def __init__(self, data_dir):
        self.language = "zh"
        self.data_dir = data_dir
        self.total_labels = set()

    def get_train_examples(self):
        """See base class."""
        with open(os.path.join(self.data_dir, "train_texts.txt"), 'rt') as fin:
            train_texts = json.load(fin)
        with open(os.path.join(self.data_dir, "train_labels.txt"), 'rt') as fin:
            train_labels = json.load(fin)
        examples = []
        for i in range(len(train_labels)):
            guid = "train-%d" % (i)
            text_a = tokenization.convert_to_unicode(train_texts[i])
            text_b = None
            label = [tokenization.convert_to_unicode(train_label_one) for train_label_one in train_labels[i]]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self):
        """See base class."""
        with open(os.path.join(self.data_dir, "valid_texts.txt"), 'rt') as fin:
            valid_texts = json.load(fin)
        with open(os.path.join(self.data_dir, "valid_labels.txt"), 'rt') as fin:
            valid_labels = json.load(fin)
        examples = []
        for i in range(len(valid_labels)):
            if any([label not in self.total_labels for label in valid_labels[i]]):
                continue
            guid = "valid-%d" % (i)
            text_a = tokenization.convert_to_unicode(valid_texts[i])
            text_b = None
            label = [tokenization.convert_to_unicode(valid_label_one) for valid_label_one in valid_labels[i]]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for prediction."""
        with open(os.path.join(self.data_dir, "test_texts.txt"), 'rt') as fin:
            test_texts = json.load(fin)
        with open(os.path.join(self.data_dir, "test_labels.txt"), 'rt') as fin:
            test_labels = json.load(fin)
        examples = []
        for i in range(len(test_labels)):
            if any([label not in self.total_labels for label in test_labels[i]]):
                continue
            guid = "test-%d" % (i)
            text_a = tokenization.convert_to_unicode(test_texts[i])
            text_b = None
            label = [tokenization.convert_to_unicode(test_label_one) for test_label_one in test_labels[i]]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        with open(os.path.join(self.data_dir, "train_labels.txt"), 'rt') as fin:
            train_labels = json.load(fin)
        with open(os.path.join(self.data_dir, "valid_labels.txt"), 'rt') as fin:
            valid_labels = json.load(fin)
        with open(os.path.join(self.data_dir, "test_labels.txt"), 'rt') as fin:
            test_labels = json.load(fin)
        for train_label in train_labels + valid_labels + test_labels:
            self.total_labels = self.total_labels.union(train_label)
        return self.total_labels


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  # while len(input_ids) < max_seq_length:
  #   input_ids.append(0)
  #   input_mask.append(0)
  #   segment_ids.append(0)
  if len(input_ids) < max_seq_length:
      padding = [0] * (max_seq_length - len(input_ids))
      input_ids.extend(padding)
      input_mask.extend(padding)
      segment_ids.extend(padding)
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = [0] * len(label_map)
  for label_one in example.label:
      label_id[label_map[label_one]] = 1
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s " % " ".join([str(x) for x in example.label]))
    tf.logging.info("one hot label: %s " % " ".join([str(x) for x in label_id]))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in tqdm(enumerate(examples)):
    # if ex_index % 10000 == 0:
    #   tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_id)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())


def file_based_input(input_file, seq_length, num_labels, is_training,
                     batch_size,drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([num_labels], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    d = tf.data.TFRecordDataset(input_file)
    if is_training:
        d = d.repeat(FLAGS.num_train_epochs)
        d = d.shuffle(buffer_size=100)
    d = d.map(lambda record: _decode_record(record, name_to_features))
    d = d.batch(batch_size, drop_remainder)
    return d


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processor = TextProcessor(FLAGS.data_dir)

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    print("Generate train TFRecords")
    train_examples = processor.get_train_examples()
    train_file = os.path.join(FLAGS.data_dir, "train_multi.tfrecord")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    print("Generate eval TFRecords")
    eval_examples = processor.get_dev_examples()
    eval_file = os.path.join(FLAGS.data_dir, "eval_multi.tfrecord")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
    print("Generate predict TFRecords")
    predict_examples = processor.get_test_examples()
    predict_file = os.path.join(FLAGS.data_dir, "predict_multi.tfrecord")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)


if __name__ == "__main__":
    tf.app.run()
