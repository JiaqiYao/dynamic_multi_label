import os
import tensorflow as tf
from heapq import nlargest
import numpy as np
from tqdm import tqdm
import pickle
from bert_multi import modeling
from bert_multi.bert_multi_config import FLAGS
from bert_multi.generate_classify_tfrecords import TextProcessor
from bert_multi.generate_classify_tfrecords import file_based_input
from bert_multi.bert_multi_model import bert_multi_model
from metric.measure import measure_multi_label
from pprint import pprint
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def load_date(mode):
    with open(os.path.join(FLAGS.data_dir, mode+'_dates.dat'), 'rb') as fin:
        dates = pickle.load(fin)
    return dates


def build_model(sess):
    tf.logging.set_verbosity(tf.logging.INFO)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)
    task_name = FLAGS.task_name.lower()
    processor = TextProcessor(FLAGS.data_dir)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    train_examples = processor.get_train_examples()
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model = bert_multi_model(bert_config, num_labels, FLAGS.init_checkpoint)
    print('Initializing Variables')
    sess.run(tf.global_variables_initializer())
    # Save and restore model
    checkpoint_dir = os.path.abspath(os.path.join(FLAGS.output_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    checkPoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkPoint is not None:
        saver = tf.train.Saver(max_to_keep=100)
        saver.restore(sess, checkPoint.model_checkpoint_path)
        print("restored %s" % checkPoint.model_checkpoint_path)
    else:
        exit()
    print("build model done")
    return sess, model, num_labels


def get_sentence_feature(sess, model, filename, num_labels):
    print("get sentence feature from " + filename)
    file = os.path.join(FLAGS.data_dir, filename)
    dataset = file_based_input(file,  seq_length=FLAGS.max_seq_length, is_training=False, num_labels=num_labels,
                               batch_size=FLAGS.eval_batch_size, drop_remainder=False)
    dataset_iterator = dataset.make_one_shot_iterator()
    dataset_next_data_op = dataset_iterator.get_next()
    label_ids_list = list()
    sentence_features = list()
    while True:
        try:
            next_data = sess.run(dataset_next_data_op)
            feed_dict = {
                model.input_ids: next_data['input_ids'],
                model.input_mask: next_data['input_mask'],
                model.segment_ids: next_data['segment_ids'],
                model.is_training: True
            }
            sentence_feature, step = sess.run([model.output_layer,
                                               model.global_step],
                                              feed_dict=feed_dict)
            label_ids_list.append(next_data['label_ids'])
            sentence_features .append(sentence_feature)
        except tf.errors.OutOfRangeError:
            break
    sentence_features = np.concatenate(sentence_features, axis=0)
    sentence_features = sentence_features / np.linalg.norm(sentence_features, axis=1, keepdims=True)
    return sentence_features, np.concatenate(label_ids_list, axis=0)


def dump_sentence_features(sess, model, num_labels):
    train_sentence_features,  train_labels_ids = get_sentence_feature(sess, model, "train_multi.tfrecord", num_labels)
    eval_sentence_features,  eval_labels_ids = get_sentence_feature(sess, model, "eval_multi.tfrecord", num_labels)
    test_sentence_features,  test_labels_ids = get_sentence_feature(sess, model, "predict_multi.tfrecord", num_labels)

    with open(os.path.join(FLAGS.data_dir, "train_sentence_features.dat"), 'wb') as fout:
        pickle.dump((train_sentence_features, train_labels_ids), fout)
    with open(os.path.join(FLAGS.data_dir, "eval_sentence_features.dat"), 'wb') as fout:
        pickle.dump((eval_sentence_features, eval_labels_ids), fout)
    with open(os.path.join(FLAGS.data_dir, "test_sentence_features.dat"), 'wb') as fout:
        pickle.dump((test_sentence_features, test_labels_ids), fout)
    print("dump done!!!")


# def multi_knn(sentence_features, label_ids, test_sentence_feature, K):
#     sim_values = np.dot(test_sentence_feature, sentence_features.T)
#     sim_values = sim_values.squeeze()
#     sim_values_indexes = zip(sim_values, range(len(sim_values)))
#     nlargest_sim_values_indexes = nlargest(K, sim_values_indexes, key=lambda x: x[0])
#     sim_indexes = [x[1] for x in nlargest_sim_values_indexes]
#     nn_label_ids = label_ids[sim_indexes]
#     nn_label_ids = np.mean(nn_label_ids, axis=0)
#     pred_label_ids = np.zeros(np.size(label_ids, axis=1))
#     pred_label_ids[nn_label_ids > K/2] = 1
#     return pred_label_ids


def query_similarity():
    # train_sentence_features,  train_labels_ids = get_sentence_feature(sess, model, "train_multi.tfrecord", num_labels)
    # eval_sentence_features,  eval_labels_ids = get_sentence_feature(sess, model, "eval_multi.tfrecord", num_labels)
    # test_sentence_features,  test_labels_ids = get_sentence_feature(sess, model, "predict_multi.tfrecord", num_labels)
    # sentence_features = np.concatenate([train_sentence_features, eval_sentence_features], axis=0)
    # labels_ids = train_labels_ids + eval_labels_ids
    with open(os.path.join(FLAGS.data_dir, "train_sentence_features.dat"), 'rb') as fin:
        train_sentence_features, train_labels_ids = pickle.load(fin)
    with open(os.path.join(FLAGS.data_dir, "eval_sentence_features.dat"), 'rb') as fin:
        eval_sentence_features, eval_labels_ids = pickle.load(fin)
    with open(os.path.join(FLAGS.data_dir, "test_sentence_features.dat"), 'rb') as fin:
        test_sentence_features, test_labels_ids = pickle.load(fin)
    sentence_features = np.concatenate([train_sentence_features, eval_sentence_features], axis=0)
    labels_ids = np.concatenate([train_labels_ids, eval_labels_ids], axis=0)
    # add time decay
    train_dates = load_date('train')
    eval_dates = load_date('valid')
    test_dates = load_date('test')
    dates = train_dates + eval_dates
    pred_label_ids = []
    K = 5
    for test_date, test_sentence_feature, test_label_ids in tqdm(zip(test_dates, test_sentence_features, test_labels_ids)):
        test_sentence_feature.shape = 1, -1
        delta_time = np.array([(test_date-date).days for date in dates])
        sim_values = np.dot(test_sentence_feature, sentence_features.T)
        decay_sim_values = sim_values * np.exp(-delta_time/90)
        sim_index = np.argmax(decay_sim_values)
        pred_label_ids.append(labels_ids[sim_index])
        #pred_label_ids.append(multi_knn(sentence_features, labels_ids, test_sentence_feature, K))
        # update label_ids, sentence_features, dates
        sentence_features = np.concatenate([sentence_features, test_sentence_feature], axis=0)
        dates.append(test_date)
        test_label_ids.shape = 1, -1
        labels_ids = np.append(labels_ids, test_label_ids, axis=0)
    # add time decay end
    # sim_index = np.max(sim_values, axis=1)
    # pred_label_ids = label_ids[sim_index]
    pred_label_ids = np.array(pred_label_ids)
    result = measure_multi_label(pred_label_ids, test_labels_ids, 'test')
    pprint(result)


def main(_):
    with tf.Session() as sess:
        # sess, model, num_labels = build_model(sess)
        # dump_sentence_features(sess, model, num_labels)
        query_similarity()


if __name__ == "__main__":
    tf.app.run()