from xmlCNN.xmlCNN_model import xmlCNN
from xmlCNN.xmlCNN_config import FLAGS
from utils.load_data_helper import load_pickle_data, load_meta_data, load_word_embeddings
from metric.measure import measure_multi_label, measure_ex
import tensorflow as tf
from collections import OrderedDict
from scipy import io as sio
import numpy as np
from pprint import pprint
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main(_):
    FLAGS.use_embedding = False
    FLAGS.is_training = False
    tokenizer, max_sentence_length, label_map = load_meta_data(FLAGS.data_dir)
    num_labels = len(label_map)
    word_embeddings = load_word_embeddings(FLAGS, tokenizer.word_index)
    test_texts_ids, test_labels_ids = load_pickle_data(FLAGS.data_dir, 'test')
    batch_size = FLAGS.batch_size
    number_of_test_data = np.size(test_texts_ids, 0)
    num_batches_test_per_epoch = int((number_of_test_data - 1) / batch_size) + 1

    model = xmlCNN(
        word_embeddings,
        max_sentence_length,
        FLAGS.hidden_dim,
        num_labels
        )
    with tf.Session() as sess:
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
        result = OrderedDict()
        test_probabilities_list = list()
        loss = 0
        for batch_num in range(num_batches_test_per_epoch):
            start = batch_num * batch_size
            end = min((batch_num + 1) * batch_size, number_of_test_data)
            feed_dict = {
                model.sentence: test_texts_ids[start:end],
                model.labels: test_labels_ids[start:end],
                model.keep_prob: 1.0,
                model.is_training: False,
            }
            curr_loss, probabilities, step = sess.run([model.loss_val,
                                                       model.probabilities,
                                                       model.global_step],
                                                      feed_dict=feed_dict)
            loss += curr_loss * (end - start)
            test_probabilities_list.append(probabilities)
        test_loss = loss / number_of_test_data
        result['test_loss'] = test_loss
        test_probabilities = np.concatenate(test_probabilities_list, axis=0)
        result.update(measure_multi_label(test_probabilities, test_labels_ids, 'test'))
        pprint(result)

        # # error analysis
        # binaries = test_probabilities >= 0.5  # thresholds
        # # revise to guarantee at least one is one
        # index = np.argmax(test_probabilities, 1)
        # binaries = np.reshape(np.array(binaries), [-1])
        # index = index + np.size(test_probabilities, 1) * np.arange(np.size(test_probabilities, 0))
        # binaries[index] = 1
        # binaries.shape = np.size(test_probabilities, 0), np.size(test_probabilities, 1)
        # #        print(np.sum(np.any(binaries,1)))
        # binaries = binaries.astype(int)
        # F1_list = measure_ex(binaries, test_labels_ids)
        # records = list()
        # for i in range(len(F1_list)):
        #     records.append((test_texts_ids[i, :], test_labels_ids[i, :], binaries[i, :], F1_list[i]))
        # sorted_records = sorted(records, key=lambda x: x[-1])
        # label_items = label_map.items()
        # label_items = sorted(label_items, key=lambda x:x[-1])
        # labels = np.array([label_item[0] for label_item in label_items])
        # with open(os.path.join(FLAGS.log_dir, "xmlCNN-"+FLAGS.data_name+'-error_analysis.txt'), 'wt') as fout:
        #     print(result, file=fout)
        #     for record in sorted_records:
        #         text_ids, label_ids, pred_b, F1 = record
        #         text = ' '.join([tokenizer.index_word[id] for id in text_ids][:100])
        #         true_labels = labels[label_ids == 1]
        #         pred_labels = labels[pred_b == 1]
        #         print(text, file=fout)
        #         print(true_labels, file=fout)
        #         print(pred_labels, file=fout)
        #         print(F1, file=fout)
        #         print('', file=fout)
        # print("Done!!!")


if __name__ == "__main__":
    tf.app.run()