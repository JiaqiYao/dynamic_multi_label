# -*- coding: utf-8 -*-
from xmlCNN.xmlCNN_model import xmlCNN
from xmlCNN.xmlCNN_config import FLAGS
from utils.load_data_helper import load_pickle_data, load_meta_data,load_word_embeddings
import tensorflow as tf
import numpy as np
from metric.measure import measure_multi_label
import os
from collections import OrderedDict
from pprint import pprint
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(_):
    tokenizer, max_sentence_length, label_map = load_meta_data(FLAGS.data_dir)
    num_labels = len(label_map)
    word_embeddings = load_word_embeddings(FLAGS, tokenizer.word_index)
    train_texts_ids, train_labels_ids = load_pickle_data(FLAGS.data_dir, 'train')
    eval_texts_ids, eval_labels_ids = load_pickle_data(FLAGS.data_dir, 'valid')
    # feed data & training
    number_of_training_data = np.size(train_texts_ids, 0)
    number_of_valid_data = np.size(eval_texts_ids, 0)
    batch_size = FLAGS.batch_size
    num_batches_train_per_epoch = int((number_of_training_data - 1) / batch_size) + 1
    num_batches_valid_per_epoch = int((number_of_valid_data - 1) / batch_size) + 1
    # shuffle data
    shuffle_indices = np.random.permutation(np.arange(number_of_training_data))
    shuffled_train_texts_ids = train_texts_ids[shuffle_indices]
    shuffled_train_labels_ids = train_labels_ids[shuffle_indices]

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
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        checkPoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkPoint is not None:
            saver = tf.train.Saver(max_to_keep=100)
            saver.restore(sess, checkPoint.model_checkpoint_path)
            print("restored %s" % checkPoint.model_checkpoint_path)
        else:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

        # Summaries for loss and accuracy
        train_summary_op = model.summary_op
        train_summary_dir = os.path.join(FLAGS.log_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Summaries for loss and accuracy
        valid_summary_op = model.summary_op
        valid_summary_dir = os.path.join(FLAGS.log_dir, "summaries", "valid")
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)
        results = list()
        for epoch in range(1, FLAGS.num_epochs+1):
            loss = 0
            train_probabilities_list = list()
            for batch_num in range(num_batches_train_per_epoch):
                start = batch_num * batch_size
                end = min((batch_num + 1) * batch_size, number_of_training_data)
                feed_dict = {
                    model.sentence: shuffled_train_texts_ids[start:end],
                    model.labels: shuffled_train_labels_ids[start:end],
                    model.keep_prob: FLAGS.keep_prob,
                    model.is_training: True,
                }
                curr_loss, summaries, probabilities, _, train_step = sess.run([model.loss_val,
                                                                               train_summary_op,
                                                                               model.probabilities,
                                                                               model.train_op,
                                                                               model.global_step],
                                                                              feed_dict=feed_dict)
                loss += curr_loss * (end - start)
                train_probabilities_list.append(probabilities)
                train_summary_writer.add_summary(summaries, train_step)
            train_loss = loss / number_of_training_data

            print("Train epcoh {} Done!".format(epoch))
            train_probabilities = np.concatenate(train_probabilities_list, axis=0)

            current_step = tf.train.global_step(sess, model.global_step)

            if epoch % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

            if epoch % FLAGS.validate_every == 0:
                result = OrderedDict()
                result['epoch'] = epoch
                result['current_step'] = train_step
                result['train_loss'] = train_loss
                print("The train loss is {}".format(train_loss))
                result.update(measure_multi_label(train_probabilities, shuffled_train_labels_ids, 'train'))
                print("eval the model-{}".format(train_step))
                valid_probabilities_list = list()
                loss = 0
                for batch_num in range(num_batches_valid_per_epoch):
                    start = batch_num * batch_size
                    end = min((batch_num + 1) * batch_size, number_of_valid_data)
                    feed_dict = {
                        model.sentence: eval_texts_ids[start:end],
                        model.labels: eval_labels_ids[start:end],
                        model.keep_prob: 1.0,
                        model.is_training: False,
                    }
                    curr_loss,  summaries, probabilities, step = sess.run([model.loss_val,
                                                                      valid_summary_op,
                                                                      model.probabilities,
                                                                      model.global_step],
                                                                     feed_dict=feed_dict)
                    loss += curr_loss * (end - start)
                    valid_summary_writer.add_summary(summaries, step)
                    valid_probabilities_list.append(probabilities)
                eval_loss = loss / number_of_valid_data
                result['eval_loss'] = eval_loss
                print("The eval loss is {}".format(eval_loss))
                valid_probabilities = np.concatenate(valid_probabilities_list, axis=0)
                result.update(measure_multi_label(valid_probabilities, eval_labels_ids, 'eval'))
                results.append(result)
    best_result = {'eval_loss': 100}
    with open(os.path.join(FLAGS.log_dir, 'xmlCNN-'+FLAGS.data_name+'-results.csv'), 'wt') as fout:
        keys = result.keys()
        fout.write(','.join(keys) + '\n')
        for result in results:
            if result['eval_loss'] < best_result['eval_loss']:
                best_result = result
            values = ','.join([str(value) for value in result.values()]) + '\n'
            fout.write(values)
    print('Done')
    pprint(best_result)


if __name__ == "__main__":
    tf.app.run()