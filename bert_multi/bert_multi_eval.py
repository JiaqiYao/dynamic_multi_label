import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from bert_multi import modeling
from bert_multi.bert_multi_config import FLAGS
from bert_multi.generate_classify_tfrecords import TextProcessor
from bert_multi.generate_classify_tfrecords import file_based_input
from bert_multi.bert_multi_model import bert_multi_model
from bert_multi.metrics import measure_b

if __name__ == "__main__":

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
    num_lables = len(label_list)

    train_examples = processor.get_train_examples()
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model = bert_multi_model(bert_config, num_lables, FLAGS.init_checkpoint)

    eval_file = os.path.join(FLAGS.data_dir, "predict_multi.tfrecord")
    eval_dataset = file_based_input(eval_file,  seq_length=FLAGS.max_seq_length, is_training=False, num_labels=num_lables,
                                     batch_size=FLAGS.eval_batch_size, drop_remainder=False)

    with tf.Session() as sess:
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
        eval_iterator = eval_dataset.make_one_shot_iterator()
        eval_next_data_op = eval_iterator.get_next()
        label_ids_list = list()
        probabilities_list = list()
        while True:
            try:
                eval_next_data = sess.run(eval_next_data_op)
                feed_dict = {
                    model.input_ids: eval_next_data['input_ids'],
                    model.input_mask: eval_next_data['input_mask'],
                    model.segment_ids: eval_next_data['segment_ids'],
                    model.label_ids: eval_next_data['label_ids'],
                    model.is_training: True
                }
                loss, probabilities, step = sess.run([model.loss,
                                                      model.probabilities,
                                                      model.global_step],
                                                      feed_dict=feed_dict)
                label_ids_list.append(eval_next_data['label_ids'])
                probabilities_list.append(probabilities)
            except tf.errors.OutOfRangeError:
                label_ids = np.concatenate(label_ids_list, axis=0)
                probabilities = np.concatenate(probabilities_list, axis=0)
                binaries = probabilities >= 0.5  # thresholds
                #        binaries = probabilities >= 0.5
                # revise to guarantee at least one is one
                index = np.argmax(probabilities, 1)
                binaries = np.reshape(np.array(binaries), [-1])
                index = index + np.size(probabilities, 1) * np.arange(np.size(probabilities, 0))
                binaries[index] = 1
                binaries.shape = np.size(probabilities, 0), np.size(probabilities, 1)
                #        print(np.sum(np.any(binaries,1)))
                binaries = binaries.astype(int)
                micro_p, micro_r, micor_f1, marco_p, marco_r, marco_f1, hamming_loss, accuracy, precision, recall, F1 = measure_b(
                    binaries, label_ids)
                print(
                    'micro_p {0:.4f}\nmicro_r {1:.4f}\nmicor_f1 {2:.4f}\nmarco_p {3:.4f}\nmarco_r {4:.4f}\nmarco_f1 {5:.4f}\nhamming_loss {6:.4f}\naccuracy {7:.4f}\nprecision {8:.4f}\nrecall {9:.4f}\nF1 {10:.4f}\n'
                        .format(micro_p, micro_r, micor_f1, marco_p, marco_r, marco_f1, hamming_loss, accuracy,
                                precision, recall, F1)
                )

                break

