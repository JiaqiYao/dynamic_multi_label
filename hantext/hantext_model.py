# coding:utf-8
import tensorflow as tf
import numpy as np
from hantext.hantext_config import FLAGS


class hantext:
    def __init__(self,
                 word_embeddings,
                 sentence_len,
                 lstm_hidden_dim,
                 fc_hidden_dim,
                 num_classes):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.word_embed_size = np.size(word_embeddings, 1)
        self.sentence_len = sentence_len
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fc_hidden_dim = fc_hidden_dim
        # add placeholder (X,label)
        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_len], name="sentence")  # X
        self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name="labels")  # y
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.k = tf.placeholder(tf.int32, name="precision_k")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        # bulid graph
        # word embedding matrix
        self.word_embeddings = tf.get_variable("word_embedding",
                                               initializer=tf.constant(word_embeddings, dtype=tf.float32))
        self.logits = self.inference()  # [None, self.num_classes]
        self.probabilities = tf.nn.sigmoid(self.logits)
        # loss
        self.loss_val = self.loss()
        # train
        self.train_op = self.train()

        self.summary_op = self.summary()

    def inference(self):
        sentence_embeddings = tf.nn.embedding_lookup(self.word_embeddings,
                                                     self.sentence)  # [None,self.sentence_len,self.embed_size]
        # dropout
        # sentence_embeddings = tf.nn.dropout(sentence_embeddings, self.keep_prob)
        # kernel size
        # with tf.name_scope("Bi-LSTM"):
        #     lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_dim)  # forward direction cell
        #     lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden_dim)  # backward direction cell
        #     #dropout
        #     lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.keep_prob)
        #     lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.keep_prob)
        #     outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
        #                                                      sentence_embeddings, dtype=tf.float32)
        #         # Concat output
        with tf.name_scope("Bi-GRU"):
            gru_fw_cell = tf.nn.rnn_cell.GRUCell(self.lstm_hidden_dim, activation=tf.nn.tanh)
            gru_bw_cell = tf.nn.rnn_cell.GRUCell(self.lstm_hidden_dim, activation=tf.nn.tanh)
            gru_fw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_fw_cell, output_keep_prob=self.keep_prob)
            gru_bw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_bw_cell, output_keep_prob=self.keep_prob)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell, gru_bw_cell, sentence_embeddings,
                                                         dtype=tf.float32)
        lstm_concat = tf.concat(outputs, axis=2)  # [batch_size, sequence_length, lstm_hidden_size * 2]

        # Attention Layer
        with tf.name_scope("attention"):
            num_units = lstm_concat.get_shape().as_list()[-1]  # Get last dimension [lstm_hidden_size * 2]
            u_attention = tf.Variable(tf.truncated_normal(shape=[num_units], stddev=0.1, dtype=tf.float32),
                                      name="u_attention")
            u = tf.layers.dense(lstm_concat, units=num_units, activation=tf.nn.tanh, use_bias=True)

            # 2. Compute weight by computing similarity of u and attention vector u_attention
            score = tf.multiply(u, u_attention)  # [batch_size, sequence_length, num_units]
            # minus max value?
            weight = tf.nn.softmax(score, axis=2)  # [batch_size, sequence_length, 1]

            # 3. Weight sum
            attention_out = tf.reduce_sum(tf.multiply(lstm_concat, weight), axis=1)  # [batch_size, num_units]

        with tf.name_scope("FC"):
            self.fc = tf.layers.dense(inputs=attention_out,
                                 units=self.num_classes,
                                 kernel_initializer=tf.glorot_uniform_initializer())
        # dropout
        with tf.name_scope("dropout"):
            hidden_dropout = tf.nn.dropout(self.fc, self.keep_prob)
        # self.output_weights = tf.get_variable(
        #     "output_weights", [self.num_classes, self.num_classes],
        #     initializer=tf.glorot_uniform_initializer()
        # )
        # # self.output_weights = tf.get_variable(
        # #     "output_weights", [self.num_classes, self.num_classes],
        # #     initializer=tf.glorot_uniform_initializer()
        # # )
        # logits = tf.matmul(hidden_dropout, self.output_weights, transpose_b=True)
        logits = tf.layers.dense(
            inputs=hidden_dropout,
            units=self.num_classes,
            kernel_initializer=tf.glorot_uniform_initializer()
            )
        return logits

    def loss(self):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        self.loss_ex = tf.reduce_mean(losses)
        #self.l2_loss = tf.reduce_sum(tf.square(self.output_weights))
        loss = self.loss_ex #+ FLAGS.l2_reg_lambda * self.l2_loss
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        #         learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
        #                                            self.global_step,
        #                                            FLAGS.decay_steps,
        #                                            FLAGS.decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        #         optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0)
        # optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.loss_val, global_step=self.global_step)
        return train_op

    #     def train(self):
    #         # """based on the loss, use SGD to update parameter"""
    #         # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    #         # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #         # with tf.control_dependencies(update_ops):
    #         #     train_op = optimizer.minimize(self.loss_val, global_step=self.global_step)
    #         # return train_op
    #         var_list = tf.trainable_variables()
    #         word_embed_var = tf.trainable_variables("word_embedding")
    #         var_list.remove(word_embed_var[0])
    #         optimizer1 = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    #         #         if(self.global_step < 2000):
    #         #             optimizer2 = tf.train.AdamOptimizer(learning_rate=0)
    #         #         else:
    #         optimizer2 = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate/3)
    #         #         optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    #         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #         with tf.control_dependencies(update_ops):
    #             #             train_op = optimizer.minimize(self.loss_val,
    #             #                               global_step = self.global_step)
    #             train_op1 = optimizer1.minimize(self.loss_val,
    #                                             global_step=self.global_step,
    #                                             var_list=var_list)
    #             train_op2 = optimizer2.minimize(self.loss_val,
    #                                             global_step=self.global_step,
    #                                             var_list=word_embed_var)
    #         train_op = tf.group(train_op1, train_op2)
    #         return train_op

    def summary(self):
        train_summary = list()
        train_summary.append(tf.summary.scalar("loss", self.loss_val))
        train_summary.append(tf.summary.scalar("loss_ex", self.loss_ex))
        #        train_summary.append(tf.summary.scalar("l2_loss", self.l2_loss))
        return tf.summary.merge(train_summary)