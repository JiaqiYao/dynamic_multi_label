# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from bert_multi import modeling
from bert_multi import optimization
import tensorflow as tf
from bert_multi.bert_multi_config import FLAGS


class bert_multi_model:
    def __init__(self, bert_config, num_labels, init_checkpoint,
                 learning_rate=0, num_train_steps=0, num_warmup_steps=0):
        # Hyperparameter settings
        self.bert_config = bert_config
        self.num_labels = num_labels
        self.init_checkpoint = init_checkpoint
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps

        # Placehoder
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.input_ids = tf.placeholder(tf.int64, [None, FLAGS.max_seq_length], name="input_ids")
        self.input_mask = tf.placeholder(tf.int64, [None, FLAGS.max_seq_length], name="input_mask")
        self.segment_ids = tf.placeholder(tf.int64, [None, FLAGS.max_seq_length], name="segment_ids")
        self.label_ids = tf.placeholder(tf.int64, [None, self.num_labels], name="labels")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        self.loss, self.logits, self.probabilities = self.create_model()

        # initialize from the pretrained models
        self.initialize_model()
        # train op
        self.train_op = optimization.create_optimizer(self.loss, self.learning_rate, self.global_step,
                                                      self.num_train_steps, self.num_warmup_steps, False)

        # summary
        self.summary_op = self.summary()

    def create_model(self):
        """Creates a classification model."""
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids)

        # In the demo, we are doing a simple classification task on the entire
        # segment.
        #
        # If you want to use the token-level output, use model.get_sequence_output()
        # instead.
        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [self.num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            # if self.is_training:
            #     # I.e., 0.1 dropout
            #     output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            output_layer = tf.layers.dropout(output_layer, rate=0.1, training=self.is_training)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.sigmoid(logits)

            # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            one_hot_labels = tf.cast(self.label_ids, tf.float32)

            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_labels,logits=logits)
            loss = tf.reduce_mean(per_example_loss)

            return loss, logits, probabilities

    def initialize_model(self):
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if self.init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
        tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

    def summary(self):
        summary = list()
        summary.append(tf.summary.scalar("loss", self.loss))
        return tf.summary.merge(summary)