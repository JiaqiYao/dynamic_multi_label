#coding:utf-8
import tensorflow as tf
import numpy as np
from xmlCNN.xmlCNN_config import FLAGS


class xmlCNN:
    def __init__(self,
                 word_embedding,
                 sentence_len,
                 hidden_dim,
                 num_classes):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.word_embed_size = np.size(word_embedding,1)
        self.sentence_len = sentence_len
        self.hidden_dim = hidden_dim
        # add placeholder (X,label)
        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_len], name="sentence")  # X
        self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name="labels")  # y
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool,name="is_training")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        #bulid graph
        # word embedding matrix
        self.word_embeddings = tf.get_variable("word_embedding",
                                               initializer=tf.constant(word_embedding, dtype=tf.float32),
                                               trainable=True
                                              )
        self.logits = self.inference() #[None, self.num_classes]
        self.probabilities = tf.nn.sigmoid(self.logits)
        #loss
        self.loss_val = self.loss()
        #train
        self.train_op = self.train()  

        self.summary_op = self.summary()

    def inference(self):

        sentence_embeddings = tf.nn.embedding_lookup(self.word_embeddings,self.sentence)  # [None,self.sentence_len,self.embed_size]
        #dropout
        sentence_embeddings = tf.nn.dropout(sentence_embeddings, self.keep_prob)
        #expand dim
        sentence_embeddings = tf.expand_dims(sentence_embeddings, -1)
        #kernel size
        
        filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
        #number of kernels per filter_sizes       
        num_filters = list(map(int, FLAGS.num_filters.split(",")))
        
        pooled_outputs = []
        for num_filter, filter_size in zip(num_filters,filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer            
                conv1 = tf.layers.conv2d(
                    inputs=sentence_embeddings,
                    filters=num_filter,
                    kernel_size=(filter_size,self.word_embed_size),
                    strides=(1, 1),
                    padding='valid',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    bias_initializer=tf.constant_initializer(0.01)
                    )
                # Avgpooling over the outputs
                pool1 = tf.layers.max_pooling2d(
                    inputs=conv1,
                    pool_size=(self.sentence_len-filter_size+1, 1),
                    strides=(1, 1),
                    padding='valid'
                    )           
                pooled_outputs.append(pool1)
        num_filters_total = np.sum(num_filters)
        pooled = tf.concat(pooled_outputs,3)
        self.pooled_outputs_flat = tf.reshape(pooled, [-1, num_filters_total])
        #full connection
        self.hidden = tf.layers.dense(
            inputs=self.pooled_outputs_flat,
            units=FLAGS.hidden_dim,
            kernel_initializer=tf.glorot_uniform_initializer())       
        #dropout
        # with tf.name_scope("dropout"):
        #     hidden_dropout = tf.nn.dropout(hidden, self.keep_prob)
        logits = tf.layers.dense(
            inputs=self.hidden,
            units=self.num_classes,
            kernel_initializer=tf.glorot_uniform_initializer()
        )
        return logits
    
    def loss(self):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
        self.loss_ex = tf.reduce_mean(losses)
#        self.l2_loss = tf.reduce_sum(tf.square(self.output_weights))
        loss = self.loss_ex #+ FLAGS.l2_reg_lambda * self.l2_loss
        return loss
    
    # def train(self):
    #     """based on the loss, use SGD to update parameter"""
    #     optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    #     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #     with tf.control_dependencies(update_ops):
    #         train_op = optimizer.minimize(self.loss_val, global_step=self.global_step)
    #     return train_op
    def train(self):
        # """based on the loss, use SGD to update parameter"""
        # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     train_op = optimizer.minimize(self.loss_val, global_step=self.global_step)
        # return train_op
        var_list = tf.trainable_variables()
        word_embed_var = tf.trainable_variables("word_embedding")
        var_list.remove(word_embed_var[0])
        optimizer1 = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        #         if(self.global_step < 2000):
        #             optimizer2 = tf.train.AdamOptimizer(learning_rate=0)
        #         else:
        optimizer2 = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate/2)
        #         optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            #             train_op = optimizer.minimize(self.loss_val,
            #                               global_step = self.global_step)
            train_op1 = optimizer1.minimize(self.loss_val,
                                            global_step=self.global_step,
                                            var_list=var_list)
            train_op2 = optimizer2.minimize(self.loss_val,
                                            global_step=self.global_step,
                                            var_list=word_embed_var)
        train_op = tf.group(train_op1, train_op2)
        return train_op
    
    def summary(self):
        train_summary = list()
        train_summary.append(tf.summary.scalar("loss", self.loss_val))
        return tf.summary.merge(train_summary)