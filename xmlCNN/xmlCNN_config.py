# coding:utf-8
import tensorflow as tf

flags = tf.app.flags

#hyper parameters settings
flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")
flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularization lambda (default: 0.0)")
flags.DEFINE_float("decay_rate", 0.8, "Rate of decay for learning rate.")
flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.")

flags.DEFINE_integer("num_epochs", 50, "number of epochs")
flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
flags.DEFINE_integer("checkpoint_every", 2, "Save model after this many steps (default: 100)")

flags.DEFINE_string("num_filters", "256,256,256", "Number of filters per filter size (default: 128)")
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_float("keep_prob", 0.5, "dropout keep prob")
flags.DEFINE_float("word_drop", 0.5, "dropout word prob")
#data settings
flags.DEFINE_boolean("use_embedding", True, "whether to use pretrained embedding or not.")
flags.DEFINE_integer("max_sentence_length", 512, "the max sentence length")
flags.DEFINE_integer("word_embed_size", 300, "word embedding size")
flags.DEFINE_integer("hidden_dim", 1000, "hidden layer dimension")
flags.DEFINE_integer("K", 3, "k largest logits")
flags.DEFINE_boolean("is_training", False, "is traning.true:tranining,false:testing/inference")

#filepath settings
flags.DEFINE_string("data_dir", r"/home/yaojq/data/text/reuters", "data file path")
flags.DEFINE_string("data_name", r"reuters", "data file name")
flags.DEFINE_string("log_dir", r"/home/yaojq/data/log/log-reuters-xmlCNN", "checkpoint location for the model")
flags.DEFINE_string("word2vec_path", r"/home/yaojq/data/word_embedding/GoogleNews-vectors-negative300.bin", "pretrained word2vec bin")


FLAGS = tf.app.flags.FLAGS
