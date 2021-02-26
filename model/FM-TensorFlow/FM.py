import tensorflow as tf
from LoadData import LoadMovieLens

class FM():
    def __init__(self, emb_dim, epochs, batch_size, learning_rate):
        self.data = LoadMovieLens()
        self.emb_dim = emb_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = 2021
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            # Input data
            self.user_features = tf.placeholder(tf.int32, shape=[None, None])
            self.item_features = tf.placeholder(tf.item32, shape=[None, None])

            # Weights
            self.weights = self._init_weights()

            # Initialize embedding lookups (Fancy dictionaries)
            self.user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'],
                                                                  self.user_features)
            self.item_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'],
                                                                  self.item_features)

            # Calculate second part of rewritten FM equation
            self.summed_user_emb = tf.reduce_sum(self.user_feature_embeddings, 1)
            self.summed_item_emb = tf.reduce_sum(self.item_feature_embeddings, 1)
            self.summed_ui_emb = tf.add(self.summed_user_emb, self.summed_item_emb)
            self.summed_ui_emb_square = tf.square(self.summed_ui_emb)

            # Calculate first part of rewritten FM equation
            self.squared_user_emb = tf.square(self.user_feature_embeddings)
            self.squared_item_emb = tf.square(self.item_feature_embeddings)
            self.squared_user_emb_sum = tf.reduce_sum(self.squared_user_emb, 1)
            self.squared_item_emb_sum = tf.reduce_sum(self.squared_item_emb, 1)
            self.squared_ui_emb_sum = tf.add(self.squared_user_emb_sum, self.squared_item_emb_sum)
            

    def _init_weights(self):
        all_weights = {}

        all_weights['user_feature_embeddings'] = tf.Variable(
            tf.random_normal([self.data.user_fields, self.emb_dim], 0.0, 0.1),
            name='user_feature_embeddings')
        all_weights['item_feature_embeddings'] = tf.Variable(
            tf.random_normal([self.data.item_fields, self.emb_dim], 0.0, 0.1),
            name='item_feature_embeddings')
        all_weights['user_feature_bias'] = tf.Variable(
            tf.random_uniform([self.user_fields, 1], 0.0, 0.1, name='user_feature_bias'))
        all_weights['item_feature_bias'] = tf.Variable(
            tf.random_uniform([self.item_fields, 1], 0.0, 0.1, name='item_feature_bias'))
        return all_weights

