import tensorflow as tf
from LoadData import LoadMovieLens
import random
import os
import numpy as np
from evaluation import evaluator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FM():
    def __init__(self, emb_dim, epochs, batch_size, learning_rate, topk, savefile_path):
        self.use_dropout = True
        self.random_seed = 2016
        self.data = LoadMovieLens()
        self.emb_dim = emb_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._init_graph()
        random.seed(self.random_seed)
        self.topk = topk
        self.evaluator = evaluator()
        self.savefile_path = savefile_path
        

    def _init_graph(self):
        
        self.graph = tf.Graph()
           
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
                
            self.user_features = tf.placeholder(tf.int32, shape=[None, None])
            self.pos_features = tf.placeholder(tf.int32, shape=[None, None])
            self.neg_features = tf.placeholder(tf.int32, shape=[None, None])

            # w is amount of features (input dimension) by output dimension
            self.weights = dict()
            self.weights['user_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.data.user_fields, self.emb_dim], 0.0, 0.1),
                name='user_feature_embeddings')
            self.weights['item_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.data.item_fields, self.emb_dim], 0.0, 0.1),
                name='item_feature_embeddings')
            self.weights['user_feature_bias'] = tf.Variable(
                tf.random_uniform([self.data.user_fields, 1], 0.0, 0.1),
                name='user_feature_bias')
            self.weights['item_feature_bias'] = tf.Variable(
                tf.random_uniform([self.data.item_fields, 1], 0.0, 0.1),
                name='item_feature_bias')
            
            self.user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'],
                                                                  self.user_features)
            self.pos_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'],
                                                                 self.pos_features)
            self.neg_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'],
                                                                 self.neg_features)
            self.user_feature_bias_sum = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['user_feature_bias'],
                                                                              self.user_features), 1)
            self.pos_item_feature_bias_sum = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['item_feature_bias'],
                                                                                  self.pos_features), 1)
            self.neg_item_feature_bias_sum = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['item_feature_bias'],
                                                                                  self.neg_features), 1)

            # Positive item
            self.user_embeddings_sum = tf.reduce_sum(self.user_feature_embeddings, 1)
            self.pos_item_embeddings_sum = tf.reduce_sum(self.pos_feature_embeddings, 1)
            self.pos_embeddings_sum = tf.add(self.user_embeddings_sum, self.pos_item_embeddings_sum)
            self.pos_first_term = tf.square(self.pos_embeddings_sum)
            
            self.user_embeddings_sq = tf.square(self.user_feature_embeddings)
            self.pos_item_embeddings_sq = tf.square(self.pos_feature_embeddings)
            self.user_embeddings_sq_sum = tf.reduce_sum(self.user_embeddings_sq, 1)
            self.pos_item_embeddings_sq_sum = tf.reduce_sum(self.pos_item_embeddings_sq, 1)
            self.pos_second_term = tf.add(self.user_embeddings_sq_sum, self.pos_item_embeddings_sq_sum)

            self.pos_pred = 0.5 * tf.subtract(self.pos_first_term, self.pos_second_term)
            if self.use_dropout:
                self.pos_pred = tf.nn.dropout(self.pos_pred, 0.8)
            
            self.bipos = tf.reduce_sum(self.pos_pred, 1, keepdims=True)

            self.pos_y_hat = tf.add_n([self.bipos, self.user_feature_bias_sum, self.pos_item_feature_bias_sum])

            # Negative item
            self.neg_item_embeddings_sum = tf.reduce_sum(self.neg_feature_embeddings, 1)
            self.neg_embeddings_sum = tf.add(self.user_embeddings_sum, self.neg_item_embeddings_sum)
            self.neg_first_term = tf.square(self.neg_embeddings_sum)
            
            self.neg_item_embeddings_sq = tf.square(self.neg_feature_embeddings)
            self.neg_item_embeddings_sq_sum = tf.reduce_sum(self.neg_item_embeddings_sq, 1)
            self.neg_second_term = tf.add(self.user_embeddings_sq_sum, self.neg_item_embeddings_sq_sum)
            
            self.neg_pred = 0.5 * tf.subtract(self.neg_first_term, self.neg_second_term)
            if self.use_dropout:
                self.neg_pred = tf.nn.dropout(self.neg_pred, 0.8)
            
            self.bineg = tf.reduce_sum(self.neg_pred, 1, keepdims=True)

            self.neg_y_hat = tf.add_n([self.bineg, self.user_feature_bias_sum, self.neg_item_feature_bias_sum])
            #self.neg_y_hat = tf.nn.dropout(self.neg_y_hat, 0.5)

            self.loss = -tf.log(tf.sigmoid(self.pos_y_hat - self.neg_y_hat))
            self.loss = tf.reduce_sum(self.loss)

            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            
    def sampler(self):
        # TODO: Refactor this

        user_ids, pos_interactions, neg_interactions = [], [], []
        while len(user_ids) < self.batch_size:
            # random user
            # random pos item
            # random neg item
            
            userId = random.choice(list(self.data.train_set_user_pos_interactions.keys()))
            pos = random.sample(self.data.train_set_user_pos_interactions[userId], 1)[0]
            neg = random.sample(self.data.train_set_user_neg_interactions[userId], 1)[0]
            
            user_ids.append(self.data.user_feature_dict[userId])
            pos_interactions.append(self.data.item_feature_dict[pos])
            neg_interactions.append(self.data.item_feature_dict[neg])

        return { 'user_ids': user_ids, 'pos_interactions': pos_interactions, 'neg_interactions': neg_interactions }

    def partial_fit(self, data):
        feed_dict = {self.user_features: data['user_ids'], self.pos_features: data['pos_interactions'],
                     self.neg_features: data['neg_interactions']}
        return self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)

    def train(self):
        for epoch in range(0, self.epochs):
            total_loss = 0
            total_batch = int(self.data.n_interactions / self.batch_size)
            for i in range(total_batch):
                batch = self.sampler()
                loss, _ = self.partial_fit(batch)
                total_loss += loss
            print(f"the total loss in {epoch}th iteration is: {total_loss}")

            if epoch > 0 and epoch % 100 == 0:
                self.evaluate()
        self.saver.save(self.sess, self.savefile_path)
    
    def evaluate(self):
        self.graph.finalize()

        scores = dict()
        for user in self.data.test_df['userId'].unique():
            user_features = self.data.user_feature_dict[user]
            item_features_list = []
            user_features_list = []
            for item in self.data.test_df['movieId'].unique():
                item_features = self.data.item_feature_dict[item]
                item_features_list.append(item_features)
                user_features_list.append(user_features)
                
            feed_dict = {self.user_features: user_features_list, self.pos_features: item_features_list}
            pos_scores = self.sess.run((self.pos_y_hat), feed_dict=feed_dict)
            pos_scores = pos_scores.reshape(self.data.n_test_items)
            scores[user] = pos_scores
        
        self.evaluator.evaluate(scores, self.data.user_ground_truth_dict, self.topk)


emb_dim=64
batch_size=95
lr = 0.1
epochs=1000
path = 'pretrain-FM-%s-emb%d-lr%d-bs%d-e%d' % ('movielens100k', emb_dim, lr, batch_size, epochs)
fm = FM(emb_dim=emb_dim, epochs=epochs, batch_size=batch_size, learning_rate=lr, topk=20, savefile_path=path)
fm.train()
