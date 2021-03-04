import tensorflow as tf
from LoadData import LoadMovieLens
import random
import os
import numpy as np
from evaluation import evaluator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FM():
    def __init__(self, emb_dim, epochs, batch_size, learning_rate, topk):
        self.data = LoadMovieLens()
        self.emb_dim = emb_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = 2021
        self._init_graph()
        random.seed(self.random_seed)
        self.topk = topk
        self.evaluator = evaluator()
        
        

    def _init_graph(self):   
        self.graph = tf.Graph()
           
        with self.graph.as_default() as graphyboi:
            if self.random_seed is not None:
                tf.set_random_seed(self.random_seed)
                
            self.user_features = tf.placeholder(tf.int32, shape=[None, None])
            self.pos_features = tf.placeholder(tf.int32, shape=[None, None])
            self.neg_features = tf.placeholder(tf.int32, shape=[None, None])

            # w is amount of features (input dimension) by output dimension
            self.weights = dict()
            self.weights['user_feature_bias'] = tf.Variable(
                tf.random_uniform([self.data.user_fields, 1], 0.0, 0.1),
                name='user_feature_bias')
            self.weights['item_feature_bias'] = tf.Variable(
                tf.random_uniform([self.data.item_fields, 1], 0.0, 0.1),
                name='item_feature_bias')
            self.weights['user_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.data.user_fields, self.emb_dim], 0.0, 0.1),
                name='user_feature_embeddings')
            self.weights['item_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.data.item_fields, self.emb_dim], 0.0, 0.1),
                name='item_feature_embeddings')
            
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

            # POsitive item
            self.user_embeddings_sum = tf.reduce_sum(self.user_feature_embeddings, 1)
            self.pos_item_embeddings_sum = tf.reduce_sum(self.pos_feature_embeddings, 1)
            self.pos_embeddings_sum = tf.add(self.user_embeddings_sum, self.pos_item_embeddings_sum)
            self.pos_first_term = tf.square(self.pos_embeddings_sum)
            
            self.user_embeddings_sq = tf.square(self.user_feature_embeddings)
            self.user_embeddings_sq_sum = tf.reduce_sum(self.user_embeddings_sq, 1)
            self.pos_item_embeddings_sq = tf.square(self.pos_feature_embeddings)
            self.pos_item_embeddings_sq_sum = tf.reduce_sum(self.pos_item_embeddings_sq, 1)
            self.pos_second_term = tf.add(self.user_embeddings_sq_sum, self.pos_item_embeddings_sq_sum)

            self.pos_pred = 0.5 * tf.subtract(self.pos_first_term, self.pos_second_term)
            self.bipos = tf.reduce_sum(self.pos_pred, 1, keepdims=True)

            self.pos_y_hat = tf.add_n([self.bipos, self.user_feature_bias_sum, self.pos_item_feature_bias_sum])
            #self.pos_y_hat = tf.nn.dropout(self.pos_y_hat, 0.5)

            # NEgative item
            self.neg_item_embeddings_sum = tf.reduce_sum(self.neg_feature_embeddings, 1)
            self.neg_embeddings_sum = tf.add(self.user_embeddings_sum, self.neg_item_embeddings_sum)
            self.neg_first_term = tf.square(self.neg_embeddings_sum)
            
            self.neg_item_embeddings_sq = tf.square(self.neg_feature_embeddings)
            self.neg_item_embeddings_sq_sum = tf.reduce_sum(self.neg_item_embeddings_sq, 1)
            self.neg_second_term = tf.add(self.user_embeddings_sq_sum, self.neg_item_embeddings_sq_sum)
            
            self.neg_pred = 0.5 * tf.subtract(self.neg_first_term, self.neg_second_term)
            self.bineg = tf.reduce_sum(self.neg_pred, 1, keepdims=True)

            self.neg_y_hat = tf.add_n([self.bineg, self.user_feature_bias_sum, self.neg_item_feature_bias_sum])
            #self.neg_y_hat = tf.nn.dropout(self.neg_y_hat, 0.5)

            self.loss = -tf.log(tf.sigmoid(self.pos_y_hat - self.neg_y_hat))
            self.loss = tf.reduce_sum(self.loss)

           # self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            # self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            
    def sampler(self):
        # TODO: Refactor this
        def _get_genre_indexes(row):
            genre_indexes = []
            genres = self.data.genrelist
            for genre in genres:
                if np.array(row[genre])[0] == 1:
                    genre_indexes.append(self.data.item_feature_index_dict[genre + str(1)])
            return genre_indexes

        user_ids, pos_interactions, neg_interactions = [], [], []
        while len(user_ids) < self.batch_size:
            random_value = random.randint(0, self.data.n_interactions)
            rand_row = self.data.train_df.sample(random_state=random_value)
            userId = np.array(rand_row['userId'])[0]
            userId_index = self.data.user_feature_index_dict['userId' + str(userId)]
            age_index = self.data.user_feature_index_dict['age' + str(np.array(rand_row['age'])[0])]
            gender_index = self.data.user_feature_index_dict['gender' + str(np.array(rand_row['gender'])[0])]
            occupation_index = self.data.user_feature_index_dict['occupation' + str(np.array(rand_row['occupation'])[0])]
            zipcode_index = self.data.user_feature_index_dict['zipcode' + str(np.array(rand_row['zipcode'])[0])]
            movieId = np.array(rand_row['movieId'])[0]
            movieId_index = self.data.item_feature_index_dict['movieId' + str(np.array(rand_row['movieId'])[0])]


            user = [userId_index, age_index, gender_index, occupation_index, zipcode_index]
            user_ids.append(user)
            movie = [movieId_index] + _get_genre_indexes(rand_row)
            pos_interactions.append(movie)

            # Sample negative item
            user_pos_items = self.data.train_df[self.data.train_df['userId'] == userId]['movieId'].unique()
            
            random_sample_value = random.randint(0, self.data.n_interactions)
            sample = self.data.train_df.sample(random_state=random_sample_value)
            
            while np.array(sample['movieId'])[0] in user_pos_items:
                random_sample_value = random.randint(0, self.data.n_interactions)
                sample = self.data.train_df.sample(random_state=random_sample_value)
            
            neg_movie_index = self.data.item_feature_index_dict['movieId' + str(np.array(sample['movieId'])[0])]
            neg_item = [neg_movie_index] + _get_genre_indexes(sample)
            neg_interactions.append(neg_item)

        # Add padding to genre lists to make sure all movies have the same amount of genres (padded)
        longest_genre_list = max(len(max(neg_interactions)), len(max(pos_interactions)))
        
        for index, pos_int in enumerate(pos_interactions):
            pos_interactions[index] = (pos_interactions[index] + longest_genre_list * [0])[:longest_genre_list]

        for index, neg_int in enumerate(neg_interactions):
            neg_interactions[index] = (neg_interactions[index] + longest_genre_list * [0])[:longest_genre_list]

        return { 'user_ids': user_ids, 'pos_interactions': pos_interactions, 'neg_interactions': neg_interactions }

    def partial_fit(self, data):
        feed_dict = {self.user_features: data['user_ids'], self.pos_features: data['pos_interactions'],
                     self.neg_features: data['neg_interactions']}
        return self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)

    def train(self):
        for epoch in range(0, self.epochs):
            total_loss = 0
            total_batch = int(self.data.n_users / self.batch_size)
            for i in range(total_batch):
                batch = self.sampler()
                loss, _ = self.partial_fit(batch)
                total_loss += loss
            print(f"the total loss in {epoch}th iteration is: {total_loss}")

            if epoch % 10 == 0:
                self.evaluate_now()
            
    def evaluate_now(self):
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

        

fm = FM(emb_dim=64, epochs=1000, batch_size=95, learning_rate=0.01, topk=20)
fm.train()
