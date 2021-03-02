import tensorflow as tf
from LoadData import LoadMovieLens
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class FM():
    def __init__(self, emb_dim, epochs, batch_size, learning_rate):
        self.data = LoadMovieLens()
        self.emb_dim = emb_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = 2021
        self._init_graph()
        random.seed(self.random_seed)

    def _init_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
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
            print("i am initializing")
            
            self.user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'], self.user_features)
            self.pos_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'], self.pos_features)
            self.neg_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'], self.neg_features)
            self.user_feature_bias_sum = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['user_feature_bias'], self.user_features), 1)
            self.pos_item_feature_bias_sum = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.pos_features), 1)
            self.neg_item_feature_bias_sum = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.neg_features), 1)

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

            self.loss_func = -tf.log(tf.sigmoid(self.pos_y_hat - self.neg_y_hat))
            self.loss_func = tf.reduce_sum(self.loss_func)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_func)
            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss_func)
            # self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(self.loss_func)

            # self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def sampler(self):
        # TODO: Refactor this
        def _get_genre_indexes(row):
            genre_indexes = []
            genres = ['unknown', 'action', 'adventure', 'animation',
                        'childrens', 'comedy', 'crime', 'documentary',
                        'drama', 'fantasy',  'film-noir', 'horror',
                        'musical', 'mystery', 'romance', 'scifi',
                        'thriller', 'war', 'western']
            for genre in genres:
                if row[genre].to_numpy()[0] == 1:
                    genre_indexes.append(self.data.item_feature_dict[genre + str(1)])
            return genre_indexes

        user_ids, pos_interactions, neg_interactions = [], [], []
        while len(user_ids) < self.batch_size:
            rand_row = self.data.train_df.sample(random_state=42)
            userId = rand_row['userId'].to_numpy()[0]
            userId_index = self.data.user_feature_dict['userId' + str(userId)]
            age_index = self.data.user_feature_dict['age' + str(rand_row['age'].to_numpy()[0])]
            gender_index = self.data.user_feature_dict['gender' + str(rand_row['gender'].to_numpy()[0])]
            occupation_index = self.data.user_feature_dict['occupation' + str(rand_row['occupation'].to_numpy()[0])]
            zipcode_index = self.data.user_feature_dict['zipcode' + str(rand_row['zipcode'].to_numpy()[0])]
            movieId = rand_row['movieId'].to_numpy()[0]
            movieId_index = self.data.item_feature_dict['movieId' + str(rand_row['movieId'].to_numpy()[0])]


            user = [userId_index, age_index, gender_index, occupation_index, zipcode_index]
            user_ids.append(user)
            movie = [movieId_index] + _get_genre_indexes(rand_row)
            pos_interactions.append(movie)

            # Sample negative item
            user_pos_items = self.data.train_df[self.data.train_df['userId'] == userId]['movieId'].unique()
            
            sample = self.data.train_df.sample(random_state=42)
            while sample.to_numpy()[0] in user_pos_items:
                sample = self.data.train_df.sample(random_state=42)
            
            neg_movie_index = self.data.item_feature_dict['movieId' + str(sample['movieId'].to_numpy()[0])]
            neg_item = [neg_movie_index] + _get_genre_indexes(sample)
            neg_interactions.append(neg_item)

        # Add padding to genre lists to make sure all movies have the same amount of genres (padded)
        longest_genre_list = max(len(max(neg_interactions)), len(max(pos_interactions)))
        
        for index, pos_int in enumerate(pos_interactions):
            pos_interactions[index] = (pos_interactions[index] + longest_genre_list * [0])[:longest_genre_list]

        for index, neg_int in enumerate(neg_interactions):
            neg_interactions[index] = (neg_interactions[index] + longest_genre_list * [0])[:longest_genre_list]

        return { 'user_ids': user_ids, 'pos_interactions': pos_interactions, 'neg_interactions': neg_interactions }

    
    def train(self):
        for epoch in range(0, self.epochs):
            total_loss = 0
            total_batch = int(self.data.n_users / self.batch_size)

            
            for i in range(total_batch):
                batch = self.sampler()
                feed_dict = {self.user_features: batch['user_ids'],
                            self.pos_features: batch['pos_interactions'],
                            self.neg_features: batch['neg_interactions']}
                batch_loss, _ = self.sess.run((self.loss_func, self.optimizer), feed_dict=feed_dict)
                total_loss += batch_loss
            print(f"the total loss in {epoch}th iteration is: {batch_loss}")

fm = FM(emb_dim=64, epochs=1000, batch_size=95, learning_rate=0.01)
fm.train()
