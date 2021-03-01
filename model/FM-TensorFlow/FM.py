import tensorflow as tf
from LoadData import LoadMovieLens
import random

class FM():
    def __init__(self, emb_dim, epochs, batch_size, learning_rate):
        tf.enable_eager_execution()
        self.data = LoadMovieLens()
        self.emb_dim = emb_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_seed = 2021
        self.optimizer = tf.train.AdamOptimizer()
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            if self.random_seed is not None:
                tf.set_random_seed(self.random_seed)
            self.X = tf.sparse_placeholder(dtype=tf.float32, shape=[None, None])
            self.y = tf.placeholder(dtype=tf.float32)

            self.user_features = tf.placeholder(tf.int32, shape=[None, None])
            self.positive_features = tf.placeholder(tf.int32, shape=[None, None])
            self.negative_features = tf.placeholder(tf.int32, shape=[None, None])

            # w is amount of features (input dimension) by output dimension
            self.w = tf.Variable(tf.truncated_normal([self.data.field_sizes, 1], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32)
            self.v = tf.Variable(tf.truncated_normal([self.data.field_sizes, self.emb_dim], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32)
            # w0 or b (global bias)
            self.w0 = tf.Variable(tf.zeros(1, dtype=tf.float32), dtype=tf.float32)
            
            self.user_feature_embeddings = tf.nn.embedding_lookup(self.v, self.user_features)
            self.positive_feature_embeddings = tf.nn.embedding_lookup(self.v, self.positive_features)
            self.negative_feature_embeddings = tf.nn.embedding_lookup(self.v, self.negative_features)
            self.user_feature_bias = tf.nn.embedding_lookup(self.w, self.user_features)
            self.pos_item_feature_bias = tf.nn.embedding_lookup(self.w, self.positive_features)
            self.neg_item_feature_bias = tf.nn.embedding_lookup(self.w, self.negative_features)

            # POsitive item
            positive_embeddings = tf.concat([self.user_feature_embeddings, self.positive_feature_embeddings], -1)
            pos_first_term = tf.square(positive_embedding)
            pos_second_term = tf.sparse_tensor_dense_matmul(tf.square(self.X), tf.square(positive_embedding))

            pos_pred = 0.5 * tf.reduce_sum(tf.subtract(pos_first_term, pos_second_term), 1)

            pos_y_hat = self.w0 + tf.sparse_tensor_dense_matmul(self.X, self.w) + pos_pred

            # NEgative item
            negative_embedding = tf.concat([self.user_feature_embeddings, self.negative_feature_embeddings], -1)
            neg_first_term = tf.square(tf.sparse_tensor_dense_matmul(self.X, negative_embedding))
            neg_second_term = tf.sparse_tensor_dense_matmul(tf.square(self.X), tf.square(negative_embedding))

            neg_pred = 0.5 * tf.reduce_sum(tf.subtract(neg_first_term, neg_second_term), 1)

            neg_y_hat = self.w0 + tf.sparse_tensor_dense_matmul(self.X, self.w) + neg_pred

            self.loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos_y_hat - neg_y_hat)))
            
            self.saver = tf.train.Saver()
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
                    genre_indexes.append(self.data.lookup_dict[genre + str(1)])
            return genre_indexes

        user_ids, pos_interactions, neg_interactions = [], [], []
        while len(user_ids) < self.batch_size:
            rand_row = self.data.train_df.sample()
            userId = rand_row['userId'].to_numpy()[0]
            userId_index = self.data.lookup_dict['userId' + str(userId)]
            age_index = self.data.lookup_dict['age' + str(rand_row['age'].to_numpy()[0])]
            gender_index = self.data.lookup_dict['gender' + str(rand_row['gender'].to_numpy()[0])]
            occupation_index = self.data.lookup_dict['occupation' + str(rand_row['occupation'].to_numpy()[0])]
            zipcode_index = self.data.lookup_dict['zipcode' + str(rand_row['zipcode'].to_numpy()[0])]
            movieId = rand_row['movieId'].to_numpy()[0]
            movieId_index = self.data.lookup_dict['movieId' + str(rand_row['movieId'].to_numpy()[0])]


            user = [userId_index, age_index, gender_index, occupation_index, zipcode_index]
            user_ids.append(user)
            movie = [movieId_index] + _get_genre_indexes(rand_row)
            pos_interactions.append(movie)

            # Sample negative item
            user_pos_items = self.data.train_df[self.data.train_df['userId'] == userId]['movieId'].unique()
            
            sample = self.data.train_df.sample()
            while sample.to_numpy()[0] in user_pos_items:
                sample = self.data.train_df.sample()
            
            neg_movie_index = self.data.lookup_dict['movieId' + str(sample['movieId'].to_numpy()[0])]
            neg_item = [neg_movie_index] + _get_genre_indexes(sample)

        return { 'user_ids': user_ids, 'pos_interaction': pos_interactions, 'neg_interaction': neg_interactions }

    def bpr_loss(user, pos, neg):
        raise NotImplementedError
    
    def train(self):
        for epoch in range(1, self.epochs + 1):
            loss = 0.0
            batch = self.sampler()
            feed_dict = {self.user_features: batch['user_ids'],
                         self.positive_features: batch['pos_interactions'],
                         self.negative_features: batch['neg_interactions']}
            batch_loss, opt = self.sess.run({self.loss, self.optimizer}, feed_dict=feed_dict)
            loss += batch_loss
            logger.info("the total loss in %d th iteration is: %f" % (epoch, loss))

fm = FM(emb_dim=64, epochs=10, batch_size=8, learning_rate=0.001)
