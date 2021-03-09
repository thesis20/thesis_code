import os
import random
import numpy as np
import tensorflow as tf
from evaluation import evaluator
from LoadData import LoadMovieLens
from tensorflow.python.client import device_lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']

class CSGCN():
    def __init__(self, emb_dim, epochs, n_layers, batch_size, learning_rate, seed, ks):
        self.random_seed = seed
        self.data = LoadMovieLens(random_seed=self.random_seed)
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ks = eval(ks)
        self.use_l2 = True
        self.weights = self._init_weights()
        self.evaluator = evaluator()
        
        self.weights = self._init_weights()
        
    
    def _init_weights(self):
        # TODO: n_context, n_user_sideinfo, n_item_sideinfo
        
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        
        all_weights['user_embedding'] = tf.Variable(initializer([self.data.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.data.n_items, self.emb_dim]),
                                                    name='item_embedding')
        all_weights['context_embedding'] = tf.Variable(initializer([self.data.n_context, self.emb_dim]),
                                                    name='context_embedding')
        all_weights['user_sideinfo_embedding'] = tf.Variable(initializer([self.data.n_user_sideinfo, self.emb_dim]),
                                                    name='user_sideinfo_embedding')
        all_weights['item_sideinfo_embedding'] = tf.Variable(initializer([self.data.n_item_sideinfo, self.emb_dim]),
                                                    name='item_sideinfo_embedding')

    def _init_graph(self):
        self.graph = tf.Graph()
            
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            
            self.users = tf.placeholder(tf.int32, shape=[None, None])
            self.pos_interactions = tf.placeholder(tf.int32, shape=[None, None])
            self.neg_interactions = tf.placeholder(tf.int32, shape=[None, None])
            self.context = tf.placeholder(tf.int32, shape=[None, None])
            self.user_sideinfo = tf.placeholder(tf.int32, shape=[None, None])
            self.item_sideinfo = tf.placeholder(tf.int32, shape=[None, None])
            
            self.weights = self._init_weights()
            
            self._csgcn_layers()
            
            self.user_embeddings = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
            self.pos_interactions_embeddings = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_interactions)
            self.neg_interactions_embeddings = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_interactions)
            self.context_embeddings = tf.nn.embedding_lookup(self.weights['context_embedding'], self.context)
            self.user_sideinfo_embeddings = tf.nn.embedding_lookup(self.weights['user_sideinfo_embedding'], self.user_sideinfo)
            self.item_sideinfo_embeddings = tf.nn.embedding_lookup(self.weights['item_sideinfo_embedding'], self.item_sideinfo)
            
            self.loss = _bpr_loss(self.user_embeddings, self.pos_interactions_embeddings, self.neg_interactions_embeddings)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            
    
    def _csgcn_layers(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.data.adj_mat)
        
        embs = tf.concat([self.weights['user_embedding'], self.weights['item_embedding'], self.weights['context_embedding']])
        matmul = tf.sparse_tensor_dense_matmul(adj_mat, embs)
        
        for k in range(0, self.n_layers):
            # TODO: Implemnter convolution formel, overvej at følge LGCN med NuNi
            continue
        
    def _bpr_loss(self, users, pos_items, neg_items):
        # TODO: Skal den her ikke lige have noget side info og context
        loss = -tf.log(tf.sigmoid(pos_items - neg_items))
        if self.use_l2:
            loss = tf.nn.l2_loss(loss)
        return tf.reduce_sum(loss)
    
    def _convert_sp_mat_to_sp_tensor(self, adj_mat):
        coo = adj_mat.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
    
    def train(self):
        for epoch in range(0, self.epochs):
            total_loss = 0
            batch = self.data.sampler()
            for interaction in range(0, len(batch)):
                total_loss += loss
            
            if epoch > 0 and epoch % 100 == 0:
                print(f"The total loss in {epoch}th iteration is: {total_loss}")
                self.evaluate(epoch)
        
    def evaluate(self):
        for k in self.ks:
            scores = dict()
            for user in self.data.test_df['userId'].unique():
                continue

        self.evaluator.evaluate(scores, self.data.user_ground_truth_dict, k, epoch)
        
        
model = CSGCN(emb_dim=64, epochs=1000, n_layers=4, batch_size=95, learning_rate=0.001, seed=2021, ks='[10,20]')
model.train()
