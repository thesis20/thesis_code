import tensorflow as tf
import numpy as np

def _dropout_sparse(X, keep_prob, n_nonzero_elems):
    """
    Dropout for sparse tensors.
    """
    noise_shape = [n_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(X, dropout_mask)

    return pre_out * tf.math.divide(1., keep_prob)

def _split_A_hat_node_dropout(self, X):
    A_fold_hat = []

    fold_len = (self.n_users + self.n_items) // self.n_fold
    for i_fold in range(self.n_fold):
        start = i_fold * fold_len
        if i_fold == self.n_fold -1:
            end = self.n_users + self.n_items
        else:
            end = (i_fold + 1) * fold_len

        temp = _convert_sp_mat_to_sp_tensor(X[start:end])
        n_nonzero_temp = X[start:end].count_nonzero()
        if self.node_dropout[0] is None:
            A_fold_hat.append(_dropout_sparse(temp, 1, n_nonzero_temp))
        else:
            A_fold_hat.append(_dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

    return A_fold_hat

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def _create_lightgcn_embed(self):
    A_fold_hat = _split_A_hat_node_dropout(self, self.norm_adj)
    
    ego_embeddings = tf.concat([self.my_weights['user_embedding'], self.my_weights['item_embedding']], axis=0)
    all_embeddings = [ego_embeddings]
    
    for k in range(0, self.n_layers):
        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(tf.sparse.sparse_dense_matmul(A_fold_hat[f], ego_embeddings))

        side_embeddings = tf.concat(temp_embed, 0)
        ego_embeddings = side_embeddings
        all_embeddings += [ego_embeddings]
    all_embeddings=tf.stack(all_embeddings,1)
    all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
    u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
    return u_g_embeddings, i_g_embeddings

# TODO: Hvad fuck er gc, bi og mlp (Bruger vi alle tre lag?)
def _init_weights(self):
    all_weights = dict()
    initializer = tf.random_normal_initializer(stddev=0.01) #tf.contrib.layers.xavier_initializer()
    all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
    all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
        
    self.weight_size_list = [self.emb_dim] + self.weight_size
    
    for k in range(self.n_layers):
        all_weights['W_gc_%d' %k] = tf.Variable(
            initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
        all_weights['b_gc_%d' %k] = tf.Variable(
            initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

        all_weights['W_bi_%d' % k] = tf.Variable(
            initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
        all_weights['b_bi_%d' % k] = tf.Variable(
            initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

        all_weights['W_mlp_%d' % k] = tf.Variable(
            initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
        all_weights['b_mlp_%d' % k] = tf.Variable(
            initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)

    return all_weights
