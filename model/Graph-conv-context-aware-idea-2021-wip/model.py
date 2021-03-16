import os
import numpy as np
import tensorflow as tf
from evaluation import evaluator
from LoadData import LoadMovieLens
from tensorflow.python.client import device_lib
import pickle
from utility.parser import parse_args
from collections import defaultdict
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']
args = parse_args()


class CSGCN():
    def __init__(self, sess, data):
        self.random_seed = args.seed
        random.seed(self.random_seed)
        self.decay = args.decay
        self.data = data
        print("Loaded data")
        self.n_layers = args.layers
        self.emb_dim = args.embed_size
        self.epochs = args.epoch
        self.batch_size = args.batch
        self.learning_rate = args.lr
        self.initializer = self._set_initializer(args.initializer)
        self.optimizer = self._set_optimizer(args.optimizer)
        self.ks = eval(args.ks)
        self.evaluator = evaluator()
        self.sess = sess
        self._init_graph()
        print("Initialized graph")


        with tf.name_scope('TRAIN_LOSS'):
            self.train_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_loss', self.train_loss)
            self.train_mf_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_mf_loss', self.train_mf_loss)
            self.train_emb_loss = tf.placeholder(tf.float32)
            tf.summary.scalar('train_emb_loss', self.train_emb_loss)
            self.train_reg_loss = tf.placeholder(tf.float32)
        self.merged_train_loss = tf.summary.merge(
            tf.get_collection(tf.GraphKeys.SUMMARIES, 'TRAIN_LOSS'))

        with tf.name_scope('TEST_ACC'):
            self.test_hr_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_hr_first', self.test_hr_first)
            self.test_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_first', self.test_ndcg_first)
            self.test_mrr_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_mrr_first', self.test_mrr_first)
            self.test_hr_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_hr_last', self.test_hr_last)
            self.test_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_last', self.test_ndcg_last)
            self.test_mrr_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_mrr_last', self.test_mrr_last)
        self.merged_test_acc = tf.summary.merge(
            tf.get_collection(tf.GraphKeys.SUMMARIES, 'TEST_ACC'))

    def _set_optimizer(self, optimizer):
        if optimizer == 'adam':
            return tf.train.AdamOptimizer(self.learning_rate)
        elif optimizer == 'adagrad':
            return tf.train.AdagradOptimizer(self.learning_rate)
        elif optimizer == 'RMSProp':
            return tf.train.RMSPropOptimizer(self.learning_rate)
        elif optimizer == 'Adadelta':
            return tf.train.AdadeltaOptimizer(self.learning_rate)
        else:
            raise Exception("No optimizer set")

    def _set_initializer(self, initializer):
        if initializer == 'normal':
            return tf.random_normal_initializer(seed=self.random_seed, stddev=0.01)
        elif initializer == 'xavier':
            return tf.contrib.layers.xavier_initializer(seed=self.random_seed)
        elif initializer == 'glorot':
            return tf.glorot_uniform_initializer(seed=self.random_seed)
        elif initializer == 'glorot_normal':
            return tf.glorot_normal_initializer(seed=self.random_seed)
        else:
            raise Exception("No initializer set")

    def _init_weights(self):
        # TODO: n_context, n_user_sideinfo, n_item_sideinfo

        all_weights = dict()

        all_weights['user_embedding'] = tf.Variable(self.initializer([self.data.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(self.initializer([self.data.n_items, self.emb_dim]),
                                                    name='item_embedding')
        all_weights['context_embedding'] = tf.Variable(self.initializer([self.data.n_context, self.emb_dim]),
                                                       name='context_embedding')
        all_weights['user_sideinfo_embedding'] = tf.Variable(self.initializer([self.data.n_user_sideinfo, self.emb_dim]),
                                                             name='user_sideinfo_embedding')
        all_weights['item_sideinfo_embedding'] = tf.Variable(self.initializer([self.data.n_item_sideinfo, self.emb_dim]),
                                                             name='item_sideinfo_embedding')

        # Biases
        all_weights['user_bias'] = tf.Variable(
            tf.zeros([self.data.n_users], dtype=tf.float32, name='user_bias'))
        all_weights['item_bias'] = tf.Variable(
            tf.zeros([self.data.n_items], dtype=tf.float32, name='item_bias'))

        return all_weights

    def _init_graph(self):
        tf.set_random_seed(self.random_seed)

        self.users = tf.placeholder(tf.int32, shape=[None, None])
        self.pos_interactions = tf.placeholder(tf.int32, shape=[None, None])
        self.neg_interactions = tf.placeholder(tf.int32, shape=[None, None])
        self.context = tf.placeholder(tf.int32, shape=[None, None])
        self.user_sideinfo = tf.placeholder(tf.int32, shape=[None, None])
        self.item_sideinfo = tf.placeholder(tf.int32, shape=[None, None])

        self.weights = self._init_weights()

        self.user_embs, self.item_embs, self.context_embs = self._csgcn_layers()

        # Learnable weights
        self.user_embeddings = tf.nn.embedding_lookup(
            self.user_embs, self.users)
        self.pos_interactions_embeddings = tf.nn.embedding_lookup(
            self.item_embs, self.pos_interactions)
        self.neg_interactions_embeddings = tf.nn.embedding_lookup(
            self.item_embs, self.neg_interactions)
        self.context_embeddings = tf.nn.embedding_lookup(
            self.context_embs, self.context)
        self.user_sideinfo_embeddings = tf.nn.embedding_lookup(
            self.weights['user_sideinfo_embedding'], self.user_sideinfo)
        self.item_sideinfo_embeddings = tf.nn.embedding_lookup(
            self.weights['item_sideinfo_embedding'], self.item_sideinfo)

        # Initial weights for BPR
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(
            self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(
            self.weights['item_embedding'], self.pos_interactions)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(
            self.weights['item_embedding'], self.neg_interactions)

        # Biases
        self.user_bias = tf.nn.embedding_lookup(
            self.weights['user_bias'], self.users)
        self.pos_item_bias = tf.nn.embedding_lookup(
            self.weights['item_bias'], self.pos_interactions)
        self.neg_item_bias = tf.nn.embedding_lookup(
            self.weights['item_bias'], self.neg_interactions)

        self.batch_ratings = tf.matmul(
            self.user_embeddings, self.pos_interactions_embeddings, transpose_a=False, transpose_b=True)

        self.pos_scores = self._predict(self.user_embeddings, self.pos_interactions_embeddings,
                                        self.context_embeddings, self.user_bias, self.pos_item_bias)
        self.neg_scores = self._predict(self.user_embeddings, self.neg_interactions_embeddings,
                                        self.context_embeddings, self.user_bias, self.neg_item_bias)

        self.pos_scores = tf.nn.dropout(self.pos_scores, args.keep_prob)
        self.neg_scores = tf.nn.dropout(self.neg_scores, args.keep_prob)

        self.loss = self._bpr_loss(self.pos_scores, self.neg_scores)
        self.opt = self.optimizer.minimize(self.loss[0])
        self.init = tf.global_variables_initializer()

    def _csgcn_layers(self):
        uic_adj_mat = self._convert_sp_mat_to_sp_tensor(self.data.norm_adj_mat)
        us_adj_mat = self._convert_sp_mat_to_sp_tensor(
            self.data.norm_us_adj_mat)
        is_adj_mat = self._convert_sp_mat_to_sp_tensor(
            self.data.norm_is_adj_mat)

        us_comb_embs = tf.concat(
            [self.weights['user_embedding'], self.weights['user_sideinfo_embedding']], axis=0)
        us_comb_matmul = tf.sparse_tensor_dense_matmul(
            us_adj_mat, us_comb_embs)
        usr_embs = tf.split(
            us_comb_matmul, [self.data.n_users, self.data.n_user_sideinfo])[0]

        is_comb_embs = tf.concat(
            [self.weights['item_embedding'], self.weights['item_sideinfo_embedding']], axis=0)
        is_comb_matmul = tf.sparse_tensor_dense_matmul(
            is_adj_mat, is_comb_embs)
        is_embs = tf.split(
            is_comb_matmul, [self.data.n_items, self.data.n_item_sideinfo])[0]

        # add sideinfo to the user embeddings and item embeddings
        ue = tf.multiply(self.weights['user_embedding'], usr_embs)
        ie = tf.multiply(self.weights['item_embedding'], is_embs)

        embs = tf.concat([ue, ie, self.weights['context_embedding']], axis=0)
        all_embeddings = [embs]

        for k in range(0, self.n_layers):
            matmul = tf.sparse_tensor_dense_matmul(uic_adj_mat, embs)
            embs = matmul
            # TODO: Implemnter convolution formel, overvej at følge LGCN med NuNi
            all_embeddings += [matmul]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        ue, ie, ce = tf.split(
            all_embeddings, [self.data.n_users, self.data.n_items, self.data.n_context], 0)

        return ue, ie, ce

    def _predict(self, user_embs, item_embs, context_embs, user_bias, item_bias):
        # TODO: Overvej om vi skal fjerne user embs og bias og tage dem fra self i stedet
        user_emb_sum = tf.reduce_sum(user_embs, 1)
        item_emb_sum = tf.reduce_sum(item_embs, 1)
        emb_sum = tf.add(user_emb_sum, item_emb_sum)
        first_term = tf.square(emb_sum)

        user_emb_sq = tf.square(user_embs)
        item_emb_sq = tf.square(item_embs)
        user_emb_sq_sum = tf.reduce_sum(user_emb_sq, 1)
        item_emb_sq_sum = tf.reduce_sum(item_emb_sq, 1)
        second_term = tf.add(user_emb_sq_sum, item_emb_sq_sum)

        pred = 0.5 * tf.subtract(first_term, second_term)

        bilinear = tf.reduce_sum(pred, 1, keepdims=True)

        y_hat = tf.add_n([bilinear, user_bias, item_bias])

        return y_hat

    def _bpr_loss(self, pos_scores, neg_scores):
        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
            self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size

        mf_loss = tf.reduce_mean(-tf.log(tf.nn.sigmoid(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer

        loss = emb_loss + mf_loss
        return loss, emb_loss, mf_loss

    def _convert_sp_mat_to_sp_tensor(self, adj_mat):
        coo = adj_mat.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _partial_fit(self, data):
        feed_dict = {self.users: data['user_ids'], self.pos_interactions: data['pos_interactions'],
                     self.neg_interactions: data['neg_interactions'], self.context: data['contexts'],
                     self.user_sideinfo: data['user_sideinfo'], self.item_sideinfo: data['item_sideinfo']}
        return self.sess.run([self.loss, self.opt], feed_dict=feed_dict)

    def train(self):
        # tensorboard file name
        setup = '[' + args.dataset + '] init[' + str(args.initializer) + '] lr[' + str(args.lr) +'] optim[' + str(args.optimizer) + '] layers[' + str(
            args.layers) + '] batch[' + str(args.batch) + '] keep[' + str(args.keep_prob) + '] decay[' + str(args.decay) + '] ks' + str(args.ks)
        tensorboard_model_path = 'tensorboard/' + setup + '/'
        if not os.path.exists(tensorboard_model_path):
            os.makedirs(tensorboard_model_path)
        run_time = 1
        while (True):
            if os.path.exists(tensorboard_model_path + '/run_' + str(run_time)):
                run_time += 1
            else:
                break
        train_writer = tf.summary.FileWriter(
            tensorboard_model_path + '/run_' + str(run_time), sess.graph)

        # Initialize variables
        self.sess.run(self.init)

        # Run epochs
        for epoch in range(0, self.epochs + 1):
            batch = self.data.sampler(self.batch_size)

            # Run training on batch
            losses, _ = self._partial_fit(batch)
            loss, emb_loss, mf_loss = losses

            # Run to get summary of train loss
            summary_train_loss = sess.run(self.merged_train_loss,
                                          feed_dict={self.train_loss: loss,
                                                     self.train_mf_loss: mf_loss,
                                                     self.train_emb_loss: emb_loss})
            train_writer.add_summary(summary_train_loss, epoch)

            if epoch % 25 == 0:
                print(f"The total loss in {epoch}th iteration is: {loss}")
            if epoch % args.eval_interval == 0:
                ret = self.evaluate(epoch)
                summary_test_acc = sess.run(self.merged_test_acc, feed_dict={self.test_hr_first: ret['hr'][0],
                                                                             self.test_hr_last: ret['hr'][-1],
                                                                             self.test_ndcg_first: ret['ndcg'][0],
                                                                             self.test_ndcg_last: ret['ndcg'][-1],
                                                                             self.test_mrr_first: ret['mrr'][0],
                                                                             self.test_mrr_last: ret['mrr'][-1]
                                                                             })
                train_writer.add_summary(summary_test_acc, epoch)

    def evaluate(self, epoch):
        scores = dict()

        unique_item_ids = self.data.test_df[self.data.itemid_column_name].unique()

        for _, row in self.data.test_df.iterrows():
            userId = row[self.data.userid_column_name]
            user_index = self.data.user_offset_dict[userId]
            user_sideinfo = self.data.user_sideinfo_dict[userId]
            context = []
            for context_col in self.data.context_list:
                context.append(row[context_col])

            user_indexes = []
            user_sideinfos = []
            item_indexes = []
            item_sideinfos = []
            contexts = []
            for item in self.data.test_df[self.data.itemid_column_name].unique():
                item_index = self.data.item_offset_dict[item]
                item_sideinfo = self.data.item_sideinfo_dict[item]
                
                # if the item has more than one genre, choose a random one
                if len(item_sideinfo) > 1:
                    item_sideinfo = [random.choice(item_sideinfo)]
                
                user_indexes.append([user_index])
                user_sideinfos.append(user_sideinfo)
                item_indexes.append([item_index])
                item_sideinfos.append(item_sideinfo)
                contexts.append(context)

            feed_dict = {self.users: user_indexes, self.pos_interactions: item_indexes,
                         self.context: contexts, self.user_sideinfo: user_sideinfos,
                         self.item_sideinfo: item_sideinfos}
            pos_scores = self.sess.run(self.pos_scores, feed_dict=feed_dict)
            pos_scores = np.sum(pos_scores, axis=1)
            pos_scores = list(zip(unique_item_ids, pos_scores))

            scores[userId] = pos_scores

        ret = defaultdict(list)
        for k in self.ks:
            hr, ndcg, mrr = self.evaluator.evaluate_loo(
                scores, self.data.user_ground_truth_dict, k, epoch)
            ret['hr'].append(hr)
            ret['ndcg'].append(ndcg)
            ret['mrr'].append(mrr)
        return ret


if __name__ == '__main__':
    dataset = args.dataset

    if args.load == 1:
        path = 'checkpoints/' + dataset + '.chk'
        file_data = open(path, 'rb')
        data = pickle.load(file_data)
    else:
        data = LoadMovieLens(random_seed=args.seed, dataset=dataset)
        path = 'checkpoints/' + dataset + '.chk'
        file_data = open(path, 'wb')
        pickle.dump(data, file_data)

    with tf.Session() as sess:
        model = CSGCN(sess, data)
        model.train()
