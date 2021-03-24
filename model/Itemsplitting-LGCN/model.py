from LoadData import LoadData
from utility.parser import parse_args
from evaluation import evaluator
from tensorflow.python.client import device_lib
from collections import defaultdict
import os
import pickle
import tensorflow as tf
import random
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']
args = parse_args()

class ISLGCN():
    def __init__(self, sess, data):
        self.random_seed = args.seed
        random.seed(self.random_seed)
        self.decay = args.decay
        self.data = data
        print("Loaded data")
        self.n_layers = len(eval(args.weight_size))
        print(self.n_layers)
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
            if args.eval_method == 'loo':
                self.test_hr_first = tf.placeholder(tf.float32)
                tf.summary.scalar('test_hr_first', self.test_hr_first)
                self.test_mrr_first = tf.placeholder(tf.float32)
                tf.summary.scalar('test_mrr_first', self.test_mrr_first)
                self.test_hr_last = tf.placeholder(tf.float32)
                tf.summary.scalar('test_hr_last', self.test_hr_last)
                self.test_mrr_last = tf.placeholder(tf.float32)
                tf.summary.scalar('test_mrr_last', self.test_mrr_last)
            elif args.eval_method == 'fold':
                self.test_precision_first = tf.placeholder(tf.float32)
                tf.summary.scalar('test_precision_first', self.test_precision_first)
                self.test_recall_first = tf.placeholder(tf.float32)
                tf.summary.scalar('test_recall_first', self.test_recall_first)
                self.test_f1_first = tf.placeholder(tf.float32)
                tf.summary.scalar('test_f1_first', self.test_f1_first)
                self.test_precision_last = tf.placeholder(tf.float32)
                tf.summary.scalar('test_precision_last', self.test_precision_last)
                self.test_recall_last = tf.placeholder(tf.float32)
                tf.summary.scalar('test_recall_last', self.test_recall_last)
                self.test_f1_last = tf.placeholder(tf.float32)
                tf.summary.scalar('test_f1_last', self.test_f1_last)
            
            self.test_ndcg_first = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_first', self.test_ndcg_first)
            self.test_ndcg_last = tf.placeholder(tf.float32)
            tf.summary.scalar('test_ndcg_last', self.test_ndcg_last)
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
        
        all_weights = dict()
        
        all_weights['user_embeddings'] = tf.Variable(self.initializer([self.data.n_users, self.emb_dim]),
                                                     name='user_embedding')
        all_weights['item_embeddings'] = tf.Variable(self.initializer([self.data.n_items_split, self.emb_dim]),
                                                     name='item_embedding')
        return all_weights
    
    def _init_graph(self):
        #tf.set_random_seed(self.random_seed)
        
        self.users = tf.placeholder(tf.int32, shape=[None,None])
        self.pos_interactions = tf.placeholder(tf.int32, shape=[None,None])
        self.neg_interactions = tf.placeholder(tf.int32, shape=[None,None])

        self.weights = self._init_weights()
        self.user_embs, self.item_embs = self._lgcn_layers()
        
        self.batch_user_embeddings = tf.nn.embedding_lookup(
            self.user_embs, self.users)
        self.batch_pos_interactions_embeddings = tf.nn.embedding_lookup(
            self.item_embs, self.pos_interactions)
        self.batch_neg_interactions_embeddings = tf.nn.embedding_lookup(
            self.item_embs, self.neg_interactions)
        self.user_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.users)
        self.pos_i_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.pos_interactions)
        self.neg_i_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.neg_interactions)
        
        self.batch_ratings = tf.matmul(self.batch_user_embeddings, self.batch_pos_interactions_embeddings, transpose_a=False, transpose_b=True)
        
        self.loss = self._bpr_loss(self.batch_user_embeddings, self.batch_pos_interactions_embeddings, self.batch_neg_interactions_embeddings)
        self.opt = self.optimizer.minimize(self.loss[0])
        self.init = tf.global_variables_initializer()
        
    def _lgcn_layers(self):
        
        norm_adj_mat = self._convert_sp_mat_to_sp_tensor(self.data.norm_adj_mat)
        
        ego_embeddings = tf.concat([self.weights['user_embeddings'], self.weights['item_embeddings']], axis=0)
        all_embeddings = [ego_embeddings]
        
        for k in range(0, self.n_layers):
            ego_embeddings = tf.sparse_tensor_dense_matmul(norm_adj_mat, ego_embeddings)
            all_embeddings += [ego_embeddings]
            
        all_embeddings = tf.stack(all_embeddings,1)
        all_embeddings = tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.data.n_users, self.data.n_items_split], 0)
        return u_g_embeddings, i_g_embeddings


    def _convert_sp_mat_to_sp_tensor(self, adj_mat):
        coo = adj_mat.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
    
    def _bpr_loss(self, users, pos_scores, neg_scores):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_scores), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_scores), axis=1)
        
        regularizer = tf.nn.l2_loss(self.user_embeddings_pre) + tf.nn.l2_loss(
            self.pos_i_embeddings_pre) + tf.nn.l2_loss(self.neg_i_embeddings_pre)
        regularizer = regularizer / self.batch_size

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer

        loss = emb_loss + mf_loss
        return loss, emb_loss, mf_loss

    def _partial_fit(self, data):
        feed_dict = {self.users: data['user_ids'], self.pos_interactions: data['pos_interactions'],
                     self.neg_interactions: data['neg_interactions']}
        return self.sess.run([self.loss, self.opt], feed_dict=feed_dict)
    
    def train(self):
        # tensorboard file name
        setup = '[' + args.dataset + '] init[' + str(args.initializer) + '] lr[' + str(args.lr) +'] optim[' + str(args.optimizer) + '] layers[' + str(
            args.weight_size) + '] batch[' + str(args.batch) + '] keep[' + str(args.keep_prob) + '] decay[' + str(args.decay) + '] ks' + str(args.ks)
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
            losses, _  = self._partial_fit(batch)
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
                if args.eval_method == 'fold':
                    ret = self.evaluate(epoch)
                    summary_test_acc = sess.run(self.merged_test_acc, feed_dict={self.test_precision_first: ret['precision'][0],
                                                                self.test_precision_last: ret['precision'][-1],
                                                                self.test_recall_first: ret['recall'][0],
                                                                self.test_recall_last: ret['recall'][-1],
                                                                self.test_f1_first: ret['f1'][0],
                                                                self.test_f1_last: ret['f1'][-1],
                                                                self.test_ndcg_first: ret['ndcg'][0],
                                                                self.test_ndcg_last: ret['ndcg'][-1]
                                                                })
                elif args.eval_method == 'loo':
                    ret = self.evaluate_loo(epoch)
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

        for userId in self.data.test_df[self.data.userid_column_name].unique():
            user_index = self.data.user_id_to_offset_dict[userId]

            user_indexes = []
            item_indexes = []
            item_ids = []
            for key, value in self.data.item_id_context_to_offset_dict.items():    
                user_indexes.append([user_index])
                item_indexes.append([value])
                item_ids.append(key[0])

            feed_dict = {self.users: user_indexes, self.pos_interactions: item_indexes}
            batch_ratings = self.sess.run(self.batch_ratings, feed_dict=feed_dict)
            batch_ratings = np.sum(batch_ratings, axis=1)

            batch_ratings = list(zip(item_ids, batch_ratings))
            
            batch_ratings_dict = dict()
            for itemId, score in batch_ratings:
                if itemId not in batch_ratings_dict:
                    batch_ratings_dict[itemId] = score
                else:
                    if score > batch_ratings_dict[itemId]:
                        batch_ratings_dict[itemId] = score
            batch_ratings_tuple_list = [(k, v) for k, v in batch_ratings_dict.items()] 
            
            if userId in self.data.train_set_user_pos_interactions:
                user_train_pos_interactions = [itemId for (itemId, context) in self.data.train_set_user_pos_interactions[userId]]
            else:
                user_train_pos_interactions = []
            batch_ratings_tuple_list = [(itemId, -np.inf) if itemId in user_train_pos_interactions else (itemId, score) for (itemId, score) in batch_ratings_tuple_list]
            scores[userId] = batch_ratings_tuple_list

        ret = defaultdict(list)
        for k in self.ks:
            precision_value, recall_value, f1_value, ndcg_value = self.evaluator.evaluate(
                scores, self.data.user_ground_truth_dict, k, epoch)
            ret['precision'].append(precision_value)
            ret['recall'].append(recall_value)
            ret['f1'].append(f1_value)
            ret['ndcg'].append(ndcg_value)
        return ret
    
if __name__ == '__main__':
    dataset = args.dataset
    
    if args.load == 1:
        path = 'checkpoints/' + dataset + '.chk'
        file_data = open(path, 'rb')
        data = pickle.load(file_data)
    else:
        data = LoadData(random_seed=args.seed, dataset=dataset, eval_method=args.eval_method)
        path = 'checkpoints/' + dataset + '.chk'
        file_data = open(path, 'wb')
        pickle.dump(data, file_data)
    
    with tf.Session() as sess:
            model = ISLGCN(sess, data)
            model.train()