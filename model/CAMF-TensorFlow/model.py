import tensorflow as tf
import random as random
import numpy as np
from utility.parser import parse_args
from tensorflow.python.client import device_lib
import os
from evaluation import evaluator
from LoadData import LoadDataset
import pickle
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']
args = parse_args()

class CAMF():
    def __init__(self, sess, data):
        self.alg_type = args.alg_type
        self.emb_dim = args.embed_size
        self.data = data
        self.evaluator = evaluator()
        self.batch_size = int(self.data.n_users // 10)
        self.learning_rate = args.lr
        self.random_seed = args.seed
        self.initializer = self._set_initializer(args.initializer)
        self.optimizer = self._set_optimizer(args.optimizer)
        self.epochs = args.epoch
        self.decay = args.decay
        self.ks = eval(args.ks)
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
        elif optimizer == 'GradientDescent':
            return tf.train.GradientDescentOptimizer(self.learning_rate)
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
        
        all_weights['user_embedding'] = tf.Variable(self.initializer([self.data.n_users, self.emb_dim]),
                                            name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(self.initializer([self.data.n_items, self.emb_dim]),
                                    name='item_embedding')
        
        all_weights['user_bias'] = tf.Variable(
                                tf.zeros([self.data.n_users], dtype=tf.float32, name='user_bias'))
        
        if self.alg_type == 'camf-c':
            all_weights['context_bias'] = tf.Variable(
                                        tf.zeros([self.data.n_context], dtype=tf.float32, name='context_bias'))
        elif self.alg_type == 'camf-ci':
            all_weights['context_bias'] = tf.Variable(
                                        tf.zeros([self.data.n_context*self.data.n_items], dtype=tf.float32, name='context_bias'))
        return all_weights
    
    def _init_graph(self):
        
        self.graph = tf.Graph()

        tf.set_random_seed(self.random_seed)
        
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.pos_context = tf.placeholder(tf.int32, shape=(None,None))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_context = tf.placeholder(tf.int32, shape=(None,None))    
        
        self.weights = self._init_weights()
        
        self.user_embeddings = tf.nn.embedding_lookup(
            self.weights['user_embedding'], self.users)
        self.pos_item_embeddings = tf.nn.embedding_lookup(
            self.weights['item_embedding'], self.pos_items)
        self.neg_item_embeddings = tf.nn.embedding_lookup(
            self.weights['item_embedding'], self.neg_items)
        self.pos_context_biases = tf.nn.embedding_lookup(
            self.weights['context_bias'], self.pos_context)
        self.neg_context_biases = tf.nn.embedding_lookup(
            self.weights['context_bias'], self.neg_context)
        self.user_biases = tf.nn.embedding_lookup(
            self.weights['user_bias'], self.users)
        
        self.pos_scores = self._predict(self.user_embeddings, self.pos_item_embeddings, self.user_biases, self.pos_context_biases)
        self.neg_scores = self._predict(self.user_embeddings, self.neg_item_embeddings, self.user_biases, self.neg_context_biases)
        
        self.loss = self._bpr_loss(self.pos_scores, self.neg_scores)
        self.opt = self.optimizer.minimize(self.loss[0])
        self.init = tf.global_variables_initializer()
        
    def _predict(self, users, items, user_bias, context_bias):
        scores = tf.matmul(users, items, transpose_b=True) + tf.expand_dims(user_bias, -1) + tf.ones([1]) + tf.reduce_sum(context_bias, axis=1)
        return scores
    
    def _bpr_loss(self, pos_scores, neg_scores):
        regularizer = tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(
            self.pos_item_embeddings) + tf.nn.l2_loss(self.neg_item_embeddings) \
                    + tf.nn.l2_loss(self.pos_context_biases) \
                    + tf.nn.l2_loss(self.neg_context_biases) \
                    + tf.nn.l2_loss(self.user_biases)
                
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer

        loss = emb_loss + mf_loss
        return loss, emb_loss, mf_loss
    
    def _partial_fit(self, data):
        if self.alg_type == 'camf-c':
                
            feed_dict = {self.users: data['user_ids'], self.pos_items: data['pos_interactions'],
                        self.neg_items: data['neg_interactions'], self.pos_context: data['contexts'],
                        self.neg_context: data['contexts']}
        elif self.alg_type == 'camf-ci':
            feed_dict = {self.users: data['user_ids'], self.pos_items: data['pos_interactions'],
                        self.neg_items: data['neg_interactions'], self.pos_context: data['pos_contexts'],
                        self.neg_context: data['neg_contexts']}
        return self.sess.run([self.loss, self.opt], feed_dict=feed_dict)
    
    def train(self):
        # tensorboard file name
        setup = '[' + args.dataset + '] init[' + str(args.initializer) + '] lr[' + str(args.lr) +'] optim[' + str(args.optimizer) + \
            '] batch[' + str(args.batch) + '] decay[' + str(args.decay) + '] ks' + str(args.ks)
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
            tensorboard_model_path + '/run_' + str(run_time), self.sess.graph)

        # Initialize variables
        self.sess.run(self.init)

        # Run epochs
        for epoch in range(0, self.epochs + 1):
            batch = self.data.sampler(self.alg_type, self.batch_size)

            # Run training on batch
            losses, _ = self._partial_fit(batch)
            loss, emb_loss, mf_loss = losses

            # Run to get summary of train loss
            summary_train_loss = self.sess.run(self.merged_train_loss,
                                          feed_dict={self.train_loss: loss,
                                                     self.train_mf_loss: mf_loss,
                                                     self.train_emb_loss: emb_loss})
            train_writer.add_summary(summary_train_loss, epoch)

            if epoch % 25 == 0:
                print(f"The total loss in {epoch}th iteration is: {loss}")
            if (epoch > 0 and epoch % args.eval_interval == 0) or epoch == self.epochs:
                if args.eval_method == 'fold':
                    ret = self.evaluate(epoch)
                    summary_test_acc = self.sess.run(self.merged_test_acc, feed_dict={self.test_precision_first: ret['precision'][0],
                                                                self.test_precision_last: ret['precision'][-1],
                                                                self.test_recall_first: ret['recall'][0],
                                                                self.test_recall_last: ret['recall'][-1],
                                                                self.test_f1_first: ret['f1'][0],
                                                                self.test_f1_last: ret['f1'][-1],
                                                                self.test_ndcg_first: ret['ndcg'][0],
                                                                self.test_ndcg_last: ret['ndcg'][-1],
                                                                })
                elif args.eval_method == 'loo':
                    ret = self.evaluate_loo(epoch)
                    summary_test_acc = self.sess.run(self.merged_test_acc, feed_dict={self.test_hr_first: ret['hr'][0],
                                                                                self.test_hr_last: ret['hr'][-1],
                                                                                self.test_ndcg_first: ret['ndcg'][0],
                                                                                self.test_ndcg_last: ret['ndcg'][-1],
                                                                                self.test_mrr_first: ret['mrr'][0],
                                                                                self.test_mrr_last: ret['mrr'][-1],
                                                                                })
                train_writer.add_summary(summary_test_acc, epoch)
                
    def evaluate(self, epoch):  
        scores = dict()
        all_pos_scores = []
        
        user_indexes = []
        item_indexes = []

        for item in self.data.full_df[self.data.itemid_column_name].unique():
            item_index = self.data.item_offset_dict[item]
            item_indexes.append(item_index)

        for user in self.data.full_df[self.data.userid_column_name].unique():
            user_index = self.data.user_offset_dict[user]
            user_indexes.append(user_index)
                
        for context_comb in self.data.context_test_combinations:
            feed_dict = {self.users: user_indexes, self.pos_items: item_indexes,
                            self.pos_context: [list(context_comb)]}
            pos_scores = self.sess.run(self.pos_scores, feed_dict=feed_dict)
            all_pos_scores.append(pos_scores)
        
        all_pos_scores = np.stack(all_pos_scores)
        all_pos_scores = np.mean(all_pos_scores, axis=0)

        all_pos_scores = np.array(all_pos_scores)
        scores = dict()
        for user in self.data.full_df[self.data.userid_column_name].unique():
            train_items = [itemId for itemId, _ in self.data.train_set_user_pos_interactions[user]]
            train_items_indexes = [self.data.item_offset_dict[itemId] for itemId in train_items]
            user_index = self.data.user_offset_dict[user]
            all_pos_scores[user_index][train_items_indexes] = -np.inf
            
            scores[user] = list(zip(self.data.full_df[self.data.itemid_column_name].unique(), all_pos_scores[user_index]))
            

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
        data = LoadDataset(random_seed=args.seed, dataset=dataset, eval_method=args.eval_method)
        path = 'checkpoints/' + dataset + '.chk'
        file_data = open(path, 'wb')
        pickle.dump(data, file_data)

    with tf.Session() as sess:
        model = CAMF(sess, data)
        model.train()