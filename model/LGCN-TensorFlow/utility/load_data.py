'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import pandas as pd

class Data(object):
    def __init__(self, path, alg_type, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.loo_eval = False
        self.alg_type = alg_type

        if 'ml100k' in self.path:
            self.user_column_name = 'userId'
            self.item_column_name = 'movieId'
            self.context_column_list = ['timeofday', 'weekday']
            self.item_sideinfo_multihot = ['unknown', 'action', 'adventure', 'animation',
                              'childrens', 'comedy', 'crime', 'documentary',
                              'drama', 'fantasy',  'film-noir', 'horror',
                              'musical', 'mystery', 'romance', 'scifi',
                              'thriller', 'war', 'western']
            self.item_sideinfo_onehot = []
            self.user_sideinfo_onehot = ['age', 'gender', 'occupation']

        if 'yelpnc' in self.path:
            self.user_column_name = 'user_id'
            self.item_column_name = 'business_id'
            self.context_column_list = ['month','day_of_week','timeofday'] # 'hour'
            self.item_sideinfo_multihot = ['Burgers','ChickenWings','Bars','Restaurants','Nightlife','SportsBars','American(Traditional)','Steakhouses','Tapas/SmallPlates','Breakfast&Brunch','American(New)','Pubs','CocktailBars','TapasBars','Gastropubs','Food','Coffee&Tea','BeerBar','Breweries','Bakeries','Southern','SoulFood','Arts&Entertainment','Sandwiches','MusicVenues','Pizza','Italian','WineBars','IceCream&FrozenYogurt','FastFood','Chinese','SushiBars','Thai','AsianFusion','Japanese','EventPlanning&Services','Venues&EventSpaces','Mexican','Vietnamese','Diners','ComfortFood','Cafes','Salad','French','Caterers','Shopping','Beer','Wine&Spirits','Delis','Barbeque','Seafood','Soup','Tex-Mex','Fashion','DepartmentStores','Lounges','Vegetarian','ActiveLife','Desserts','LatinAmerican','Cinema','Vegan','Gluten-Free','Home&Garden','Cajun/Creole','Bagels','Beauty&Spas','SpecialtyFood','Mediterranean','Grocery','Noodles','FoodTrucks','LocalFlavor','Greek','Indian','EthnicFood','BeerGardens','Korean','Flowers&Gifts']
            self.item_sideinfo_onehot = ['city']
            self.user_sideinfo_onehot = ['yelping_since', 'fans', 'average_stars']

        if 'yelpon' in self.path:
            self.user_column_name = 'user_id'
            self.item_column_name = 'business_id'
            self.context_column_list = ['date']
            self.item_sideinfo_multihot = ['SpecialtyFood','Restaurants','EthnicFood','Chinese','Caterers','Food','EventPlanning&Services','Nightlife','Steakhouses','Bars','Seafood','SportsBars','Canadian(New)','American(Traditional)','Burgers','Italian','CocktailBars','Mediterranean','Hotels&Travel','Venues&EventSpaces','Gastropubs','Arts&Entertainment','Mexican','Barbeque','ComfortFood','Thai','AsianFusion','Pakistani','Buffets','American(New)','Beauty&Spas','Grocery','Coffee&Tea','FastFood','Pizza','Salad','SushiBars','Bakeries','French','Breakfast&Brunch','Korean','IceCream&FrozenYogurt','Noodles','Desserts','Cafes','Diners','Soup','Sandwiches','Ramen','Japanese','Pubs','Vietnamese','Shopping','MiddleEastern','Halal','Taiwanese','Vegan','TeaRooms','Vegetarian','DimSum','WineBars','JuiceBars&Smoothies','Fashion','Caribbean','Beer','Wine&Spirits','ChickenWings','Lounges','TapasBars','Tapas/SmallPlates','Indian','BubbleTea','ActiveLife','Greek','Gluten-Free']
            self.item_sideinfo_onehot = ['city']
            self.user_sideinfo_onehot = ['yelping_since', 'fans', 'average_stars']

        if 'frappe' in self.path:
            self.user_column_name = 'user'
            self.item_column_name = 'item'
            self.context_column_list = ['weekday', 'isweekend', 'country', 'city', 'weather', 'timeofday']
            self.item_sideinfo_multihot = ['cost']
            self.item_sideinfo_onehot = []
            self.user_sideinfo_onehot = []

        if 'ml1m' in self.path:
            self.user_column_name = 'userId'
            self.item_column_name = 'movieId'
            self.context_column_list = ['weekday','timeofday']
            self.item_sideinfo_multihot = ['Action','Adventure','Animation','Children\'s','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
            self.item_sideinfo_onehot = []
            self.user_sideinfo_onehot = ['age','gender','occupation']

        if self.loo_eval:
            self.init_loo_split()
        else:
            self.init_train_test_split()

        self.n_user_sideinfo, self.n_item_sideinfo = self.sideinfo_counter()

        if self.alg_type in ['csgcn']:
            self.test_context_combinations = self.get_test_context_combinations()

    def sideinfo_counter(self):
        n_user_sideinfo = 0
        for column_name in self.user_sideinfo_onehot:
            n_user_sideinfo += self.train_df[column_name].nunique()

        n_item_sideinfo = 0
        for column_name in self.item_sideinfo_onehot:
            n_item_sideinfo += self.train_df[column_name].nunique()

        n_item_sideinfo += len(self.item_sideinfo_multihot)

        return n_user_sideinfo, n_item_sideinfo

    def get_test_context_combinations(self):
        # get the unique context combinations in the test set
        combinations = set()

        for _, row in self.test_df.iterrows():
            context_list = []
            for context in self.context_column_list:
                context_list.append(self.context_offset_dict[context + str(row[context])])
            combinations.add(tuple(context_list))
        return combinations

    def init_loo_split(self):
        full_file = self.path + '/out.txt'

        self.n_users, self.n_items, self.n_contexts = 0, 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        self.full_df = pd.read_csv(full_file)
        reverse_full_df = self.full_df[::-1]
        self.train_df = self.full_df

        self.n_items = self.full_df[self.item_column_name].max()
        self.n_users = self.full_df[self.user_column_name].max()
        self.unique_users = self.full_df[self.user_column_name].unique()
        loo_interactions = []
        #count = 0
        for userid in self.unique_users:
            #if count == 20:
               # break
            for index, row in reverse_full_df.iterrows():
                if row[self.user_column_name] == userid:
                    loo_interactions.append(row)
                    self.train_df.drop(index, inplace=True)
                    break

        self.test_df = pd.DataFrame(loo_interactions)
        print("test")
        print(self.test_df.shape)
        print("train")
        print(self.train_df.shape)
        self.create_positive_interactions()


    def init_train_test_split(self):
        train_file = self.path + '/train.txt'
        test_file = self.path + '/test.txt'

        self.n_users, self.n_items, self.n_contexts = 0, 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        self.train_df = pd.read_csv(train_file)
        self.test_df = pd.read_csv(test_file)
        self.full_df = self.train_df.append(self.test_df)

        self.n_items = self.full_df[self.item_column_name].max()
        self.n_users = self.full_df[self.user_column_name].max()
        self.n_contexts = sum([self.full_df[context].nunique() for context in self.context_column_list])

        self.create_positive_interactions()


    def create_positive_interactions(self):
        self.exist_users = self.full_df[self.user_column_name].unique()
        self.n_train = len(self.train_df.index)
        self.n_test = len(self.test_df.index)

        if self.alg_type in ['csgcn']:
            context_offset_dict = {}
            item_context_offset_dict = {}
            item_sideinfo_offset_dict = {}
            user_sideinfo_offset_dict = {}
            item_sideinfo_dict = {}
            user_sideinfo_dict = {}


            offset = 0
            for column in self.item_sideinfo_multihot:
                item_sideinfo_offset_dict[column] = offset
                offset += 1

            for column in self.item_sideinfo_onehot:
                for value in self.full_df[column].unique():
                    item_sideinfo_offset_dict[column + str(value)] = offset

            offset = 0
            for column in self.user_sideinfo_onehot:
                for value in self.full_df[column].unique():
                    user_sideinfo_offset_dict[column + str(value)] = offset

            offset = 0
            for column in self.context_column_list:
                for value in self.full_df[column].unique():
                    context_offset_dict[column + str(value)] = offset
                    offset += 1

            offset = 0
            for _, value in enumerate(self.full_df[self.item_column_name].unique()):
                for context in context_offset_dict.keys():
                    item_context_offset_dict[(value, context)] = offset
                    offset += 1

            self.context_offset_dict = context_offset_dict
            self.item_context_offset_dict = item_context_offset_dict
            self.item_sideinfo_offset_dict = item_sideinfo_offset_dict
            self.user_sideinfo_offset_dict = user_sideinfo_offset_dict

        self.n_items += 1
        self.n_users += 1
        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = {}, {}
        # Key: userID || value: list of positive interactions
        # self.R: [uid, i] = 1 for hver interaction

        for _, row in self.train_df.iterrows():
            userId = row[self.user_column_name]
            itemId = row[self.item_column_name]
            self.R[userId, itemId] = 1.


            if self.alg_type in ['csgcn']:
                pos_interaction = (row[self.item_column_name],)
                contexts = tuple()
                for context in self.context_column_list:
                    contexts = contexts + (row[context],)
                pos_interaction = pos_interaction + (contexts,)

                if userId not in self.train_items:
                    self.train_items[userId] = [pos_interaction]
                else:
                    self.train_items[userId].append(pos_interaction)

                # Add side-info dicts
                # self.user_sideinfo_indexes
                user_sideinfo = []
                for column in self.user_sideinfo_onehot:
                    user_sideinfo.append(self.user_sideinfo_offset_dict[column + str(row[column])])

                # self.item_sideinfo_indexes
                item_sideinfo_multihot = []
                item_sideinfo_onehot = []
                for column in self.item_sideinfo_multihot:
                    if row[column] == 1:
                        item_sideinfo_multihot.append(self.item_sideinfo_offset_dict[column])
                for column in self.item_sideinfo_onehot:
                    item_sideinfo_onehot.append(self.item_sideinfo_offset_dict[column + str(row[column])])

                user_sideinfo_dict[userId] = user_sideinfo
                item_sideinfo_dict[itemId] = (item_sideinfo_onehot, item_sideinfo_multihot)

            else:
                if userId not in self.train_items:
                    self.train_items[userId] = [itemId]
                else:
                    self.train_items[userId].append(itemId)

        # If in CSGCN, add sideinfo
        if self.alg_type in ['csgcn']:
            self.user_sideinfo_dict = user_sideinfo_dict
            self.item_sideinfo_dict = item_sideinfo_dict

        for _, row in self.test_df.iterrows():
            userId = row[self.user_column_name]
            itemId = row[self.item_column_name]

            if self.alg_type in ['csgcn']:
                pos_interaction = (row[self.item_column_name],)
                contexts = tuple()

                for context in self.context_column_list:
                    contexts = contexts + (row[context],)
                pos_interaction = pos_interaction + (contexts,)

                if userId not in self.test_set:
                    self.test_set[userId] = [pos_interaction]
                else:
                    self.test_set[userId].append(pos_interaction)
            else:
                if userId not in self.test_set:
                    self.test_set[userId] = [itemId]
                else:
                    self.test_set[userId].append(itemId)


    def get_adj_mat(self):
        if self.alg_type in ['csgcn']:
            prefix = '/csgcn'
        else:
            prefix = '/lgcn'

        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + prefix + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + prefix + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + prefix + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + prefix + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + prefix + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + prefix + '/s_mean_adj_mat.npz', mean_adj_mat)

        try:
            pre_adj_mat = sp.load_npz(self.path + prefix + '/s_pre_adj_mat.npz')
        except Exception:
            adj_mat=adj_mat
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()

            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            sp.save_npz(self.path + prefix + '/s_pre_adj_mat.npz', norm_adj)

        return adj_mat, norm_adj_mat, mean_adj_mat,pre_adj_mat

    def create_adj_mat(self):
        t1 = time()

        def csgcn_adj():
            adj_size = self.n_users + self.n_items + self.n_user_sideinfo + self.n_item_sideinfo
            adj_mat = sp.dok_matrix((adj_size, adj_size), dtype=np.float32).tolil()
            R = self.R.tolil()

            for _, row in self.train_df.iterrows():
                userId = row[self.user_column_name]
                itemId = row[self.item_column_name]

                user_sideinfo_indexes = self.user_sideinfo_dict[userId]
                item_sideinfos = self.item_sideinfo_dict[itemId] # (OneHot, Multihot)

                item_offset = self.n_users + itemId

                for user_sideinfo in user_sideinfo_indexes:
                    user_sideinfo_offset = self.n_users + self.n_items + user_sideinfo
                    adj_mat[userId, user_sideinfo_offset] = 1 # IS

                for onehot in item_sideinfos[0]: # Onehot
                    item_sideinfo_offset = self.n_users + self.n_items + self.n_user_sideinfo + onehot
                    adj_mat[itemId, item_sideinfo_offset] = 1 # US (OneHot)
                    # TODO Prøv at indsætte identity matrix i stedet
                    #adj_mat[item_sideinfo_offset, itemId] = 1 # USt (OneHot)
                for multihot in item_sideinfos[1]: # Multihot
                    item_sideinfo_offset = self.n_users + self.n_items + self.n_user_sideinfo + multihot
                    adj_mat[itemId, item_sideinfo_offset] = 1 # US (Multihot)
                    #adj_mat[item_sideinfo_offset, itemId] = 1 # USt (Multihot)

                for i in range(self.n_user_sideinfo + self.n_item_sideinfo):
                    offset = self.n_users + self.n_items
                    adj_mat[i + offset, i + offset] = 1

                adj_mat[userId, item_offset] = 1 # R
                adj_mat[item_offset, userId] = 1 # Rt

            return adj_mat

        def lgcn_adj():
            adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.R.tolil()
            # prevent memory from overflowing

            for i in range(5):
                # 0,
                adj_mat[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5), self.n_users:] =\
                R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)]

                adj_mat[self.n_users:,int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)] =\
                R[int(self.n_users*i/5.0):int(self.n_users*(i+1.0)/5)].T
            adj_mat = adj_mat.todok()
            print('already create adjacency matrix', adj_mat.shape, time() - t1)
            return adj_mat

        if self.alg_type in ['csgcn']:
            adj_mat = csgcn_adj()
        else:
            adj_mat = lgcn_adj()

        t2 = time()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def negative_pool(self):
        t1 = time()
        for u in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[u]))
            pools = [rd.choice(neg_items) for _ in range(100)]
            self.neg_pools[u] = pools
        print('refresh negative pools', time() - t1)

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(list(self.exist_users), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]


        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]

                if self.alg_type in ['csgcn']:
                    if neg_id not in [itemId for itemId,_ in self.train_items[u]] and neg_id not in neg_items:
                        neg_items.append(neg_id)
                else:
                    if neg_id not in self.train_items[u] and neg_id not in neg_items:
                        neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items, pos_items_context, neg_items_context = [], [], [], []
        for u in users:
            if self.alg_type in ['csgcn']:
                pos_item_id, context = sample_pos_items_for_u(u, 1)[0]
                neg_item_id = sample_neg_items_for_u(u, 1)[0]
                pos_items.append(pos_item_id)
                neg_items.append(neg_item_id)

                pos_item_context, neg_item_context = [], []
                for index, context_col in enumerate(self.context_column_list, 0):
                    pos_item_context.append(self.item_context_offset_dict[(pos_item_id, context_col + str(context[index]))])
                    neg_item_context.append(self.item_context_offset_dict[(neg_item_id, context_col + str(context[index]))])

                pos_items_context.append(pos_item_context)
                neg_items_context.append(neg_item_context)
            else:
                pos_items += sample_pos_items_for_u(u, 1)
                neg_items += sample_neg_items_for_u(u, 1)

        if self.alg_type in ['csgcn']:
            return users, pos_items, neg_items, pos_items_context, neg_items_context
        else:
            return users, pos_items, neg_items

    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if self.alg_type in ['csgcn']:
                    if neg_id not in [itemId for itemId,_ in (self.test_set[u]+self.train_items[u])] and neg_id not in neg_items:
                        neg_items.append(neg_id)
                else:
                    if neg_id not in (self.test_set[u]+self.train_items[u]) and neg_id not in neg_items:
                        neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        for u in users:
            pos_items, neg_items = [], []
            if self.alg_type in ['csgcn']:
                pos_items_context, neg_items_context = [], []
                pos_item_id, context = sample_pos_items_for_u(u, 1)[0]
                neg_item_id = sample_neg_items_for_u(u, 1)[0]
                pos_items.append(pos_item_id)
                neg_items.append(neg_item_id)

                pos_item_context, neg_item_context = [], []
                for index, context_col in enumerate(self.context_column_list, 0):
                    pos_item_context.append(
                        self.item_context_offset_dict[(pos_item_id, context_col + str(context[index]))])
                    neg_item_context.append(
                        self.item_context_offset_dict[(neg_item_id, context_col + str(context[index]))])

                pos_items_context.append(pos_item_context)
                neg_items_context.append(neg_item_context)
            else:
                pos_items += sample_pos_items_for_u(u, 1)
                neg_items += sample_neg_items_for_u(u, 1)

        if self.alg_type in ['csgcn']:
            return users, pos_items, neg_items, pos_items_context, neg_items_context
        else:
            return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

        if self.batch_size != self.n_users // 10:
            print(f"!! Wrong batch size, you have {self.batch_size}, consider {self.n_users // 10} !!")


    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + '/sparsity.split', 'r').readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(' ')])
            print('get sparsity split.')

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + '/sparsity.split', 'w')
            for idx in range(len(split_state)):
                f.write(split_state[idx] + '\n')
                f.write(' '.join([str(uid) for uid in split_uids[idx]]) + '\n')
            print('create sparsity split.')

        return split_uids, split_state



    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = (self.n_train + self.n_test)
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' %(n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = '#inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)



        return split_uids, split_state
