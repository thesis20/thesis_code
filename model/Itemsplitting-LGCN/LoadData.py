import pandas as pd
from scipy import sparse
import numpy as np
import random
import itertools

class LoadData():
    def __init__(self, random_seed, dataset='ml100k', eval_method='fold'):

            if dataset == 'ml100k':
                self.genrelist = ['unknown', 'action', 'adventure', 'animation',
                                'childrens', 'comedy', 'crime', 'documentary',
                                'drama', 'fantasy',  'film-noir', 'horror',
                                'musical', 'mystery', 'romance', 'scifi',
                                'thriller', 'war', 'western']
                self.user_sideinfo_columns = [
                    'age', 'gender', 'occupation', 'zipcode']
                self.context_list = ['weekday', 'timeofday']
                self.userid_column_name = 'userId'
                self.itemid_column_name = 'movieId'
                self.path = '../../Data/ml100k/'
            elif dataset == 'ml1m':
                self.genrelist = ['Action', 'Adventure', 'Animation', 'Children\'s',
                                'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
                self.user_sideinfo_columns = [
                    'age', 'gender', 'occupation', 'zipcode']
                self.context_list = ['weekday', 'timeofday']
                self.userid_column_name = 'userId'
                self.itemid_column_name = 'movieId'
                self.path = '../../Data/ml1m/'

            self.eval_method = eval_method
            self.train_file = self.path + "train.txt"
            self.test_file = self.path + "test.txt"
            self.full_file = self.path + "out.txt"
            random.seed(random_seed)
            print("Loading data")
            self.load_data()  # Load train_df, test_df and full_data
            self.n_train_users, self.n_test_users, self.n_users = self.user_counter()
            self.n_train_items, self.n_test_items, self.n_items = self.item_counter()
            self.n_user_sideinfo, self.n_item_sideinfo = self.sideinfo_counter()
            self.n_context = self.context_counter()
            self.n_train_interactions = len(self.train_df.index)
            print("Counting dimensions")
            self.context_combinations = self.get_context_combinations()
            self.count_dimensions()
            print("Building dictionaries")
            self.build_dictionaries()
            print("Creating adj matrices")
            self.adj_mat, self.norm_adj_mat = self._create_adj_mat()
            print(f"n_users: {self.n_users}")
            print(f"n_items: {self.n_items}")
            print(f"n_items_split: {self.n_items_split}")
            print(f"n_user_sideinfo: {self.n_user_sideinfo}")
            print(f"n_item_sideinfo: {self.n_item_sideinfo}")
            print(f"n_context: {self.n_context}")
            print("-------- LEARNING TIME --------")
            
        
    def user_counter(self):
        train_users = self.train_df[self.userid_column_name].nunique()
        test_users = self.test_df[self.userid_column_name].nunique()
        total_users = self.full_df[self.userid_column_name].nunique()
        return train_users, test_users, total_users

    def item_counter(self):
        train_items = self.train_df[self.itemid_column_name].nunique()
        test_items = self.test_df[self.itemid_column_name].nunique()
        total_items = self.full_df[self.itemid_column_name].nunique()
        return train_items, test_items, total_items

    def sideinfo_counter(self):
        n_user_sideinfo = 0
        for column_name in self.user_sideinfo_columns:
            n_user_sideinfo += self.full_df[column_name].nunique()
        n_item_sideinfo = len(self.genrelist)
        return n_user_sideinfo, n_item_sideinfo

    def context_counter(self):
        context_count = 0
        for column_name in self.context_list:
            context_count += self.full_df[column_name].nunique()
        return context_count

    def load_data(self):
        self.full_df = pd.read_csv(self.full_file, sep=',')
        
        if self.eval_method == 'loo':
            indices = self.full_df.index
            test_indices = []
            for userId in self.full_df[self.userid_column_name].unique():
                user_df = self.full_df[self.full_df[self.userid_column_name] == userId]
                newest_entry = user_df.index[user_df['timestamp']
                                                == user_df['timestamp'].max()].tolist()
                newest_row = newest_entry[-1]
                test_indices.append(indices[newest_row])
            self.test_df = self.full_df.loc[test_indices]
            train_indices = list(set(indices).difference(test_indices))
            self.train_df = self.full_df.loc[train_indices]
        elif self.eval_method == 'fold':
            self.test_df = pd.read_csv(self.test_file, sep=',')
            self.train_df = pd.read_csv(self.train_file, sep=',')
    
    def get_context_combinations(self):
        unique_contexts = []
        for context_col in self.context_list:
            unique_contexts.append(self.full_df[context_col].unique())
        
        context_combs = list(itertools.product(*unique_contexts))
        return context_combs
            
    def count_dimensions(self):
        user_id_to_offset_dict = {}
        item_id_to_offset_dict = {}
        item_id_context_to_offset_dict = {}
        user_sideinfo_offset_dict = {}
        item_sideinfo_offset_dict = {}
        context_offset_dict = {}
        
        for column in [self.userid_column_name]:
            for index, value in enumerate(self.full_df[column].unique()):
                user_id_to_offset_dict[value] = index
        
        offset = 0
        for column in [self.itemid_column_name]:
            for index, value in enumerate(self.full_df[column].unique()):
                item_id_to_offset_dict[value] = index
                
                # Build itemsplit dictionary
                if len(self.context_combinations) > 0:
                    for context_comb in self.context_combinations:
                        item_id_context_to_offset_dict[(value, context_comb)] = offset
                        offset += 1 
                        
                else:
                    item_id_context_to_offset_dict[(value,)] = offset
                    offset += 1 

        offset = 0
        for column in self.user_sideinfo_columns:
            for value in self.full_df[column].unique():
                user_sideinfo_offset_dict[column + str(value)] = offset
                offset += 1

        genre_offset = 0
        for column in self.genrelist:
            item_sideinfo_offset_dict[column + str(1)] = genre_offset
            genre_offset += 1

        offset = 0
        for column in self.context_list:
            for value in self.full_df[column].unique():
                context_offset_dict[column + str(value)] = offset
                offset += 1

        self.user_id_to_offset_dict = user_id_to_offset_dict
        self.user_offset_to_id_dict = dict((v,k) for k,v in user_id_to_offset_dict.items())
        self.item_id_to_offset_dict = item_id_to_offset_dict
        self.item_offset_to_id_dict = dict((v,k) for k,v in item_id_to_offset_dict.items())
        self.item_id_context_to_offset_dict = item_id_context_to_offset_dict
        self.n_items_split = len(item_id_context_to_offset_dict.keys())
        self.user_sideinfo_offset_dict = user_sideinfo_offset_dict
        self.item_sideinfo_offset_dict = item_sideinfo_offset_dict
        self.context_offset_dict = context_offset_dict
        
    def build_dictionaries(self):
        user_ground_truth_dict = dict()
        train_set_user_pos_interactions = dict()
        train_set_user_neg_interactions = dict()
        user_sideinfo_dict = dict()
        item_sideinfo_dict = dict()

        print('  - Test_df dictionaries')
        for _, row in self.test_df.iterrows():
            # Add interactions as ground truths
            if row[self.userid_column_name] not in user_ground_truth_dict:
                user_ground_truth_dict[row[self.userid_column_name]] = [
                    row[self.itemid_column_name]]
                # If user has not been observed yet, save sideinfo
                user_sideinfo_indexes = []
                for column in self.user_sideinfo_columns:
                    user_sideinfo_indexes.append(
                        self.user_sideinfo_offset_dict[column + str(row[column])])
                user_sideinfo_dict[row[self.userid_column_name]
                                   ] = user_sideinfo_indexes
            else:
                if row[self.itemid_column_name] not in user_ground_truth_dict[row[self.userid_column_name]]:
                    user_ground_truth_dict[row[self.userid_column_name]].append(
                        row[self.itemid_column_name])

            if row[self.itemid_column_name] not in item_sideinfo_dict:
                # If movie has not been observed yet, save sideinfo
                item_sideinfo_indexes = []
                for column in self.genrelist:
                    if row[column] == 1:
                        item_sideinfo_indexes.append(
                            self.item_sideinfo_offset_dict[column + str(1)])
                item_sideinfo_dict[row[self.itemid_column_name]
                                   ] = item_sideinfo_indexes

        print('  - Train_df dictionaries')
        for _, row in self.train_df.iterrows():
            pos_interaction = (row[self.itemid_column_name],)
            contexts = tuple()
            for context in self.context_list:
                contexts = contexts + (row[context],)
            pos_interaction = pos_interaction + (contexts,)
            
            if row[self.userid_column_name] not in train_set_user_pos_interactions:
                train_set_user_pos_interactions[row[self.userid_column_name]] = [
                    pos_interaction]
                # If user has not been observed yet, save sideinfo
                user_sideinfo_indexes = []
                for column in self.user_sideinfo_columns:
                    user_sideinfo_indexes.append(
                        self.user_sideinfo_offset_dict[column + str(row[column])])
                user_sideinfo_dict[row[self.userid_column_name]
                                   ] = user_sideinfo_indexes
            else:
                train_set_user_pos_interactions[row[self.userid_column_name]].append(
                    pos_interaction)

            if row[self.itemid_column_name] not in item_sideinfo_dict:
                # If movie has not been observed yet, save sideinfo
                item_sideinfo_indexes = []
                for column in self.genrelist:
                    if row[column] == 1:
                        item_sideinfo_indexes.append(
                            self.item_sideinfo_offset_dict[column + str(1)])
                item_sideinfo_dict[row[self.itemid_column_name]
                                   ] = item_sideinfo_indexes

        print("  - Negative interactions")
        unique_train_items = self.train_df[self.itemid_column_name].unique()
        for key, value in train_set_user_pos_interactions.items():
            train_set_user_neg_interactions[key] = list(
                set(unique_train_items).difference(value[0]))

        self.user_sideinfo_dict = user_sideinfo_dict
        self.item_sideinfo_dict = item_sideinfo_dict
        # dict with key --> (test userId), value --> (ground truth interactions)
        self.user_ground_truth_dict = user_ground_truth_dict
        # dict with key --> (train userId), value --> (list of positive interaction Ids)
        self.train_set_user_pos_interactions = train_set_user_pos_interactions
        # dict with key --> (train userId), value --> (list of negative interaction Ids)
        self.train_set_user_neg_interactions = train_set_user_neg_interactions
        
    def sampler(self, batch_size):
        # Negative sampling, samples a random set of users in train set, finds a
        # positive and negative interaction for this user.
        # Repeat for size of batch

        user_ids, pos_interactions, neg_interactions = [], [], []

        random_userIds = random.choices(
            list(self.train_set_user_pos_interactions.keys()), k=batch_size)

        for userId in random_userIds:
            pos = random.choices(
                list(self.train_set_user_pos_interactions[userId]), k=1)[0]
            neg = random.choices(
                list(self.train_set_user_neg_interactions[userId]), k=1)[0]
            context = pos[1]
            neg_item_id = neg
            
            user_ids.append([self.user_id_to_offset_dict[userId]])
            pos_interactions.append([self.item_id_context_to_offset_dict[pos]])
            neg_interactions.append([self.item_id_context_to_offset_dict[(neg, context)]])

        return {'user_ids': user_ids, 'pos_interactions': pos_interactions, 'neg_interactions': neg_interactions}
        
    def _create_adj_mat(self):
        print("  - Adj matrix")        
        adj_mat_size = self.n_users + self.n_items_split
        adj_mat = sparse.dok_matrix(
            (adj_mat_size, adj_mat_size), dtype=np.float32)
        
        for _, row in self.train_df.iterrows():
            user_index = self.user_id_to_offset_dict[row[self.userid_column_name]]
            context = []
            for c in self.context_list:
                context.append(row[c])
            item_id = row[self.itemid_column_name]
            item_split_index = self.item_id_context_to_offset_dict[(item_id, tuple(context))]
            
            adj_mat[user_index, self.n_users + item_split_index] = 1
            adj_mat[self.n_users + item_split_index, user_index] = 1
        
        norm_adj_mat = self._normalize_adj_mat(adj_mat)
        
        return adj_mat, norm_adj_mat
    
    def _normalize_adj_mat(self, adj_mat):
        try:
            result = sparse.load_npz('norm_adj_mat.npz')
        except Exception as e:
            print(e)
            print("  - Norm Adj matrix")  
            mat_size = self.n_users + self.n_items_split
            diagonal_mat = sparse.dok_matrix(
                (mat_size, mat_size), dtype=np.float32)
            for i in range(mat_size):
                if i % 500 == 0:
                    print(i, 'of', mat_size, 'rows counted')
                non_zero_count = adj_mat[i].count_nonzero()
                diagonal_mat[i,i] = non_zero_count

            diag_mat_pow = diagonal_mat.power(-0.5)
            #D^-(1/2)*A*D^-(1/2)
            temp = diag_mat_pow.dot(adj_mat)
            
            result = temp.dot(diag_mat_pow)
            sparse.save_npz('adj_mat', adj_mat.tocsr())
            sparse.save_npz('norm_adj_mat', result.tocsr())    
        return result