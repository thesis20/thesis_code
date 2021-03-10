'''
Dataloader

'''
import pandas as pd
from scipy import sparse
import numpy as np
import random
from sklearn.preprocessing import normalize

class LoadMovieLens():
    def __init__(self, random_seed):
        self.genrelist = ['unknown', 'action', 'adventure', 'animation',
                        'childrens', 'comedy', 'crime', 'documentary',
                        'drama', 'fantasy',  'film-noir', 'horror',
                        'musical', 'mystery', 'romance', 'scifi',
                        'thriller', 'war', 'western']
        self.user_sideinfo_columns = ['age', 'gender', 'occupation', 'zipcode']
        self.context_list = ['weekday', 'timeofday']
        self.path = 'Data/ml100k/'
        self.train_file = self.path + "train.txt"
        self.test_file = self.path + "test.txt"
        random.seed(random_seed)
        self.load_data() # Load train_df, test_df and full_data
        self.count_dimensions()
        self.build_dictionaries()
        self.n_train_users, self.n_test_users, self.n_users = self.user_counter()
        self.n_train_items, self.n_test_items, self.n_items = self.item_counter()
        self.n_user_sideinfo, self.n_item_sideinfo = self.sideinfo_counter()
        self.n_context = self.context_counter()
        self.n_train_interactions = len(self.train_df.index)
        self.uic_adj_mat, self.us_adj_mat, self.is_adj_mat, self.norm_adj_mat = self._create_adj_mat()
        print(f"n_users: {self.n_users}")
        print(f"n_items: {self.n_items}")
        

    def user_counter(self):
        train_users = self.train_df['userId'].nunique()
        test_users = self.test_df['userId'].nunique()
        total_users = self.full_df['userId'].nunique()
        return train_users, test_users, total_users

    def item_counter(self):
        train_items = self.train_df['movieId'].nunique()
        test_items = self.test_df['movieId'].nunique()
        total_items = self.full_df['movieId'].nunique()
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
        self.train_df = pd.read_csv(self.train_file, sep=',')
        self.test_df = pd.read_csv(self.test_file, sep=',')
        self.full_df = self.train_df.append(self.test_df)

    def count_dimensions(self):
        user_offset_dict = {}
        item_offset_dict = {}
        user_sideinfo_offset_dict = {}
        item_sideinfo_offset_dict = {}
        context_offset_dict = {}
        
        for column in ['userId']:
            for index, value in enumerate(self.full_df[column].unique()):
                user_offset_dict[value] = index
        for column in ['movieId']:
            for index, value in enumerate(self.full_df[column].unique()):
                item_offset_dict[value] = index

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
                
        self.user_offset_dict = user_offset_dict
        self.item_offset_dict = item_offset_dict
        self.user_sideinfo_offset_dict = user_sideinfo_offset_dict
        self.item_sideinfo_offset_dict = item_sideinfo_offset_dict
        self.context_offset_dict = context_offset_dict
         
    def build_dictionaries(self):
        user_ground_truth_dict = dict()
        train_set_user_pos_interactions = dict()
        train_set_user_neg_interactions = dict()
        user_sideinfo_dict = dict()
        item_sideinfo_dict = dict()

        for _, row in self.test_df.iterrows():
            # Add interactions as ground truths 
            if row['userId'] not in user_ground_truth_dict:
                user_ground_truth_dict[row['userId']] = [row['movieId']]
                # If user has not been observed yet, save sideinfo
                user_sideinfo_indexes = []
                for column in self.user_sideinfo_columns:
                    user_sideinfo_indexes.append(self.user_sideinfo_offset_dict[column + str(row[column])])
                user_sideinfo_dict[row['userId']] = user_sideinfo_indexes
            else:
                if row['movieId'] not in user_ground_truth_dict[row['userId']]:
                    user_ground_truth_dict[row['userId']].append(row['movieId'])
            
            if row['movieId'] not in item_sideinfo_dict:
                # If movie has not been observed yet, save sideinfo
                item_sideinfo_indexes = []
                for column in self.genrelist:
                    if row[column] == 1:
                        item_sideinfo_indexes.append(self.item_sideinfo_offset_dict[column + str(1)])
                item_sideinfo_dict[row['movieId']] = item_sideinfo_indexes
        
        for _, row in self.train_df.iterrows():
            if row['userId'] not in train_set_user_pos_interactions:
                # TODO: This shouldn't be hard-coded
                train_set_user_pos_interactions[row['userId']] = [(row['movieId'], row['weekday'], row['timeofday'])]
                # If user has not been observed yet, save sideinfo
                user_sideinfo_indexes = []
                for column in self.user_sideinfo_columns:
                    user_sideinfo_indexes.append(self.user_sideinfo_offset_dict[column + str(row[column])])
                user_sideinfo_dict[row['userId']] = user_sideinfo_indexes
            else:
                train_set_user_pos_interactions[row['userId']].append((row['movieId'], row['weekday'], row['timeofday']))
                
            if row['movieId'] not in item_sideinfo_dict:
                # If movie has not been observed yet, save sideinfo
                item_sideinfo_indexes = []
                for column in self.genrelist:
                    if row[column] == 1:
                        item_sideinfo_indexes.append(self.item_sideinfo_offset_dict[column + str(1)])
                item_sideinfo_dict[row['movieId']] = item_sideinfo_indexes       
                 
        unique_train_items = self.train_df['movieId'].unique()
        for key, value in train_set_user_pos_interactions.items():
            train_set_user_neg_interactions[key] = list(set(unique_train_items).difference(value[0]))
        
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
        contexts, user_sideinfo, item_sideinfo = [], [], []
        
        random_userIds = random.choices(list(self.train_set_user_pos_interactions.keys()), k=batch_size)
        
        for userId in random_userIds:
            pos = random.choices(list(self.train_set_user_pos_interactions[userId]), k=1)[0]
            neg = random.choices(list(self.train_set_user_neg_interactions[userId]), k=1)[0]
            
            user_ids.append(self.user_offset_dict[userId])
            pos_interactions.append(self.item_offset_dict[pos[0]])
            neg_interactions.append(self.item_offset_dict[neg[0]])
            contexts.append([self.context_offset_dict[pos[1]], self.context_offset_dict[pos[2]]])
            user_sideinfo.append(self.user_sideinfo_dict[userId])
            item_sideinfo.append(self.item_sideinfo_dict[pos[0]])
            

        return { 'user_ids': user_ids, 'pos_interactions': pos_interactions, 'neg_interactions': neg_interactions,
                    'contexts': contexts, 'user_sideinfo': user_sideinfo, 'item_sideinfo': item_sideinfo}

    def _create_adj_mat(self):
        adj_mat_size = self.n_users + self.n_items + self.n_context
        adj_mat = sparse.dok_matrix((adj_mat_size, adj_mat_size), dtype=np.float32)
        
        user_sideinfo_adj_mat_size = self.n_users + self.n_user_sideinfo
        item_sideinfo_adj_mat_size = self.n_items + self.n_item_sideinfo
        user_sideinfo_adj_mat = sparse.dok_matrix((user_sideinfo_adj_mat_size, user_sideinfo_adj_mat_size), dtype=np.float32)
        item_sideinfo_adj_mat = sparse.dok_matrix((item_sideinfo_adj_mat_size, item_sideinfo_adj_mat_size), dtype=np.float32)
        
        for _, row in self.train_df.iterrows():
            user_index = self.user_offset_dict[row['userId']]
            item_index = self.item_offset_dict[row['movieId']]
            context_indexes = [self.context_offset_dict[column + str(row[column])] for column in self.context_list]
            
            item_offset = self.n_users + item_index
            user_offset = user_index + self.n_users

            for context_index in context_indexes:
                context_offset = self.n_users + self.n_items + context_index
                adj_mat[user_index, context_offset] = 1
                adj_mat[item_offset, context_offset] = 1
            
            adj_mat[user_index, item_offset] = 1 # R
            adj_mat[user_offset, item_index] = 1 # Rt

            #   U  I  C
            # U 0  R  uc
            # I Rt 0  ic
            # C 0  0  0
            
        for userId in self.train_df['userId'].unique():
            sideinfo_indexes = self.user_sideinfo_dict[userId]
            user_index = self.user_offset_dict[userId]
            
            for sideinfo_index in sideinfo_indexes:
                user_offset = sideinfo_index + self.n_users
                user_sideinfo_adj_mat[user_index, user_offset] = 1
                user_sideinfo_adj_mat[user_offset, user_index] = 1
                
        for movieId in self.train_df['movieId'].unique():
            sideinfo_indexes = self.item_sideinfo_dict[movieId]
            item_index = self.item_offset_dict[movieId]
            
            for sideinfo_index in sideinfo_indexes:
                item_offset = sideinfo_index + self.n_items
                item_sideinfo_adj_mat[item_index, item_offset] = 1
                item_sideinfo_adj_mat[item_offset, item_index] = 1

        norm_us_adj_mat = self.normalize_sideinfo(user_sideinfo_adj_mat)
        norm_is_adj_mat = self.normalize_sideinfo(item_sideinfo_adj_mat)
        print("here")
        norm_adj_mat = self._normalize_adj_matrix(adj_mat)
        
        return adj_mat, user_sideinfo_adj_mat, item_sideinfo_adj_mat, norm_adj_mat
    
    def _normalize_adj_matrix(self, x):
        
        matrix_copy = sparse.dok_matrix(x.shape, dtype=np.float32)
        places = sparse.find(x)
        
        rows, cols, values = places 
        
        entries = list(zip(rows, cols))
        
        ui_counter = dict()
        # UI
        # 1 2
        # 2 1
        # 3 1
        uc_counter = dict()
        ic_counter = dict()
        
        for row, col in entries:
            if row < self.n_users: #Rui and Ruc
                if col > self.n_users + self.n_items: # Ruc
                    if row in uc_counter:
                        uc_counter[row] += 1
                    else:
                        uc_counter[row] = 1
                else: # Rui
                    if row in ui_counter:
                        ui_counter[row] += 1
                    else:
                        ui_counter[row] = 1
            else:
                if col > self.n_users + self.n_items: # Ric
                    if row in ic_counter:
                        ic_counter[row] += 1
                    else:
                        ic_counter[row] = 1
                        
        for row, col in entries:
            if row < self.n_users: #Rui and Ruc
                if col > self.n_users + self.n_items: # Ruc
                    matrix_copy[row, col] = 1.0/uc_counter[row]
                else: # Rui and RuiT
                    matrix_copy[row, col] = 1.0/ui_counter[row]
                    matrix_copy[col, row] = 1.0/ui_counter[row]
            else:
                if col > self.n_users + self.n_items: # Ric
                    matrix_copy[row, col] = 1.0/ic_counter[row]
            
        return matrix_copy
        
    def normalize_sideinfo(self, x):
        matrix_copy = sparse.dok_matrix(x.shape, dtype=np.float32)
        places = sparse.find(x)
        rows, cols, values = places
        entries = list(zip(rows, cols))
        counter = {}
        
        for row, col in entries:
            if row in counter:
                if x[row,col] == 1:
                    counter[row] += 1
            else:
                if x[row,col] == 1:
                    counter[row] = 1
        
        for row, col in entries:
            if x[row,col] == 1:
                matrix_copy[row,col] = 1.0/counter[row]
         
        return matrix_copy
    