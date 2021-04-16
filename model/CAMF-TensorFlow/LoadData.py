'''
Dataloader

'''
import pandas as pd
from scipy import sparse
import numpy as np
import random
from sklearn.preprocessing import normalize
import math as m


class LoadDataset():
    def __init__(self, random_seed, dataset='ml100k', eval_method='fold'):
        prefix = '../Graph-conv-context-aware-idea-2021-wip/'
        if dataset == 'ml100k':
            self.context_list = ['weekday', 'timeofday']
            self.userid_column_name = 'userId'
            self.itemid_column_name = 'movieId'
            self.path = prefix + 'Data/ml100k/'
        elif dataset == 'ml1m':
            self.context_list = ['weekday', 'timeofday']
            self.userid_column_name = 'userId'
            self.itemid_column_name = 'movieId'
            self.path = prefix + 'Data/ml1m/'
        elif dataset == 'frappe':
            self.context_list = ['weekday', 'timeofday', 'isweekend', 'weather', 'cost', 'country', 'city']
            self.userid_column_name = 'user'
            self.itemid_column_name = 'item'
            self.path = prefix + 'Data/Frappe/'
        elif dataset == 'yelpnc':
            self.context_list = ['hour', 'day_of_week']
            self.userid_column_name = 'user_id'
            self.itemid_column_name = 'business_id'
            self.path = prefix + 'Data/yelpnc/'
        elif dataset == 'yelpon':
            self.context_list = ['hour', 'day_of_week']
            self.userid_column_name = 'user_id'
            self.itemid_column_name = 'business_id'
            self.path = prefix + 'Data/yelpon/'
        else:
            print("No dataset defined")
            exit()

        self.eval_method = eval_method
        self.train_file = self.path + "train.txt"
        self.test_file = self.path + "test.txt"
        self.full_file = self.path + "out.txt"
        random.seed(random_seed)
        print("Loading data")
        self.load_data()  # Load train_df, test_df and full_data
        print("Counting dimensions")
        self.count_dimensions()
        print("Building dictionaries")
        self.build_dictionaries()
        self.n_train_users, self.n_test_users, self.n_users = self.user_counter()
        self.n_train_items, self.n_test_items, self.n_items = self.item_counter()
        self.n_context = self.context_counter()
        self.n_train_interactions = len(self.train_df.index)
        self.context_test_combinations = self.get_test_context_combinations()
        print(f"n_users: {self.n_users}")
        print(f"n_items: {self.n_items}")
        print(f"n_context: {self.n_context}")
        print(f"context combinations: {len(self.context_test_combinations)}")
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

    def count_dimensions(self):
        user_offset_dict = {}
        item_offset_dict = {}
        context_offset_dict = {}
        item_context_offset_dict = {}

        for column in [self.userid_column_name]:
            for index, value in enumerate(self.full_df[column].unique()):
                user_offset_dict[value] = index
        for column in [self.itemid_column_name]:
            for index, value in enumerate(self.full_df[column].unique()):
                item_offset_dict[value] = index

        offset = 0
        for column in self.context_list:
            for value in self.full_df[column].unique():
                context_offset_dict[column + str(value)] = offset
                offset += 1
        
        offset = 0
        for itemId in self.full_df[self.itemid_column_name].unique():
            for column in self.context_list:
                for value in self.full_df[column].unique():
                    item_context_offset_dict[(itemId, column + str(value))] = offset
                    offset += 1

        self.user_offset_dict = user_offset_dict
        self.user_offset_to_id_dict = dict((v,k) for k,v in user_offset_dict.items())
        self.item_offset_dict = item_offset_dict
        self.item_offset_to_id_dict = dict((v,k) for k,v in item_offset_dict.items())
        self.context_offset_dict = context_offset_dict
        self.item_offset_to_id_dict = item_context_offset_dict

    def build_dictionaries(self):
        user_ground_truth_dict = dict()
        train_set_user_pos_interactions = dict()
        train_set_user_neg_interactions = dict()

        print('  - Test_df dictionaries')
        for _, row in self.test_df.iterrows():
            # Add interactions as ground truths
            if row[self.userid_column_name] not in user_ground_truth_dict:
                user_ground_truth_dict[row[self.userid_column_name]] = [
                    row[self.itemid_column_name]]

            else:
                if row[self.itemid_column_name] not in user_ground_truth_dict[row[self.userid_column_name]]:
                    user_ground_truth_dict[row[self.userid_column_name]].append(
                        row[self.itemid_column_name])

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
            
            else:
                train_set_user_pos_interactions[row[self.userid_column_name]].append(
                    pos_interaction)


        print("  - Negative interactions")
        unique_train_items = self.train_df[self.itemid_column_name].unique()
        for key, value in train_set_user_pos_interactions.items():
            train_set_user_neg_interactions[key] = list(
                set(unique_train_items).difference(value[0]))

        
        self.user_ground_truth_dict = user_ground_truth_dict
        # dict with key --> (train userId), value --> (list of positive interaction Ids)
        self.train_set_user_pos_interactions = train_set_user_pos_interactions
        # dict with key --> (train userId), value --> (list of negative interaction Ids)
        self.train_set_user_neg_interactions = train_set_user_neg_interactions

    def sampler(self, alg_type, batch_size):
        # Negative sampling, samples a random set of users in train set, finds a
        # positive and negative interaction for this user.
        # Repeat for size of batch

        user_ids, pos_interactions, neg_interactions = [], [], []
        if alg_type == 'camf-c':
            contexts = []
        elif alg_type == 'camf-ci':
            pos_contexts, neg_contexts = [], []

        random_userIds = random.choices(
            list(self.train_set_user_pos_interactions.keys()), k=batch_size)

        for userId in random_userIds:
            pos = random.choices(
                list(self.train_set_user_pos_interactions[userId]), k=1)[0]
            neg = random.choices(
                list(self.train_set_user_neg_interactions[userId]), k=1)[0]

            user_ids.append(self.user_offset_dict[userId])
            pos_interactions.append(self.item_offset_dict[pos[0]])
            neg_interactions.append(self.item_offset_dict[neg])
            if alg_type == 'camf-c':
                context = []
                
                for index, context_col in enumerate(self.context_list, 0):
                    context.append(
                        self.context_offset_dict[context_col + str(pos[1][index])])
                contexts.append(context)


                
            elif alg_type == 'camf-ci':
                pos_context, neg_context = [], []
                
                for index, context_col in enumerate(self.context_list, 0):
                    pos_context.append(
                        self.item_offset_to_id_dict[(pos[0], context_col + str(pos[1][index]))])
                    neg_context.append(
                        self.item_offset_to_id_dict[(neg, context_col + str(pos[1][index]))])
                pos_contexts.append(pos_context)
                neg_contexts.append(neg_context)
                    
        if alg_type == 'camf-c':
            return {'user_ids': user_ids, 'pos_interactions': pos_interactions, 'neg_interactions': neg_interactions,
            'contexts': contexts}
        elif alg_type == 'camf-ci':
            return {'user_ids': user_ids, 'pos_interactions': pos_interactions, 'neg_interactions': neg_interactions,
                    'pos_contexts': pos_contexts, 'neg_contexts': neg_contexts}
                
                
    def get_test_context_combinations(self):
        # get the unique context combinations in the test set
        combinations = set()
        
        for _, row in self.test_df.iterrows():
            context_list = []
            for context in self.context_list:
                context_list.append(self.context_offset_dict[context + str(row[context])])
            combinations.add(tuple(context_list))
        return combinations