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

        # TODO: Tag som argument
        self.use_full_adj_mat = True

        if dataset == 'ml100k':
            self.genrelist = ['unknown', 'action', 'adventure', 'animation',
                              'childrens', 'comedy', 'crime', 'documentary',
                              'drama', 'fantasy',  'film-noir', 'horror',
                              'musical', 'mystery', 'romance', 'scifi',
                              'thriller', 'war', 'western']
            self.item_sideinfo_columns = ['genre']
            self.user_sideinfo_columns = [
                'age', 'gender', 'occupation']
            self.context_list = ['weekday', 'timeofday']
            self.userid_column_name = 'userId'
            self.itemid_column_name = 'movieId'
            self.path = 'Data/ml100k/'
        elif dataset == 'ml1m':
            self.genrelist = ['Action', 'Adventure', 'Animation', 'Children\'s',
                              'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                              'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
            self.item_sideinfo_columns = ['genre']
            self.user_sideinfo_columns = [
                'age', 'gender', 'occupation', 'zipcode']
            self.context_list = ['weekday', 'timeofday']
            self.userid_column_name = 'userId'
            self.itemid_column_name = 'movieId'
            self.path = 'Data/ml1m/'
        elif dataset == 'frappe':
            self.genrelist = []
            self.context_list = ['weekday', 'timeofday', 'isweekend', 'weather']
            self.item_sideinfo_columns = ['cost']
            self.user_sideinfo_columns = ['city']
            self.userid_column_name = 'user'
            self.itemid_column_name = 'item'
            self.path = 'Data/Frappe/'
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
        self.n_user_sideinfo, self.n_item_sideinfo = self.sideinfo_counter()
        self.n_context = self.context_counter()
        self.context_test_combinations = self.get_test_context_combinations()
        self.n_train_interactions = len(self.train_df.index)
        print("Creating adj matrices")

        if self.use_full_adj_mat:
            self.adj_mat, self.norm_adj_mat = self._create_full_adj_mat()
        else:
            self.adj_mat, self.us_adj_mat, self.is_adj_mat, self.norm_adj_mat, self.norm_us_adj_mat, self.norm_is_adj_mat = self._create_adj_mat()
        print(f"n_users: {self.n_users}")
        print(f"n_items: {self.n_items}")
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
        if 'genre' in self.item_sideinfo_columns:
            n_item_sideinfo = len(self.genrelist) + len(self.item_sideinfo_columns) - 1
        else:
            # TODO: Hardcoded 1 fordi at der kan være 2 værdier pr sideinfo
            n_item_sideinfo = len(self.item_sideinfo_columns) + 1
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

    def count_dimensions(self):
        user_offset_dict = {}
        item_offset_dict = {}
        user_sideinfo_offset_dict = {}
        item_sideinfo_offset_dict = {}
        context_offset_dict = {}

        for column in [self.userid_column_name]:
            for index, value in enumerate(self.full_df[column].unique()):
                user_offset_dict[value] = index
        for column in [self.itemid_column_name]:
            for index, value in enumerate(self.full_df[column].unique()):
                item_offset_dict[value] = index

        offset = 0
        for column in self.user_sideinfo_columns:
            for value in self.full_df[column].unique():
                user_sideinfo_offset_dict[column + str(value)] = offset
                offset += 1

        offset = 0
        if 'genre' in self.item_sideinfo_columns:
            for column in self.genrelist:
                item_sideinfo_offset_dict[column + str(1)] = offset
                offset += 1

        for column in self.item_sideinfo_columns:
            if column == 'genre':
                continue
            for value in self.full_df[column].unique():
                item_sideinfo_offset_dict[column + str(value)] = offset
                offset += 1


        offset = 0
        for column in self.context_list:
            for value in self.full_df[column].unique():
                context_offset_dict[column + str(value)] = offset
                offset += 1

        self.user_offset_dict = user_offset_dict
        self.user_offset_to_id_dict = dict((v,k) for k,v in user_offset_dict.items())
        self.item_offset_dict = item_offset_dict
        self.item_offset_to_id_dict = dict((v,k) for k,v in item_offset_dict.items())
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

                if 'genre' in self.item_sideinfo_columns:
                    for column in self.genrelist:
                        if row[column] == 1:
                            item_sideinfo_indexes.append(
                                self.item_sideinfo_offset_dict[column + str(1)])
                    item_sideinfo_dict[row[self.itemid_column_name]
                                    ] = item_sideinfo_indexes

                for column in self.item_sideinfo_columns:
                    if column == 'genre':
                        continue
                    value = row[column]
                    item_sideinfo_indexes.append(self.item_sideinfo_offset_dict[column + str(value)])
                item_sideinfo_dict[row[self.itemid_column_name]] = item_sideinfo_indexes

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

                if 'genre' in self.item_sideinfo_columns:
                    for column in self.genrelist:
                        if row[column] == 1:
                            item_sideinfo_indexes.append(
                                self.item_sideinfo_offset_dict[column + str(1)])
                    item_sideinfo_dict[row[self.itemid_column_name]
                                    ] = item_sideinfo_indexes

                for column in self.item_sideinfo_columns:
                    if column == 'genre':
                        continue
                    value = row[column]
                    item_sideinfo_indexes.append(self.item_sideinfo_offset_dict[column + str(value)])
                item_sideinfo_dict[row[self.itemid_column_name]] = item_sideinfo_indexes

        print("  - Negative interactions")
        unique_train_items = self.train_df[self.itemid_column_name].unique()
        for key, value in train_set_user_pos_interactions.items():
            train_set_user_neg_interactions[key] = list(
                set(unique_train_items).difference(value[0]))

        # TODO Fjern padding
        # longest_genre_list = len(item_sideinfo_dict[max(
        #     item_sideinfo_dict, key=lambda x: len(item_sideinfo_dict[x]))])
        # for key, value in item_sideinfo_dict.items():
        #     item_sideinfo_dict[key] = (
        #         value + longest_genre_list * [0])[:longest_genre_list]

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

        random_userIds = random.choices(
            list(self.train_set_user_pos_interactions.keys()), k=batch_size)

        for userId in random_userIds:
            pos = random.choices(
                list(self.train_set_user_pos_interactions[userId]), k=1)[0]
            neg = random.choices(
                list(self.train_set_user_neg_interactions[userId]), k=1)[0]

            user_ids.append([self.user_offset_dict[userId]])
            pos_interactions.append([self.item_offset_dict[pos[0]]])
            neg_interactions.append([self.item_offset_dict[neg]])
            user_contexts = []

            for index, context in enumerate(self.context_list, 0):
                user_contexts.append(
                    self.context_offset_dict[context + str(pos[1][index])])
            contexts.append(user_contexts)
            user_sideinfo.append(self.user_sideinfo_dict[userId])
            
            sideinfo = self.item_sideinfo_dict[pos[0]]
            # TODO if the item has more than one genre, choose a random one
            if 'genre' in self.item_sideinfo_columns:
                if len(sideinfo) > 1:
                    sideinfo = [random.choice(sideinfo)]
            item_sideinfo.append(sideinfo)

        return {'user_ids': user_ids, 'pos_interactions': pos_interactions, 'neg_interactions': neg_interactions,
                'contexts': contexts, 'user_sideinfo': user_sideinfo, 'item_sideinfo': item_sideinfo}

    # TODO Husk at switche på den her
    def _create_full_adj_mat(self):
        print(" - Full adj matrix")
        adj_mat_size = self.n_users + self.n_items + 2*self.n_context + self.n_user_sideinfo

        adj_mat = sparse.dok_matrix((adj_mat_size, adj_mat_size), dtype=np.float32)

        for _, row in self.train_df.iterrows():
            userId = row[self.userid_column_name]
            itemId = row[self.itemid_column_name]
            user_index = self.user_offset_dict[userId]
            item_index = self.item_offset_dict[itemId]
            user_sideinfo_indexes = self.user_sideinfo_dict[userId]
            #item_sideinfo_indexes = self.item_sideinfo_dict[itemId]
            context_indexes = [self.context_offset_dict[column +
                                            str(row[column])] for column in self.context_list]

            item_offset = self.n_users + item_index

            for context_index in context_indexes:
                context_offset = self.n_users + self.n_items + context_index
                adj_mat[user_index, context_offset] = 1
                adj_mat[context_offset, user_index] = 1
                adj_mat[item_offset, context_offset + self.n_context] = 1
                adj_mat[context_offset + self.n_context, item_offset] = 1

            for user_sideinfo in user_sideinfo_indexes:
                user_sideinfo_offset = self.n_users + self.n_items + 2*self.n_context + user_sideinfo
                adj_mat[user_index, user_sideinfo_offset] = 1
                adj_mat[user_sideinfo_offset, user_index] = 1

            #for item_sideinfo in item_sideinfo_indexes:
             #   item_sideinfo_offset = self.n_users + self.n_items + 2*self.n_context + self.n_user_sideinfo + item_sideinfo
              #  adj_mat[item_index, item_sideinfo_offset] = 1
               # adj_mat[item_sideinfo_offset, item_index] = 1

            adj_mat[user_index, item_offset] = 1
            adj_mat[item_offset, user_index] = 1

        # # add I for sideinfo
        # for i in range(self.n_user_sideinfo + self.n_item_sideinfo):
        #     offset = self.n_users + self.n_items + self.n_context
        #     adj_mat[i + offset,i + offset] = 1

        # Add 2I for context
        for i in range(2 * self.n_context):
            offset = self.n_users + self.n_items
            adj_mat[i + offset,i + offset] = 1


        norm_adj_mat = self._normalize_adj_matrix(adj_mat)

        return adj_mat, norm_adj_mat

    def _create_adj_mat(self):
        print("  - Adj matrix")
        adj_mat_size = self.n_users + self.n_items + self.n_context
        adj_mat = sparse.dok_matrix(
            (adj_mat_size, adj_mat_size), dtype=np.float32)

        user_sideinfo_adj_mat_size = self.n_users + self.n_user_sideinfo
        item_sideinfo_adj_mat_size = self.n_items + self.n_item_sideinfo
        user_sideinfo_adj_mat = sparse.dok_matrix(
            (user_sideinfo_adj_mat_size, user_sideinfo_adj_mat_size), dtype=np.float32)
        item_sideinfo_adj_mat = sparse.dok_matrix(
            (item_sideinfo_adj_mat_size, item_sideinfo_adj_mat_size), dtype=np.float32)

        for _, row in self.train_df.iterrows():
            user_index = self.user_offset_dict[row[self.userid_column_name]]
            item_index = self.item_offset_dict[row[self.itemid_column_name]]
            context_indexes = [self.context_offset_dict[column +
                                                        str(row[column])] for column in self.context_list]

            item_offset = self.n_users + item_index
            user_offset = user_index + self.n_users

            for context_index in context_indexes:
                context_offset = self.n_users + self.n_items + context_index
                adj_mat[user_index, context_offset] = 1
                adj_mat[item_offset, context_offset] = 1

            try:
                adj_mat[user_index, item_offset] = 1  # R
                adj_mat[item_offset, user_index] = 1  # Rt
            except Exception as e:
                print(
                    f'user: {user_index}, item_index: {item_index}, item_offset: {item_offset}, user_offset: {user_offset}')
                raise e
            
        #add 2 times identity matrix
        for i in range(self.n_context):
            offset = self.n_users + self.n_items
            adj_mat[i + offset,i + offset] = 2

            #   U  I  C
            # U 0  R  uc
            # I Rt 0  ic
            # C 0  0  2I

        print("  - User sideinfo matrix")
        for userId in self.train_df[self.userid_column_name].unique():
            sideinfo_indexes = self.user_sideinfo_dict[userId]
            user_index = self.user_offset_dict[userId]

            for sideinfo_index in sideinfo_indexes:
                user_offset = sideinfo_index + self.n_users
                user_sideinfo_adj_mat[user_index, user_offset] = 1
                user_sideinfo_adj_mat[user_offset, user_index] = 1

        print("  - Item sideinfo matrix")
        for movieId in self.train_df[self.itemid_column_name].unique():
            sideinfo_indexes = self.item_sideinfo_dict[movieId]
            item_index = self.item_offset_dict[movieId]

            for sideinfo_index in sideinfo_indexes:
                item_offset = sideinfo_index + self.n_items
                item_sideinfo_adj_mat[item_index, item_offset] = 1
                item_sideinfo_adj_mat[item_offset, item_index] = 1

        print("  - Normalizing user sideinfo adj")
        norm_us_adj_mat = self.normalize_sideinfo(user_sideinfo_adj_mat)
        
        print("  - Normalizing item sideinfo adj")
        norm_is_adj_mat = self.normalize_sideinfo(item_sideinfo_adj_mat)
        
        print("  - Normalizing full adj")
        norm_adj_mat = self._normalize_adj_matrix(adj_mat)

        return adj_mat, user_sideinfo_adj_mat, item_sideinfo_adj_mat, norm_adj_mat, norm_us_adj_mat, norm_is_adj_mat


    def _normalize_adj_matrix(self, x):
        print(" - Normalizing")
        diagonal_mat = sparse.dok_matrix(
            x.shape, dtype=np.float32)
        for i in range(x.shape[0]):
            non_zero_count = x[i].count_nonzero()
            diagonal_mat[i,i] = non_zero_count

        frac_diagonal_mat = diagonal_mat.power(-0.5)
        #sqrt(2)*D^-(1/2)*A
        frac_mul_adj_mat = frac_diagonal_mat.dot(x)
        result = m.sqrt(2)*frac_mul_adj_mat
            
        return result
        

    def normalize_sideinfo(self, x):
        matrix_copy = sparse.dok_matrix(x.shape, dtype=np.float32)
        rows, cols, _ = sparse.find(x)
        entries = list(zip(rows, cols))
        counter = {}

        for row, col in entries:
            if row in counter:
                if x[row, col] == 1:
                    counter[row] += 1
            else:
                if x[row, col] == 1:
                    counter[row] = 1

        for row, col in entries:
            if x[row, col] == 1:
                matrix_copy[row, col] = 1.0/counter[row]

        return matrix_copy

    def get_test_context_combinations(self):
        # get the unique context combinations in the test set
        combinations = set()
        
        for _, row in self.test_df.iterrows():
            context_list = []
            for context in self.context_list:
                context_list.append(self.context_offset_dict[context + str(row[context])])
            combinations.add(tuple(context_list))
        return combinations