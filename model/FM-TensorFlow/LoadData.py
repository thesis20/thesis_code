'''
Dataloader

'''
import pandas as pd
from scipy import sparse
import numpy as np

class LoadMovieLens():
    def __init__(self):
        self.genrelist = ['unknown', 'action', 'adventure', 'animation',
                        'childrens', 'comedy', 'crime', 'documentary',
                        'drama', 'fantasy',  'film-noir', 'horror',
                        'musical', 'mystery', 'romance', 'scifi',
                        'thriller', 'war', 'western']
        self.user_feature_columns = ['userId', 'age', 'gender', 'occupation', 'zipcode']
        self.path = 'Data/ml100k/'
        self.train_file = self.path + "train.txt"
        self.test_file = self.path + "test.txt"
        self.load_data() # Load train_df, test_df and full_data
        self.count_dimensions()
        self.build_dictionaries()
        self.n_train_users, self.n_test_users, self.n_users = self.user_counter()
        self.n_train_items, self.n_test_items, self.n_items = self.item_counter()
        self.n_interactions = len(self.train_df.index)
        print(f"n_users: {self.n_users}")
        print(f"n_items: {self.n_items}")
        
        # self.resave_data()
        # self.one_hot()

    def resave_data(self):
        line = ""
        for _, row in self.test_df.iterrows():
            userId = row['userId']
            userfeatures = self.user_feature_dict[userId]
            
            movieId = row['movieId']
            moviefeatures = self.item_feature_dict[movieId]
    
            for index, feature in enumerate(userfeatures):
                if index == 0:
                    line += str(feature)
                else:
                    line += "-" + str(feature)
            line += ','

            for index, feature in enumerate(moviefeatures):
                if index == 0:
                    line += str(feature)
                else:
                    line += "-" + str(feature)
            line += '\n'
            
        trainfile = open('test.csv', 'w')
        trainfile.write(line)
        trainfile.close()

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

    def load_data(self):
        self.train_df = pd.read_csv(self.train_file, sep=',')
        self.test_df = pd.read_csv(self.test_file, sep=',')
        self.full_df = self.train_df.append(self.test_df)

    def count_dimensions(self):
        lookup_dict = {}
        user_feature_dict = {}
        item_feature_dict = {}

        offset = 0
        user_feature_dict_offset = 0
        item_feature_dict_offset = 0
        genre_count = 0
        for column_name in self.full_df:
            unique_vals = self.full_df[column_name].unique()
            if column_name in self.user_feature_columns:
                for value in unique_vals:
                    lookup_dict[str(column_name) + str(value)] = offset
                    offset += 1
                    user_feature_dict[str(column_name) + str(value)] = user_feature_dict_offset
                    user_feature_dict_offset += 1
                    
            elif column_name == 'movieId':
                for value in unique_vals:
                    lookup_dict[str(column_name) + str(value)] = offset
                    offset += 1
                    item_feature_dict[str(column_name) + str(value)] = item_feature_dict_offset
                    item_feature_dict_offset += 1
            else:
                lookup_dict[str(column_name) + str(1)] = offset
                item_feature_dict[str(column_name) + str(1)] = item_feature_dict_offset
                offset += 1
                item_feature_dict_offset += 1
                genre_count += 1


        self.user_fields = len(user_feature_dict)
        self.item_fields = len(item_feature_dict)
        self.user_feature_index_dict = user_feature_dict
        self.item_feature_index_dict = item_feature_dict

        self.lookup_dict = lookup_dict
    
    def build_dictionaries(self):
        user_feature_dict = dict()
        item_feature_dict = dict()
        user_ground_truth_dict = dict()
        train_set_user_pos_interactions = dict()
        train_set_user_neg_interactions = dict()
        
        for index, row in self.full_df.iterrows():
            user_feature_indexes = []
            if row['userId'] not in user_feature_dict:
                for column in self.user_feature_columns:
                    user_feature_indexes.append(self.user_feature_index_dict[column + str(row[column])])
                user_feature_dict[row['userId']] = user_feature_indexes

            if row['movieId'] not in item_feature_dict:
                genres = []
                movie_id_index = self.item_feature_index_dict['movieId' + str(row['movieId'])]
                for genre in self.genrelist:
                    if row[genre] == 1:
                        genre_index = self.item_feature_index_dict[genre + str(1)]
                        genres.append(genre_index)
                item_feature_dict[row['movieId']] = [movie_id_index] + genres

        longest_genre_list = len(item_feature_dict[max(item_feature_dict, key=lambda x: len(item_feature_dict[x]))])
        test = item_feature_dict[max(item_feature_dict, key=lambda x: len(item_feature_dict[x]))]
        for key, value in item_feature_dict.items():
            item_feature_dict[key] = (value + longest_genre_list * [0])[:longest_genre_list]


        for _, row in self.test_df.iterrows():
            if row['userId'] not in user_ground_truth_dict:
                user_ground_truth_dict[row['userId']] = [row['movieId']]
            else:
                if row['movieId'] not in user_ground_truth_dict[row['userId']]:
                    user_ground_truth_dict[row['userId']].append(row['movieId'])
        
        for _, row in self.train_df.iterrows():
            if row['userId'] not in train_set_user_pos_interactions:
                train_set_user_pos_interactions[row['userId']] = [row['movieId']]
            else:
                if row['movieId'] not in train_set_user_pos_interactions[row['userId']]:
                    train_set_user_pos_interactions[row['userId']].append(row['movieId'])
        
        unique_train_items = self.train_df['movieId'].unique()
        for key, value in train_set_user_pos_interactions.items():
            train_set_user_neg_interactions[key] = list(set(unique_train_items).difference(value))
        
        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.user_ground_truth_dict = user_ground_truth_dict
        self.train_set_user_pos_interactions = train_set_user_pos_interactions
        self.train_set_user_neg_interactions = train_set_user_neg_interactions

    def one_hot(self):
        train_col = []
        train_row = []

        for index, row in self.train_df.iterrows():
            for column_name in self.train_df:
                value = row[column_name]
                if value != 0:
                    train_row.append(index)
                    train_col.append(self.lookup_dict[str(column_name)+str(value)])

        train_data = np.ones(len(train_col))
        coo_train = sparse.coo_matrix((train_data, (train_row, train_col)), shape=(len(self.train_df), len(self.lookup_dict)), dtype=np.int8)

        test_col = []
        test_row = []

        for index, row in self.test_df.iterrows():
            for column_name in self.test_df:
                value = row[column_name]
                if value != 0:
                    test_row.append(index)
                    test_col.append(self.lookup_dict[str(column_name)+str(value)])

        test_data = np.ones(len(test_col))
        coo_test = sparse.coo_matrix((test_data, (test_row, test_col)), shape=(len(self.test_df), len(self.lookup_dict)), dtype=np.int8)

        self.lil_train = coo_train.tolil()
        self.lil_test = coo_test.tolil()
