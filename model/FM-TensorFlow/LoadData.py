'''
Dataloader

'''
import pandas as pd
from scipy import sparse
import numpy as np

class LoadMovieLens():

    def __init__(self):
        self.path = 'Data/ml100k/'
        self.train_file = self.path + "train.txt"
        self.test_file = self.path + "test.txt"
        self.load_data() # Load train_df, test_df and full_data
        self.count_dimensions()
        self.n_users = self.user_counter()
        self.n_items = self.item_counter()
        print(f"n_users: {self.n_users}")
        print(f"n_items: {self.n_items}")
        self.one_hot()

    def user_counter(self):
        return self.full_data['userId'].nunique()

    def item_counter(self):
        return self.full_data['movieId'].nunique()

    def load_data(self):
        self.train_df = pd.read_csv(self.train_file, sep=',')
        self.test_df = pd.read_csv(self.test_file, sep=',')
        self.full_data = self.train_df.append(self.test_df)

    def count_dimensions(self):
        lookup_dict = {}

        offset = 0
        genre_count = 0
        for column_name in self.full_data:
            if column_name in ['userId', 'age', 'gender', 'occupation', 'zipcode', 'movieId']:
                unique_vals = self.full_data[column_name].unique()
                for value in unique_vals:
                    lookup_dict[str(column_name) + str(value)] = offset
                    offset += 1
            else:
                lookup_dict[str(column_name) + str(1)] = offset
                offset += 1
                genre_count += 1

        self.user_fields = self.full_data['userId'].nunique() + \
            self.full_data['age'].nunique() + self.full_data['gender'].nunique() + \
            self.full_data['occupation'].nunique() + self.full_data['zipcode'].nunique()
        
        self.item_fields = self.full_data['movieId'].nunique() + genre_count


        self.lookup_dict = lookup_dict
        self.field_sizes = len(self.lookup_dict)

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
