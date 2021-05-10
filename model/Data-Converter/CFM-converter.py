import pandas as pd



class CFMConverter():
    def __init__(self, path, item_column_name):
        self.train = pd.read_csv(path + '/train.txt', sep=',', )
        self.test = pd.read_csv(path + '/test.txt', sep=',', )
        self.ratings = pd.read_csv(path + '/out.txt', sep=',', )
        self.train_items = self.train[item_column_name]
        self.test_items = self.test[item_column_name]
        self.all_items = self.ratings[item_column_name]
        self.train.drop(item_column_name, inplace=True, axis=1)
        self.test.drop(item_column_name, inplace=True, axis=1)
        self.ratings.drop(item_column_name, inplace=True, axis=1)

        if path == 'yelpnc':
            self.train.drop('date', inplace=True, axis=1)
            self.test.drop('date', inplace=True, axis=1)
            self.ratings.drop('date', inplace=True, axis=1)
            self.train.drop('review_count_x', inplace=True, axis=1)
            self.test.drop('review_count_x', inplace=True, axis=1)
            self.ratings.drop('review_count_x', inplace=True, axis=1)
            self.train.drop('review_count_y', inplace=True, axis=1)
            self.test.drop('review_count_y', inplace=True, axis=1)
            self.ratings.drop('review_count_y', inplace=True, axis=1)
            self.train.drop('state', inplace=True, axis=1)
            self.test.drop('state', inplace=True, axis=1)
            self.ratings.drop('state', inplace=True, axis=1)
            self.train.drop('month', inplace=True, axis=1)
            self.test.drop('month', inplace=True, axis=1)
            self.ratings.drop('month', inplace=True, axis=1)
            self.train.drop('timeofday', inplace=True, axis=1)
            self.test.drop('timeofday', inplace=True, axis=1)
            self.ratings.drop('timeofday', inplace=True, axis=1)
        
        if path =='ml1m':
            self.train.drop('zipcode', inplace=True, axis=1)
            self.test.drop('zipcode', inplace=True, axis=1)
            self.ratings.drop('zipcode', inplace=True, axis=1)
            self.train.drop('timestamp', inplace=True, axis=1)
            self.test.drop('timestamp', inplace=True, axis=1)
            self.ratings.drop('timestamp', inplace=True, axis=1)
        


        self.n_items = self.all_items.nunique()

        convert_frappe(self)
    
 
def get_column_offsets(self, path):
    if path == 'Frappe':
        user_offset = self.ratings['user'].nunique()
        daytime_offset = self.ratings['timeofday'].nunique()
        weekday_offset = self.ratings['weekday'].nunique()
        isweekend_offset = self.ratings['isweekend'].nunique()
        cost_offset = self.ratings['cost'].nunique()
        weather_offset = self.ratings['weather'].nunique()
        country_offset = self.ratings['country'].nunique()
        city_offset = self.ratings['city'].nunique()
        all_offsets = [user_offset,weekday_offset,isweekend_offset,cost_offset,weather_offset,country_offset,city_offset, daytime_offset]
        one_hot_length = sum(all_offsets)
    elif path == 'ml1m':
        user_offset = self.ratings['userId'].nunique()
        daytime_offset = self.ratings['timeofday'].nunique()
        weekday_offset = self.ratings['weekday'].nunique()
        age_offset = self.ratings['age'].nunique()
        gender_offset = self.ratings['gender'].nunique()
        occupation_offset = self.ratings['occupation'].nunique()
        #TODO: GENRE OFFSET
        all_offsets = [user_offset,daytime_offset,weekday_offset,age_offset,gender_offset,occupation_offset,]
    else:
        user_offset = self.ratings['user_id'].nunique()
        day_offset = self.ratings['day_of_week'].nunique()
        hour_offset = self.ratings['hour'].nunique()
        yelping_since_offset = self.ratings['yelping_since'].nunique()
        fans_offset = self.ratings['fans'].nunique()
        average_stars_offset = self.ratings['average_stars'].nunique()
        city_offset = self.ratings['city'].nunique()
        #TODO: BUSINESS GENRE OFFSET
        all_offsets = [user_offset,day_offset,hour_offset,yelping_since_offset,fans_offset,average_stars_offset,city_offset]
    
    return all_offsets


def create_city_dict_frappe(self):
    cities = self.ratings['city'].unique()
    city_onehot_dict = {}
    i = 0

    for city in cities:
        city_onehot_dict[city] = i
        i += 1
    
    return city_onehot_dict


def convert_frappe(self):
    city_onehot = create_city_dict_frappe(self)
    all_offsets = get_column_offsets(self, path)  
    cfm_train, cfm_test = [], []

    for i, row in self.train.iterrows():
        current_offset = 0
        one_hot_indices_user =  []

        for index, value in enumerate(row):
            # index 6 is city, need to look up the offset in the dict
            if index == 6:
                one_hot_indices_user.append(city_onehot[value] + current_offset)
                current_offset = current_offset + all_offsets[index] 
            else:
                one_hot_indices_user.append(value + current_offset) 
                current_offset = current_offset + all_offsets[index]


        cfm_string = (",".join(str(x) for x in one_hot_indices_user)).replace(',', '-')
        cfm_string = cfm_string + ',' + str(self.train_items[i]) + '-' + str(self.n_items) 
        cfm_train.append(cfm_string)

    for i, row in self.test.iterrows():
        current_offset = 0
        one_hot_indices_user =  []

        for index, value in enumerate(row):
            # index 6 is city, need to look up the offset in the dict
            if index == 6:
                one_hot_indices_user.append(city_onehot[value] + current_offset)
                current_offset = current_offset + all_offsets[index] 
            else:
                one_hot_indices_user.append(value + current_offset) 
                current_offset = current_offset + all_offsets[index]


        cfm_string = (",".join(str(x) for x in one_hot_indices_user)).replace(',', '-')
        cfm_string = cfm_string + ',' + str(self.test_items[i]) + '-' + str(self.n_items) 
        cfm_test.append(cfm_string)

    cfm_train_df = pd.DataFrame(cfm_train)
    cfm_test_df = pd.DataFrame(cfm_test)

    cfm_train_df.to_csv('train.csv', index=False)
    cfm_test_df.to_csv('test.csv', index=False)

def convert_ml1m(self):
    all_offsets = get_column_offsets(self, path)  
    cfm_train, cfm_test = [], []

    for i, row in self.train.iterrows():
        current_offset = 0
        one_hot_indices_user =  []

        for index, value in enumerate(row):
            # index 6 is city, need to look up the offset in the dict
            if index == 6:
                one_hot_indices_user.append(city_onehot[value] + current_offset)
                current_offset = current_offset + all_offsets[index] 
            else:
                one_hot_indices_user.append(value + current_offset) 
                current_offset = current_offset + all_offsets[index]


        cfm_string = (",".join(str(x) for x in one_hot_indices_user)).replace(',', '-')
        cfm_string = cfm_string + ',' + str(self.train_items[i]) + '-' + str(self.n_items) 
        cfm_train.append(cfm_string)

    for i, row in self.test.iterrows():
        current_offset = 0
        one_hot_indices_user =  []

        for index, value in enumerate(row):
            # index 6 is city, need to look up the offset in the dict
            if index == 6:
                one_hot_indices_user.append(city_onehot[value] + current_offset)
                current_offset = current_offset + all_offsets[index] 
            else:
                one_hot_indices_user.append(value + current_offset) 
                current_offset = current_offset + all_offsets[index]


        cfm_string = (",".join(str(x) for x in one_hot_indices_user)).replace(',', '-')
        cfm_string = cfm_string + ',' + str(self.test_items[i]) + '-' + str(self.n_items) 
        cfm_test.append(cfm_string)

    cfm_train_df = pd.DataFrame(cfm_train)
    cfm_test_df = pd.DataFrame(cfm_test)

    cfm_train_df.to_csv('train.csv', index=False)
    cfm_test_df.to_csv('test.csv', index=False)

#CFMConverter('Frappe', 'item')

#CFMConverter('yelpnc', 'business_id')
#CFMConverter('yelpon', 'business_id')
#CFMConverter('ml1m', 'movieId')