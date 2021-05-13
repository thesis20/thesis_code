import pandas as pd
import numpy as np



class CFMConverter():
    def __init__(self, path, item_column_name):
        self.train = pd.read_csv( path +  '/train.txt', sep=',', )
        self.test = pd.read_csv(path + '/test.txt', sep=',', )
        self.ratings = pd.read_csv(path + '/out.txt', sep=',', )
        self.train_items = self.train[item_column_name]
        self.test_items = self.test[item_column_name]
        self.all_items = self.ratings[item_column_name]
        self.train.drop(item_column_name, inplace=True, axis=1)
        self.test.drop(item_column_name, inplace=True, axis=1)
        self.ratings.drop(item_column_name, inplace=True, axis=1)
        self.n_items = self.all_items.nunique()
        self.path = path

        if path == 'yelpnc' or path == 'yelpon':
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
            self.relevant_columns = ['user_id', 'yelping_since', 'fans', 'average_stars', 'day_of_week', 'hour', 'city']
            self.train.drop_duplicates(inplace=True)
            convert_yelp(self, path)
        elif path =='ml1m':
            self.train.drop('zipcode', inplace=True, axis=1)
            self.test.drop('zipcode', inplace=True, axis=1)
            self.ratings.drop('zipcode', inplace=True, axis=1)
            self.train.drop('timestamp', inplace=True, axis=1)
            self.test.drop('timestamp', inplace=True, axis=1)
            self.ratings.drop('timestamp', inplace=True, axis=1)
            self.relevant_columns = ['userId', 'age', 'gender', 'occupation', 'weekday', 'timeofday']
            convert_ml1m(self, path)
        else:
            convert_frappe(self, path)
 
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
       
        #offset_pre_multihot = user_offset + daytime_offset + weekday_offset + age_offset + gender_offset + occupation_offset
        all_offsets = [user_offset, age_offset, gender_offset,occupation_offset,weekday_offset, daytime_offset]
    else:
        user_offset = self.ratings['user_id'].nunique()
        yelping_since_offset = self.ratings['yelping_since'].nunique()
        fans_offset = self.ratings['fans'].nunique()
        average_stars_offset = self.ratings['average_stars'].nunique()
        day_offset = self.ratings['day_of_week'].nunique()
        hour_offset = self.ratings['hour'].nunique()
        city_offset = self.ratings['city'].nunique()
        all_offsets = [user_offset, yelping_since_offset, fans_offset, average_stars_offset, day_offset, hour_offset, city_offset]
    
    return all_offsets

def multihot_conversion(self, mulithot_list, offsets, row, i, train_flag):
    multihot_value_list = []
    offset = sum(offsets)
    for i, value in enumerate(row):
        rowname = self.train.columns.values[i]
        if rowname in mulithot_list:
            if train_flag == 1:
                if value == 1:
                    offset += 1
                    multihot_value_list.append(offset)
    
    return multihot_value_list

def create_city_dict(self):
    cities = self.ratings['city'].unique()
    city_onehot_dict = {}
    i = 0

    for city in cities:
        city_onehot_dict[city] = i
        i += 1
    
    return city_onehot_dict

def create_yelping_since_dict(self):
    years = self.ratings['yelping_since'].unique()
    year_onehot_dict = {}
    i = 0

    for year in years:
        year_onehot_dict[year] = i
        i += 1
    
    return year_onehot_dict

def create_fans_dict(self):
    fans_nan = self.ratings['fans'].unique()
    #remove nan entry from entries in the dataset that have no value set
    if self.path =='yelpnc':
        fans = np.delete(fans_nan, -1)
    else:
        fans = fans_nan[~np.isnan(fans_nan)]
    fans_onehot_dict = {}
    i = 0

    for fan in fans:
        fans_onehot_dict[fan] = i
        i += 1
    
    return fans_onehot_dict

def create_gender_dict(self):
    genders = self.ratings['gender'].unique()
    gender_onehot_dict = {}
    i = 0

    for gender in genders:
        gender_onehot_dict[gender] = i
        i += 1
    
    return gender_onehot_dict

def create_stars_dict(self):
    stars = self.ratings['average_stars'].unique()
    stars_onehot_dict = {}
    i = 0

    for star in stars:
        stars_onehot_dict[star] = i
        i += 1
    
    return stars_onehot_dict

def create_hour_dict(self):
    hours = self.ratings['hour'].unique()
    hour_onehot_dict = {}
    i = 0

    for hour in hours:
        hour_onehot_dict[hour] = i
        i += 1
    
    return hour_onehot_dict

def create_weekday_dict(self):
    weekdays = self.ratings['day_of_week'].unique()
    weekday_onehot_dict = {}
    i = 0

    for wd in weekdays:
        weekday_onehot_dict[wd] = i
        i += 1
    
    return weekday_onehot_dict

def convert_frappe(self, path):
    city_onehot = create_city_dict(self)
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

def convert_ml1m(self, path):
    all_offsets = get_column_offsets(self, path)  
    gender_onehot = create_gender_dict(self)   
    cfm_train, cfm_test = [], []

    for i, row in self.train.iterrows():
        current_offset = 0
        one_hot_indices_user =  []
        index = 0
        
        for j, value in enumerate(row):
            rowname = self.train.columns.values[j]
            if rowname in self.relevant_columns:
                if rowname == 'gender':
                    one_hot_indices_user.append(gender_onehot[value] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                else:
                    one_hot_indices_user.append(row[j] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                index += 1

        cfm_string = (",".join(str(x) for x in one_hot_indices_user)).replace(',', '-')
        cfm_string = cfm_string + ',' + str(self.train_items[i]) + '-' + str(self.n_items) 
        cfm_train.append(cfm_string)

    for i, row in self.test.iterrows():
        current_offset = 0
        one_hot_indices_user =  []
        index = 0

        for j, value in enumerate(row):
            rowname = self.train.columns.values[j]
            if rowname in self.relevant_columns:
                if rowname == 'gender':
                    one_hot_indices_user.append(gender_onehot[value] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                else:
                    one_hot_indices_user.append(row[j] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                index += 1

        cfm_string = (",".join(str(x) for x in one_hot_indices_user)).replace(',', '-')
        cfm_string = cfm_string + ',' + str(self.test_items[i]) + '-' + str(self.n_items) 
        cfm_test.append(cfm_string)

    cfm_train_df = pd.DataFrame(cfm_train)
    cfm_test_df = pd.DataFrame(cfm_test)

    cfm_train_df.to_csv('train.csv', index=False)
    cfm_test_df.to_csv('test.csv', index=False)

def convert_yelp(self, path):
    city_onehot = create_city_dict(self)
    yelping_onehot = create_yelping_since_dict(self)
    fans_onehot = create_fans_dict(self)
    stars_onehot = create_stars_dict(self)
    weekday_onehot = create_weekday_dict(self)
    hour_onehot = create_hour_dict(self)
    all_offsets = get_column_offsets(self, path)  
    
    cfm_train, cfm_test = [], []

    for i, row in self.train.iterrows():
        print("asdf")
        current_offset = 0
        one_hot_indices_user =  []
        index = 0

        for j, value in enumerate(row):
            rowname = self.train.columns.values[j]
            if rowname in self.relevant_columns:
                if rowname == 'city':
                    one_hot_indices_user.append(city_onehot[value] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                elif rowname == 'yelping_since':
                    one_hot_indices_user.append(yelping_onehot[value] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                elif rowname == 'fans' and not np.isnan(value):
                    one_hot_indices_user.append(fans_onehot[value] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                elif rowname == 'fans' and np.isnan(value):
                    one_hot_indices_user.append(0 + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                elif rowname == 'average_stars':
                    one_hot_indices_user.append(stars_onehot[value] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                elif rowname == 'day_of_week':
                    one_hot_indices_user.append(weekday_onehot[value] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                elif rowname == 'hour':
                    one_hot_indices_user.append(hour_onehot[value] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                else:
                    one_hot_indices_user.append(row[j] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                index += 1

        cfm_string = (",".join(str(x) for x in one_hot_indices_user)).replace(',', '-')
        cfm_string = cfm_string + ',' + str(self.train_items[i]) + '-' + str(self.n_items) 
        cfm_train.append(cfm_string)

    for i, row in self.test.iterrows():
        current_offset = 0
        one_hot_indices_user =  []
        index = 0

        for j, value in enumerate(row):
            rowname = self.train.columns.values[j]
            if rowname in self.relevant_columns:
                if rowname == 'city':
                    one_hot_indices_user.append(city_onehot[value] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                elif rowname == 'yelping_since':
                    one_hot_indices_user.append(yelping_onehot[value] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                elif rowname == 'fans' and not np.isnan(value):
                    one_hot_indices_user.append(fans_onehot[value] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                elif rowname == 'fans' and np.isnan(value):
                    one_hot_indices_user.append(0 + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                elif rowname == 'average_stars':
                    one_hot_indices_user.append(stars_onehot[value] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                else:
                    one_hot_indices_user.append(row[j] + current_offset) 
                    current_offset = current_offset + all_offsets[index]
                index += 1

        cfm_string = (",".join(str(x) for x in one_hot_indices_user)).replace(',', '-')
        cfm_string = cfm_string + ',' + str(self.test_items[i]) + '-' + str(self.n_items) 
        cfm_test.append(cfm_string)

    cfm_train_df = pd.DataFrame(cfm_train)
    cfm_test_df = pd.DataFrame(cfm_test)
    cfm_concat = pd.concat([cfm_test_df, cfm_train_df])
    cfm_concat.drop_duplicates(inplace=True)
    if path=='yelpnc':
        cfm_new_train = cfm_concat[2274:]
        cfm_new_test = cfm_concat[:2274]
    else:
        cfm_new_train = cfm_concat[5185:]
        cfm_new_test = cfm_concat[:5185]

    cfm_new_train.to_csv('train.csv', index=False)
    cfm_new_test.to_csv('test.csv', index=False)


#CFMConverter('Frappe', 'item')
#CFMConverter('yelpnc', 'business_id')
CFMConverter('yelpon', 'business_id')
#CFMConverter('ml1m', 'movieId')