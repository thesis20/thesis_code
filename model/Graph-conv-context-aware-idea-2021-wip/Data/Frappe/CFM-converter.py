import pandas as pd

train = pd.read_csv('train.txt', sep=',', )
test = pd.read_csv('test.txt', sep=',', )
ratings = pd.read_csv('out.txt', sep=',', )
train_items = train['item']
test_items = test['item']
all_items = ratings['item']
train.drop('item', inplace=True, axis=1)
test.drop('item', inplace=True, axis=1)
ratings.drop('item', inplace=True, axis=1)

n_items = all_items.nunique()
 
one_hot_length = 1297
user_offset = 957
daytime_offset = 7
weekday_offset = 7
isweekend_offset = 2
cost_offset = 2
weather_offset = 9
country_offset = 80
city_offset = 233
all_offsets = [user_offset,weekday_offset,isweekend_offset,cost_offset,weather_offset,country_offset,city_offset, daytime_offset]

def create_city_dict():
    cities = ratings['city'].unique()
    n_cities = ratings['city'].nunique()
    city_onehot_dict = {}
    i = 0

    for city in cities:
        city_onehot_dict[city] = i
        i += 1
    
    return city_onehot_dict

city_onehot = create_city_dict()  
cfm_train, cfm_test = [], []

for i, row in train.iterrows():
    current_offset = 0
    one_hot_indices_user =  []

    for index, value in enumerate(row):
        if index != 1:
            one_hot_indices_user.append(value + current_offset) 
            current_offset = current_offset + all_offsets[index]


    cfm_string = (",".join(str(x) for x in one_hot_indices_user)).replace(',', '-')
    cfm_string = cfm_string + ',' + str(train_items[i]) + '-' + str(n_items) 
    cfm_train.append(cfm_string)

for i, row in test.iterrows():
    current_offset = 0
    one_hot_indices_user =  []

    for index, value in enumerate(row):
        if index != 1:
            one_hot_indices_user.append(value + current_offset) 
            current_offset = current_offset + all_offsets[index]


    cfm_string = (",".join(str(x) for x in one_hot_indices_user)).replace(',', '-')
    cfm_string = cfm_string + ',' + str(train_items[i]) + '-' + str(n_items) 
    cfm_test.append(cfm_string)
