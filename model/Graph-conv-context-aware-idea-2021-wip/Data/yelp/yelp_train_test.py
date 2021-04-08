import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

state = 'NV'

file_root = state + '/'
data_file_path = file_root + state + '_out.txt'
df = pd.read_csv(data_file_path)

orig_len = len(df)
print(f"DF before filter: {orig_len}")
df = df.groupby('user_id').filter(lambda x: len(x) > 20)
df = df.groupby('business_id').filter(lambda x: len(x) > 20)

len_1 = len(df)
len_2 = 0
while (len_1 != len_2):
    len_1 = len(df)
    df = df.groupby('user_id').filter(lambda x: len(x) > 20)
    df = df.groupby('business_id').filter(lambda x: len(x) > 20)
    len_2 = len(df)
    print(f"DF after filter: {len_2}")

print(f"DF after final filter: {len_2}")

users = df['user_id'].nunique()
items = df['business_id'].nunique()
interactions = len(df.index)

sparsity = interactions / (users * items)

print(f"Sparsity: {sparsity}")

categories_dict = dict()

for categories in df['categories'].unique():
    categories = str(categories)
    categories = categories.strip("\"") # Without quotes
    categories = categories.replace(" ", "") # Without spaces
    categories = categories.split(",")
    for category in categories:
        if category not in categories_dict:
            categories_dict[category] = []


for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    categories = str(row['categories'])
    categories = categories.strip("\"") # Without quotes
    categories = categories.replace(" ", "") # Without spaces
    categories = categories.split(",") # Convert to list
    for category in categories_dict.keys():
        if category in categories:
            categories_dict[category].append(1)
        else:
            categories_dict[category].append(0)

for key, value in tqdm(categories_dict.items()):
    df[key] = value

df.drop('categories', inplace=True, axis=1)


df = df.loc[:, (df==0).mean() < .99]

def convert_date_to_month(date_time_str):
    # 2019-01-06 20:04:51
    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    month = date_time_obj.month
    return month

def convert_date_to_year(date_time_str):
    # 2019-01-06 20:04:51
    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    year = date_time_obj.year
    return year

df['date'] = df['date'].apply(lambda x: convert_date_to_month(x))
df['yelping_since'] = df['yelping_since'].apply(lambda x: convert_date_to_year(x))

user_id_ints = dict()
for value, key in enumerate(df['user_id'].unique()):
    user_id_ints[key] = value

def convert_userid_to_int(userid):
    return user_id_ints[userid]


business_id_ints = dict()
for value, key in enumerate(df['business_id'].unique()):
    business_id_ints[key] = value

def convert_businessid_to_int(businessid):
    return business_id_ints[businessid]


df['user_id'] = df['user_id'].apply(lambda x: convert_userid_to_int(x))
df['business_id'] = df['business_id'].apply(lambda x: convert_businessid_to_int(x))


saved = False
prev = None
seed = 0

while not saved:
    print(f"Trying seed {seed}")
    train, test = train_test_split(df, test_size=0.2, random_state=seed)


    train_users = set(train['user_id'].unique())
    test_users = set(test['user_id'].unique())

    train_items = set(train['business_id'].unique())
    test_items = set(test['business_id'].unique())

    if test_users.issubset(train_users) and test_items.issubset(train_items):
        print(f"Found good seed: {seed}")
        df.to_csv(file_root + 'out.txt', index=False)
        train.to_csv(file_root + 'train.txt', index=False)
        test.to_csv(file_root + 'test.txt', index=False)
        break
    else:
        seed += 1

