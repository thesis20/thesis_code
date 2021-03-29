import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
state = 'WI'

file_root = state + '/'
data_file_path = file_root + state + '_out.txt'
df = pd.read_csv(data_file_path)

seed = 0

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


train, test = train_test_split(df, test_size=0.2, random_state=seed)

df.to_csv(file_root + 'out.txt', index=False)
train.to_csv(file_root + 'train.txt', index=False)
test.to_csv(file_root + 'test.txt', index=False)
