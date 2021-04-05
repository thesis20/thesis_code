import pandas as pd
from sklearn.model_selection import train_test_split
import datetime

ratings = pd.read_csv('ratings.dat', sep='::', names=['userId', 'movieId', 'rating', 'timestamp'])
user = pd.read_csv('users.dat', sep='::', names=['userId', 'gender', 'age', 'occupation', 'zipcode'])

def convert_datetime_to_timeofday(datetime):
    # convert the datetime to different timeofday intervals
    if datetime.hour <= 4:
        return 0
    elif 4 < datetime.hour <= 8:
        return 1
    elif 8 < datetime.hour <= 12:
        return 2
    elif 12 < datetime.hour <= 16:
        return 3
    elif 16 < datetime.hour <= 20:
        return 4
    elif 20 < datetime.hour <= 24:
        return 5



zipcode_ints = dict()
for value, key in enumerate(user['zipcode'].unique()):
    zipcode_ints[key] = value

def convert_zipcode_to_int(zipcode):
    # convert the datetime to different timeofday intervals
    return zipcode_ints[zipcode]

# convert zipcode to int
user['zipcode'] = user['zipcode'].apply(lambda x: convert_zipcode_to_int(x))

# convert timestamp to a datetime
ratings['datetime'] = ratings['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
# Convert datetimes to weekday integer
ratings['weekday'] = ratings['datetime'].apply(lambda x: x.weekday())
# convert datetimes to timeofday intervals 
ratings['timeofday'] = ratings['datetime'].apply(lambda x: convert_datetime_to_timeofday(x))

items = pd.read_csv('movies.dat', sep='::',
                    names=['movieId', 'title', 'genres'], encoding='latin-1')

genres = ['Action', 'Adventure', 'Animation', 'Children\'s',
          'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir',
          'Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
genre_dict = dict()
for index, row in items.iterrows():
    genre_list = row['genres'].split('|')
    for genre in genres:
        if genre in genre_list:
            if genre not in genre_dict.keys():
                genre_dict[genre] = [1]
            else:
                genre_dict[genre] += [1]
        else:
            if genre not in genre_dict.keys():
                genre_dict[genre] = [0]
            else:
                genre_dict[genre] += [0]

for key, value in genre_dict.items():
    items[key] = value

joined = ratings.merge(items).merge(user)
joined.drop('rating', inplace=True, axis=1)
joined.drop('title', inplace=True, axis=1)
joined.drop('genres', inplace=True, axis=1)
#joined.drop('release', inplace=True, axis=1)
#joined.drop('videorelease', inplace=True, axis=1)

column_names = ['userId', 'age', 'gender', 'occupation', 'zipcode', 'movieId',
                'Action', 'Adventure', 'Animation', 'Children\'s',
          'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir',
          'Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western', 'weekday', 'timeofday', 'timestamp']

joined = joined.reindex(columns=column_names)

print(joined.head())

joined.to_csv('out.txt', index=False)
train, test = train_test_split(joined, test_size=0.2)

train.to_csv('train.txt', index=False)
test.to_csv('test.txt', index=False)