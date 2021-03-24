import pandas as pd
from sklearn.model_selection import train_test_split
import datetime

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

def convert_age_to_age_group(age):
    if age < 10:
        return 0
    elif age < 15:
        return 1
    elif age < 20:
        return 2
    elif age < 30:
        return 3
    elif age < 40:
        return 4
    elif age < 50:
        return 5
    elif age < 60:
        return 6
    else:
        return 7


ratings = pd.read_csv('u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
user = pd.read_csv('u.user', sep='|', names=['userId', 'age', 'gender', 'occupation', 'zipcode'])
user['age'] = user['age'].apply(lambda x: convert_age_to_age_group(x))

# convert timestamp to a datetime
ratings['datetime'] = ratings['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
# Convert datetimes to weekday integer
ratings['weekday'] = ratings['datetime'].apply(lambda x: x.weekday())
# convert datetimes to timeofday intervals 
ratings['timeofday'] = ratings['datetime'].apply(lambda x: convert_datetime_to_timeofday(x))

items = pd.read_csv('u.item', sep='|',
                    names=['movieId', 'title', 'release', 'videorelease', 'imdb',
                        'unknown', 'action', 'adventure', 'animation',
                        'childrens', 'comedy', 'crime', 'documentary',
                        'drama', 'fantasy',  'film-noir', 'horror',
                        'musical', 'mystery', 'romance', 'scifi',
                        'thriller', 'war', 'western'], encoding='latin-1')



joined = ratings.merge(items).merge(user)
joined.drop('rating', inplace=True, axis=1)
joined.drop('title', inplace=True, axis=1)
joined.drop('imdb', inplace=True, axis=1)
joined.drop('release', inplace=True, axis=1)
joined.drop('videorelease', inplace=True, axis=1)

column_names = ['userId', 'age', 'gender', 'occupation', 'zipcode', 'movieId', 'unknown', 'action', 'adventure', 'animation',
                        'childrens', 'comedy', 'crime', 'documentary',
                        'drama', 'fantasy',  'film-noir', 'horror',
                        'musical', 'mystery', 'romance', 'scifi',
                        'thriller', 'war', 'western', 'weekday', 'timeofday', 'timestamp']

joined = joined.reindex(columns=column_names)

print(joined.head())

joined.to_csv('out.txt', index=False)
train, test = train_test_split(joined, test_size=0.2)

train.to_csv('train.txt', index=False)
test.to_csv('test.txt', index=False)