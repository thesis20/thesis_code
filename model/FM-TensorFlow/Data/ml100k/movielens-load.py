import pandas as pd
from sklearn.model_selection import train_test_split


ratings = pd.read_csv('u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
user = pd.read_csv('u.user', sep='|', names=['userId', 'age', 'gender', 'occupation', 'zipcode'])

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
                        'thriller', 'war', 'western']

joined = joined.reindex(columns=column_names)

print(joined.head())

joined.to_csv('out.txt', index=False)
train, test = train_test_split(joined, test_size=0.2)

train.to_csv('train.txt', index=False)
test.to_csv('test.txt', index=False)