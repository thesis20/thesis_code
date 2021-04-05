import pandas as pd
from sklearn.model_selection import train_test_split

businesses = pd.read_json(r'yelp_academic_dataset_business.json', lines=True)
reviews = pd.read_json(r'yelp_academic_dataset_review.json', lines=True)
tips = pd.read_json(r'yelp_academic_dataset_tip.json', lines=True)
users = pd.read_json(r'yelp_academic_dataset_user.json', lines=True)

businesses.drop(['name', 'address', 'postal_code', 'latitude', 'longitude', 'stars', 'is_open', 'attributes', 'hours'], inplace=True, axis=1)
tips.drop(['text', 'compliment_count'], inplace=True, axis=1)
users.drop(['name', 'useful', 'funny', 'cool', 'elite', 'friends', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos'], inplace=True, axis=1)
reviews.drop(['review_id', 'stars', 'useful', 'funny', 'cool', 'text'], inplace=True, axis=1)


interactions = reviews.append(tips)
interactions = interactions.merge(users, on='user_id')
interactions = interactions.merge(businesses, on='business_id')

for state in businesses['state'].unique():
    temp_interactions = interactions.loc[interactions['state'] == state]
    filename = state + '_out.txt'
    temp_interactions.to_csv(filename, index=False)

