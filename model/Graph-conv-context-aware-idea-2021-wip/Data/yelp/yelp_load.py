import pandas as pd
from sklearn.model_selection import train_test_split

businesses = pd.read_json(r'yelp_academic_dataset_business.json', lines=True)
print("loaded business")
reviews = pd.read_json(r'yelp_academic_dataset_review.json', lines=True)
print("loaded reviews")
tips = pd.read_json(r'yelp_academic_dataset_tip.json', lines=True)
print("loaded tips")
users = pd.read_json(r'yelp_academic_dataset_user.json', lines=True)
print("loaded users")

businesses.drop(['name', 'address', 'postal_code', 'latitude', 'longitude', 'stars', 'is_open', 'attributes', 'hours'], inplace=True, axis=1)
print("drop business done")
tips.drop(['text', 'compliment_count'], inplace=True, axis=1)
print("drop tips done")
users.drop(['name', 'useful', 'funny', 'cool', 'elite', 'friends', 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos'], inplace=True, axis=1)
print("drop reviews done")
reviews.drop(['review_id', 'stars', 'useful', 'funny', 'cool', 'text'], inplace=True, axis=1)
print("drop tables done")


interactions = reviews.append(tips)
print("append done")
interactions = interactions.merge(users, on='user_id')
print("user merge done")
interactions = interactions.merge(businesses, on='business_id')
print("business merge done")

for state in businesses['state'].unique():
    print(f"Sctarting {state}")
    temp_interactions = interactions.loc[interactions['state'] == state]
    filename = state + '_out.txt'
    print(f"Saving {state}")
    temp_interactions.to_csv(filename, index=False)

