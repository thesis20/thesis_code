import pandas as pd
import numpy as np

dataset = 'yelpnc'


if dataset == 'ml100k':
    genrelist = ['unknown', 'action', 'adventure', 'animation',
                        'childrens', 'comedy', 'crime', 'documentary',
                        'drama', 'fantasy',  'film-noir', 'horror',
                        'musical', 'mystery', 'romance', 'scifi',
                        'thriller', 'war', 'western']
    item_sideinfo_columns = ['genre']
    user_sideinfo_columns = [
        'age', 'gender', 'occupation']
    context_list = ['weekday', 'timeofday']
    userid_column_name = 'userId'
    itemid_column_name = 'movieId'
    path = '../model/Graph-conv-context-aware-idea-2021-wip/Data/ml100k/'
elif dataset == 'ml1m':
    genrelist = ['Action', 'Adventure', 'Animation', 'Children\'s',
                        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    item_sideinfo_columns = ['genre']
    user_sideinfo_columns = [
        'age', 'gender', 'occupation', 'zipcode']
    context_list = ['weekday', 'timeofday']
    userid_column_name = 'userId'
    itemid_column_name = 'movieId'
    path = '../model/Graph-conv-context-aware-idea-2021-wip/Data/ml1m/'
elif dataset == 'frappe':
    genrelist = []
    context_list = ['weekday', 'timeofday', 'isweekend', 'weather']
    item_sideinfo_columns = ['cost']
    user_sideinfo_columns = ['city']
    userid_column_name = 'user'
    itemid_column_name = 'item'
    path = '../model/Graph-conv-context-aware-idea-2021-wip/Data/Frappe/'
elif dataset == 'yelpnc':
    genrelist = ['Shopping','LocalServices','Fashion','Sandwiches','Food','Bagels','Restaurants','Burgers','ChickenWings','Bars','Nightlife','SportsBars','American(Traditional)','Steakhouses','Tapas/SmallPlates','Breakfast&Brunch','American(New)','Pubs','CocktailBars','TapasBars','Gastropubs','Coffee&Tea','BeerBar','Breweries','Bakeries','Southern','SoulFood','Delis','SpecialtyFood','WineBars','EventPlanning&Services','Caterers','Cafes','Arts&Entertainment','MusicVenues','Pizza','Italian','IceCream&FrozenYogurt','FastFood','Chinese','SushiBars','Thai','AsianFusion','Japanese','Venues&EventSpaces','Mexican','Vietnamese','Hotels&Travel','Automotive','Diners','ComfortFood','Salad','LocalFlavor','Seafood','NailSalons','HairRemoval','Beauty&Spas','French','Beer','Wine&Spirits','Barbeque','Soup','Tex-Mex','DepartmentStores','Lounges','Vegetarian','ActiveLife','Desserts','LatinAmerican','Vegan','Gluten-Free','Greek','Home&Garden','Health&Medical','Mediterranean','JuiceBars&Smoothies','Grocery','Noodles','Indian','EthnicFood','Flowers&Gifts']
    item_sideinfo_columns = ['genre']
    user_sideinfo_columns = ['yelping_since', 'fans', 'average_stars']
    context_list = ['date']
    userid_column_name = 'user_id'  # done
    itemid_column_name = 'business_id'  # done
    path = '../model/Graph-conv-context-aware-idea-2021-wip/Data/yelpnc/'  # done
elif dataset == 'yelpon':
    genrelist = ['SpecialtyFood','Restaurants','EthnicFood','Chinese','Caterers','Food','EventPlanning&Services','Nightlife','Steakhouses','Bars','Seafood','SportsBars','Canadian(New)','American(Traditional)','Burgers','Italian','CocktailBars','Mediterranean','Hotels&Travel','Venues&EventSpaces','Gastropubs','Arts&Entertainment','Mexican','Barbeque','ComfortFood','Thai','AsianFusion','Pakistani','Buffets','American(New)','Beauty&Spas','Grocery','Coffee&Tea','FastFood','Pizza','Salad','SushiBars','Bakeries','French','Breakfast&Brunch','Korean','IceCream&FrozenYogurt','Noodles','Desserts','Cafes','Diners','Soup','Sandwiches','Ramen','Japanese','Pubs','Vietnamese','Shopping','MiddleEastern','Halal','Taiwanese','Vegan','TeaRooms','Vegetarian','DimSum','WineBars','JuiceBars&Smoothies','Fashion','Caribbean','Beer','Wine&Spirits','ChickenWings','Lounges','TapasBars','Tapas/SmallPlates','Indian','BubbleTea','ActiveLife','Greek','Gluten-Free']
    item_sideinfo_columns = ['genre']
    user_sideinfo_columns = ['yelping_since', 'average_stars']
    context_list = ['date']
    userid_column_name = 'user_id'  # done
    itemid_column_name = 'business_id'  # done
    path = '../model/Graph-conv-context-aware-idea-2021-wip/Data/yelpon/'  # done
else:
    print("No dataset defined")
    exit()

# Load DF
df = pd.read_csv(path + 'out.txt')
interactions = len(df.index)
users = df[userid_column_name].nunique()
items = df[itemid_column_name].nunique()

print("----------------------------------")
print(f"Interactions: {interactions}")
print(f"Users: {users}")
print(f"Items: {items}")

# - Sparsity
print(f"Sparsity: {(interactions / (users * items)):.5f}")

print("----------------------------------")


# - Average sideinfo count per item and user
item_genre_info_counter = dict()
for index, row in df.iterrows():
    if row[itemid_column_name] in item_genre_info_counter:
        continue
    row_counter = 0
    for side_info in genrelist:
        if row[side_info] == 1:
            row_counter += 1
    item_genre_info_counter[row[itemid_column_name]] = row_counter

print(f"Least genres: {min(item_genre_info_counter.values())}")
print(f"Most genres: {max(item_genre_info_counter.values())}")
print(f"Average genres: {(sum(item_genre_info_counter.values()) / len(item_genre_info_counter.values())):.2f}")
print("----------------------------------")

# - Items and users per side-info
print("User side-info:")
for side_info in user_sideinfo_columns:
    count = df[side_info].value_counts()
    print(count)
    print("----------------------------------")


print("Item side-info:")
for side_info in item_sideinfo_columns:
    if side_info == 'genre':
        continue
    count = df[side_info].value_counts()
    print(count)
    print("----------------------------------")

genre_count = []
if 'genre' in item_sideinfo_columns:
    for genre in genrelist:
        count = df[genre].value_counts()
        genre_count.append(count)
        print(genre + ": " + str(count[1]))

    print("----------------------------------")


# - # Context
print(f"Context #: {len(context_list)}")
print("----------------------------------")


# - Interactions per contextual variable
context_interactions = dict()
for index, row in df.iterrows():
    for context in context_list:
        if context + str(row[context]) not in context_interactions:
            context_interactions[context + str(row[context])] = 1
        else:
            context_interactions[context + str(row[context])] += 1

for key, value in context_interactions.items():
    print(str(key) + ": " + str(value))

print("----------------------------------")

print(f"Average interactions per context: {sum(context_interactions.values()) / len(context_interactions.values())}")

print("----------------------------------")


# - Average interactions per user
average_interactions = df[userid_column_name].value_counts().agg(['count', 'mean'])
print(f"Average interactions per user {average_interactions['mean']:.2f}")
print("----------------------------------")

# - Median af user interactions
average_interactions = df[userid_column_name].value_counts().agg(['count', 'median'])
print(f"Median interaction # {average_interactions['median']:.2f}")
print("----------------------------------")
