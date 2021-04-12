import pandas as pd
from sklearn.model_selection import train_test_split

ratings = pd.read_csv('frappe.csv', sep='\t', )
meta = pd.read_csv('meta.csv', sep='\t', names=['item', 'package', 'category', 'downloads', 'developer', 'icon', 'language', 'description', 'name', 'price', 'rating', 'short desc'])

def convert_daytime_to_timeofday(daytime):
    if daytime == 'sunrise':
        return 0
    elif daytime == 'morning':
        return 1
    elif daytime == 'noon':
        return 2
    elif daytime == 'afternoon':
        return 3
    elif daytime == 'evening':
        return 4
    elif daytime == 'sunset':
        return 5
    elif daytime == 'night':
        return 6
    else:
        print(f"Illegal: {daytime}")
        exit()

def convert_weekday_to_int(weekday):
    if weekday == 'monday':
        return 0
    elif weekday == 'tuesday':
        return 1
    elif weekday == 'wednesday':
        return 2
    elif weekday == 'thursday':
        return 3
    elif weekday == 'friday':
        return 4
    elif weekday == 'saturday':
        return 5
    elif weekday == 'sunday':
        return 6
    else:
        print(weekday)
        exit()

def convert_isweekend_to_bool(isweekend):
    if isweekend == 'weekend':
        return 1
    else:
        return 0

def convert_weather_to_int(weather):
    if weather == 'sunny':
        return 0
    if weather == 'cloudy':
        return 1
    if weather == 'unknown':
        return 2
    if weather == 'foggy':
        return 3
    if weather == 'stormy':
        return 4
    if weather == 'rainy':
        return 5
    if weather == 'drizzle':
        return 6
    if weather == 'snowy':
        return 7
    if weather == 'sleet':
        return 8

def convert_cost_to_int(cost):
    if cost == 'free':
        return 0
    else:
        return 1

def create_country_dict():
    countries = ratings['country'].unique()
    n_countries = ratings['country'].nunique()
    country_dict = {}
    i = 0

    for country in countries:
        country_dict[country] = i
        i += 1
    
    return country_dict

def convert_country_to_int(country, country_dict):
    return country_dict[country] 


print(ratings['cnt'].nunique())

ratings['timeofday'] = ratings['daytime'].apply(lambda x: convert_daytime_to_timeofday(x))
ratings['weekday'] = ratings['weekday'].apply(lambda x: convert_weekday_to_int(x))
ratings['isweekend'] = ratings['isweekend'].apply(lambda x: convert_isweekend_to_bool(x))
ratings['weather'] = ratings['weather'].apply(lambda x: convert_weather_to_int(x))
ratings['cost'] = ratings['cost'].apply(lambda x: convert_cost_to_int(x))
country_dict = create_country_dict()
ratings['country'] = ratings['country'].apply(lambda x: convert_country_to_int(x, country_dict))


joined = ratings.merge(meta)

joined.drop('cnt', inplace=True, axis=1)
joined.drop('daytime', inplace=True, axis=1)
joined.drop('homework', inplace=True, axis=1)
joined.drop('package', inplace=True, axis=1)
joined.drop('category', inplace=True, axis=1)
joined.drop('downloads', inplace=True, axis=1)
joined.drop('developer', inplace=True, axis=1)
joined.drop('icon', inplace=True, axis=1)
joined.drop('language', inplace=True, axis=1)
joined.drop('description', inplace=True, axis=1)
joined.drop('name', inplace=True, axis=1)
joined.drop('price', inplace=True, axis=1)
joined.drop('short desc', inplace=True, axis=1)
joined.drop('rating', inplace=True, axis=1)


orig_len = len(joined)
print(f"DF before filter: {orig_len}")
joined = joined.groupby('user').filter(lambda x: len(x) > 10)
joined = joined.groupby('item').filter(lambda x: len(x) > 10)

len_1 = len(joined)
len_2 = 0
while (len_1 != len_2):
    len_1 = len(joined)
    joined = joined.groupby('user').filter(lambda x: len(x) > 10)
    joined = joined.groupby('item').filter(lambda x: len(x) > 10)
    len_2 = len(joined)
    print(f"DF after filter: {len_2}")

print(f"DF after final filter: {len_2}")


saved = False
seed = 0

while not saved:
    print(f"Trying seed {seed}")
    train, test = train_test_split(joined, test_size=0.2, random_state=seed)

    train_users = set(train['user'].unique())
    test_users = set(test['user'].unique())

    train_items = set(train['item'].unique())
    test_items = set(test['item'].unique())

    if test_users.issubset(train_users) and test_items.issubset(train_items):
        joined.to_csv('out.txt', index=False)
        train.to_csv('file_root + ''train.txt', index=False)
        test.to_csv('test.txt', index=False)
        break
    else:
        seed += 1

#joined.to_csv('out.txt', index=False, sep=',')
#train, test = train_test_split(joined, test_size=0.2)

#train.to_csv('train.txt', index=False, sep=',')
#test.to_csv('test.txt', index=False, sep=',')
