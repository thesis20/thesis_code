import pandas as pd
from sklearn.model_selection import train_test_split

ratings = pd.read_csv('frappe.csv', sep='\t', names=['userId', 'itemId', 'count', 'daytime', 'weekday', 'isweekend', 'homework', 'cost', 'weather', 'country', 'city'])
meta = pd.read_csv('meta.csv', sep='\t', names=['itemId', 'package', 'category', 'downloads', 'developer', 'icon', 'language', 'description', 'name', 'price', 'rating', 'short desc'])

def convert_daytime_to_timeofday(daytime):
    if daytime == 'sunrise':
        return 0
    if daytime == 'morning':
        return 1
    if daytime == 'noon':
        return 2
    if daytime == 'afternoon':
        return 3
    if daytime == 'evening':
        return 4
    if daytime == 'sunset':
        return 5
    if daytime == 'night':
        return 6

def convert_weekday_to_int(weekday):
    if weekday == 'monday':
        return 0
    if weekday == 'tuesday':
        return 1
    if weekday == 'wednesday':
        return 2
    if weekday == 'thursday':
        return 3
    if weekday == 'friday':
        return 4
    if weekday == 'saturday':
        return 5
    if weekday == 'sunday':
        return 6

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

ratings['daytime'] = ratings['daytime'].apply(lambda x: convert_daytime_to_timeofday(x))
ratings['weekday'] = ratings['weekday'].apply(lambda x: convert_weekday_to_int(x))
ratings['isweekend'] = ratings['isweekend'].apply(lambda x: convert_isweekend_to_bool(x))
ratings['weather'] = ratings['weather'].apply(lambda x: convert_weather_to_int(x))
ratings['cost'] = ratings['cost'].apply(lambda x: convert_cost_to_int(x))

joined = ratings.merge(meta)

joined.drop('count', inplace=True, axis=1)
joined.drop('homework', inplace=True, axis=1)
joined.drop('country', inplace=True, axis=1)
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

joined.to_csv('out.txt', index=False)
train, test = train_test_split(joined, test_size=0.2)

train.to_csv('train.txt', index=False)
test.to_csv('test.txt', index=False)
