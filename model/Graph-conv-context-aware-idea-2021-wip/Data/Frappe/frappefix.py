import pandas as pd
from sklearn.model_selection import train_test_split
import datetime

df = pd.read_csv('out.txt', names=['user','item','weekday','isweekend','cost','weather','country','city','timeofday'])
test = pd.read_csv('test.txt', names=['user','item','weekday','isweekend','cost','weather','country','city','timeofday'])
train = pd.read_csv('train.txt', names=['user','item','weekday','isweekend','cost','weather','country','city','timeofday'])

print(len(df['user'].unique()))

print(len(df['user'].unique()))
user_id_ints = dict()
for value, key in enumerate(df['user'].unique()):
    user_id_ints[key] = value


def convert_userid_to_int(userid):
    if isinstance(userid, str):
        userid = int(userid)
    return user_id_ints[userid]


business_id_ints = dict()
for value, key in enumerate(df['item'].unique()):
    business_id_ints[key] = value

def convert_businessid_to_int(businessid):
    if isinstance(businessid, str):
        userid = int(businessid)
    return business_id_ints[businessid]

test['user'] = test['user'].apply(lambda x: convert_userid_to_int(x))
test['item'] = test['item'].apply(lambda x: convert_businessid_to_int(x))
train['user'] = train['user'].apply(lambda x: convert_userid_to_int(x))
train['item'] = train['item'].apply(lambda x: convert_businessid_to_int(x))
df['user'] = df['user'].apply(lambda x: convert_userid_to_int(x))
df['item'] = df['item'].apply(lambda x: convert_businessid_to_int(x))

df.to_csv('out.txt', index=False)
train.to_csv('train.txt', index=False)
test.to_csv('test.txt', index=False)

