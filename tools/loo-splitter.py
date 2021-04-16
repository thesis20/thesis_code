import pandas as pd

dataset = 'yelpon'
path = 'Data/' + dataset + '/out.txt'
out_path = 'Data/' + dataset

df = pd.read_csv(path)

user_newest = df.sort_values(['date']).drop_duplicates(['user_id'],keep='last')
remaining = pd.merge(df,user_newest, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

df.to_csv(out_path + '/loo_out.txt', index=False)
remaining.to_csv(out_path + '/loo_train.txt', index=False)
user_newest.to_csv(out_path + '/loo_test.txt', index=False)