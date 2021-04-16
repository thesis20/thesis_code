import pandas as pd
import matplotlib

out_df = pd.read_csv('out.txt')
train_df = pd.read_csv('train.txt')
test_df = pd.read_csv('test.txt')

# 'fans', 'average_stars'

def fix_fans(x):
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x < 5:
        return 5
    if x < 100:
        return 100
    if x > 200:
        return 200
    if x > 300:
        return 300
    if x > 400:
        return 400
    if x > 500:
        return 500
    if x > 600:
        return 600
    if x > 400:
        return 700

def fix_average_stars(x):
    return round(x * 2) / 2


out_df['fans'] = out_df['fans'].apply(lambda x: fix_fans(x))
train_df['fans'] = train_df['fans'].apply(lambda x: fix_fans(x))
test_df['fans'] = test_df['fans'].apply(lambda x: fix_fans(x))

out_df['average_stars'] = out_df['average_stars'].apply(lambda x: fix_average_stars(x))
train_df['average_stars'] = train_df['average_stars'].apply(lambda x: fix_average_stars(x))
test_df['average_stars'] = test_df['average_stars'].apply(lambda x: fix_average_stars(x))


out_df.to_csv('out.txt', index=False)
train_df.to_csv('train.txt', index=False)
test_df.to_csv('test.txt', index=False)