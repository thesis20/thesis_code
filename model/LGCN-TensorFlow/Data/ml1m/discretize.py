import pandas as pd
import matplotlib

# out_df = pd.read_csv('out.txt')
train_df = pd.read_csv('train.txt')
test_df = pd.read_csv('test.txt')

# 'userId', 'movieId'

if set(test_df['movieId'].unique()).issubset(set(train_df['movieId'].unique())):
    print("It is")
else:
    print("It not")

# user_id_remap = dict()
# for remap, orig in enumerate(out_df['userId'].unique()):
#     user_id_remap[orig] = remap

# movie_id_remap = dict()
# for remap, orig in enumerate(out_df['movieId'].unique()):
#     movie_id_remap[orig] = remap


# out_df['userId'] = out_df['userId'].apply(lambda x: user_id_remap[x])
# train_df['userId'] = train_df['userId'].apply(lambda x: user_id_remap[x])
# test_df['userId'] = test_df['userId'].apply(lambda x: user_id_remap[x])

# out_df['movieId'] = out_df['movieId'].apply(lambda x: movie_id_remap[x])
# train_df['movieId'] = train_df['movieId'].apply(lambda x: movie_id_remap[x])
# test_df['movieId'] = test_df['movieId'].apply(lambda x: movie_id_remap[x])

# def gender_fix(x):
#     if x == 'M':
#         return 0
#     else:
#         return 1

# out_df['gender'] = out_df['gender'].apply(lambda x: gender_fix(x))
# train_df['gender'] = train_df['gender'].apply(lambda x: gender_fix(x))
# test_df['gender'] = test_df['gender'].apply(lambda x: gender_fix(x))


# out_df.to_csv('out.txt', index=False)
# train_df.to_csv('train.txt', index=False)
# test_df.to_csv('test.txt', index=False)