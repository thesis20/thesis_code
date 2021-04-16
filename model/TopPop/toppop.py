import pandas as pd
import sys
sys.path.insert(0,'../Graph-conv-context-aware-idea-2021-wip/')
import evaluation
from evaluation import evaluator


class TopPop():
    def __init__(self, dataframe_path, test_path, userid_column_name, itemid_column_name, ks):
        self.dataframe = pd.read_csv(dataframe_path + '.txt', sep=',', )
        self.test_df = pd.read_csv(test_path + '.txt', sep=',', )
        self.evaluator = evaluator()

        pop_series = self.dataframe[itemid_column_name].value_counts()
        for k in ks:
            k_pop_series = pop_series.head(k)
            k_pop_ids = k_pop_series.index.values.tolist()
            user_ground_truth_dict = construct_user_ground_truth_dict(self.test_df, userid_column_name, itemid_column_name)
            user_top_k_dict = {}
            for id in self.test_df[userid_column_name].unique():
                user_top_k_dict[id] = k_pop_ids
            
            precs, recs, f1s, ndcgs = [], [], [], []
            
            for id in user_top_k_dict.keys():
                prec, rec, f1, ndcg = evaluator.evaluate_one_user(self.evaluator, user_top_k_dict[id], user_ground_truth_dict[id])
                precs.append(prec)
                recs.append(rec)
                f1s.append(f1)
                ndcgs.append(ndcg)
            
            precision_value = sum(precs) / len(precs)
            recall_value = sum(recs) / len(recs)
            f1_value = sum(f1s) / len(f1s)
            ndcg_value = sum(ndcgs) / len(ndcgs)
            print(k)
            print(f"Precision: {precision_value}")
            print(f"Recall: {recall_value}")
            print(f"F1: {f1_value}")
            print(f"NDCG: {ndcg_value}")




class TopPopContext():
    def __init__(self, dataframe_path, loo_path, userid_column_name, itemid_column_name, ks):
        self.dataframe = pd.read_csv(dataframe_path + '.txt', sep=',', )
        self.loo_df = pd.read_csv(loo_path + '.txt', sep=',', )
        self.evaluator = evaluator()

        pop_series = self.dataframe[itemid_column_name].value_counts()
        for k in ks:
            k_pop_series = pop_series.head(k)
            k_pop_ids = k_pop_series.index.values.tolist()
            user_ground_truth_dict = construct_user_ground_truth_dict(self.test_df, userid_column_name, itemid_column_name)
            user_top_k_dict = {}
            for id in self.test_df[userid_column_name].unique():
                user_top_k_dict[id] = k_pop_ids

            hr, ndcg, mrr = evaluator.evaluate_loo_no_sort(self.evaluator, user_top_k_dict, user_ground_truth_dict, k)

            print(k)
            print(f"HR: {hr}")
            print(f"NDCG: {ndcg}")
            print(f"MRR: {mrr}")

def construct_user_ground_truth_dict(loo_df, userid_column_name, itemid_column_name):
    user_ground_truth_dict = {}
    for _, row in loo_df.iterrows():
            # Add interactions as ground truths
            if row[userid_column_name] not in user_ground_truth_dict:
                user_ground_truth_dict[row[userid_column_name]] = [
                    row[itemid_column_name]]
            else:
                if row[itemid_column_name] not in user_ground_truth_dict[row[userid_column_name]]:
                    user_ground_truth_dict[row[userid_column_name]].append(
                        row[itemid_column_name])
    
    return user_ground_truth_dict



#TopPop('ml100k', 'ml100ktest', 'userId', 'movieId', [20, 50, 100])
#TopPop('yelpnc', 'yelpnctest', 'user_id', 'business_id', [20, 50, 100])
#TopPop('yelpon', 'yelpontest', 'user_id', 'business_id', [20, 50, 100])
#TopPop('frappe', 'frappetest', 'userId', 'movieId', [20, 50, 100])