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
        if dataframe_path == 'FRAPPEloo_out.txt':
            self.context_list = ['weekday', 'timeofday', 'isweekend', 'weather']
        elif dataframe_path == 'yelpon':
            self.context_list = ['month','day_of_week','timeofday', 'hour']
        elif dataframe_path =='yelpnc':
            self.context_list = ['month','day_of_week','timeofday', 'hour']
        self.dataframe = pd.read_csv('Data/' + dataframe_path, sep=',', )
        self.loo_df = pd.read_csv('Data/' + loo_path, sep=',', )
        self.evaluator = evaluator()

       # Get popular items in case the context does not have enough items to fill a list of k size
        pop_series = self.dataframe[itemid_column_name].value_counts()
        pop_ids = pop_series.index.values.tolist()
        user_top_k_dict = {}

        for _, row in self.loo_df.iterrows():          
            context_df = self.loo_df
            # Get the current context of the interaction
            interaction_context = []
            for context in self.context_list:
                interaction_context.append(row[context]) 
            
            # Filter test set such that only interactions in that context appear
            for index in range(len(self.context_list)):
                context_df = context_df[(context_df[self.context_list[index]] == interaction_context[index])]

            # Make initial top k dict from the popular items in the context
            pop_context_series = context_df[itemid_column_name].value_counts()
            user_top_k_dict[row[userid_column_name]] = pop_context_series.index.values.tolist()
        
        
        user_ground_truth_dict = construct_user_ground_truth_dict(self.loo_df, userid_column_name, itemid_column_name)
        for k in ks:
            # Ensure top k dick has values that are k length
            # If the value is shorter than k, pad it with the most popular items not already in the list, regardless of context
            for key, value in user_top_k_dict.items():
                if len(value) < k:
                    for id in pop_ids:
                        if id not in value and len(user_top_k_dict[key]) < k:
                            user_top_k_dict[key].append(id)    
                else:
                    user_top_k_dict[key] = user_top_k_dict[key][:k]

            hr, ndcg, mrr = evaluator.evaluate_loo_no_sort(self.evaluator, user_top_k_dict, user_ground_truth_dict, k)

            print(k)
            print(f"HR: {hr}")
            print(f"NDCG: {ndcg}")
            print(f"MRR: {mrr}")

            
def construct_user_ground_truth_dict(dataframe, userid_column_name, itemid_column_name):
    user_ground_truth_dict = {}
    for _, row in dataframe.iterrows():
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
TopPopContext('FRAPPEloo_out.txt', 'FRAPPEloo_test.txt', 'user', 'item', [20, 50, 100])
