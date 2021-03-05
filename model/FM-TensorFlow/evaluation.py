import numpy as np
import math

class evaluator():
    def __init__(self):
        print("Evaluating initialized")
    
    def ndcg(self, user_topk, ground_truth):
        dcg, idcg, gain = 0.0, 0.0, 1
        position = 0
        number_of_hits = 0
        
        # Calculate gain if the recommended item is in the ground truth list.
        for item in user_topk:
            position += 1
            if item in ground_truth:
                number_of_hits += 1
                dcg += gain / math.log2(position + 1)

        # Sort it such that all hits appear in the front of the list.    
        if number_of_hits > 0:
            for i in range(number_of_hits):
                idcg += 1 / math.log2((i+1) + 1)

        if idcg == 0:
            return 0
        else:
            return dcg / idcg

    def precision(self, user_topk, ground_truth):
        hits = [1 if item in ground_truth else 0 for item in user_topk]
        precision = np.sum(hits, dtype=np.float) / len(user_topk)
        return precision

    def recall(self, user_topk, ground_truth):
        hits = [1 if item in ground_truth else 0 for item in user_topk]
        recall = np.sum(hits, dtype=np.float32) / len(ground_truth)
        return recall

    def f1(self, precision, recall):
        try:
            return (2.0*float(precision)*float(recall)) / (float(precision) + float(recall))
        except:
            return 0.0
    
    def evaluate(self, score_dict, ground_truth_dict, topk, epoch):
        sorted_scores = {}
        # The input is a dictionary of scores with user id as key.
        # Argsort sorts it such that the position of the item in the list
        # is returned, which is equal to the corresponding movieid
        # We then take the top k of this sorted list of movieids for all users
        for key, value in score_dict.items():
            sorted_scores[key] = np.argsort(score_dict[key])[::-1]
            sorted_scores[key] = sorted_scores[key][:topk]
        
        precs, recs, f1s, ndcgs = [], [], [], []
        status = 0
        for key in sorted_scores.keys():
            prec, rec, f1, ndcg = self.evaluate_one_user(sorted_scores[key], ground_truth_dict[key])
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
            ndcgs.append(ndcg)
        
        precision_value = sum(precs) / len(precs)
        recall_value = sum(recs) / len(recs)
        f1_value = self.f1(precision_value, recall_value)
        ndcg_value = sum(ndcgs) / len(ndcgs)
        print(f"Precision: {precision_value}")
        print(f"Recall: {recall_value}")
        print(f"F1: {f1_value}")
        print(f"NDCG: {ndcg_value}")
        
        f = open("results.txt", "a")
        line = "Epoch: " + str(epoch) + " "
        line += "P: " + str(precision_value) + " "
        line += "R: " + str(recall_value) + " "
        line += "F1: " + str(f1_value) + " "
        line += "NDCG: " + str(ndcg_value) + "\n"
        f.write(line)
        f.close()

    def evaluate_one_user(self, user_topk, ground_truth_user):
        precision = self.precision(user_topk, ground_truth_user)
        recall = self.recall(user_topk, ground_truth_user)
        f1 = self.f1(precision, recall)
        ndcg = self.ndcg(user_topk, ground_truth_user)
        return precision, recall, f1, ndcg

