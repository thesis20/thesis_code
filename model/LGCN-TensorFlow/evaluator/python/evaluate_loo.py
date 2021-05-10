"""
@author: Zhongchuan Sun
"""
import itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys
import heapq
def argmax_top_k(a, top_k=50):
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    return np.array([idx for ele, idx in ele_idx], dtype=np.intc)


def hit(rank, ground_truth):
    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if isinstance(ground_truth, list):
            if item in ground_truth:
                last_idx = idx
                break
        else:
            if item == ground_truth:
                last_idx = idx
                break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0
    return result


def ndcg(rank, ground_truth):
    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if isinstance(ground_truth, list):
            if item in ground_truth:
                last_idx = idx
                break
        else:
            if item == ground_truth:
                last_idx = idx
                break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0/np.log2(last_idx+2)
    return result


def mrr(rank, ground_truth):
    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if isinstance(ground_truth, list):
            if item in ground_truth:
                last_idx = idx
                break
        else:
            if item == ground_truth:
                last_idx = idx
                break
    result = np.zeros(len(rank), dtype=np.float32)
    result[last_idx:] = 1.0/(last_idx+1)
    return result


def evaluate_hr_mrr(scores, ground_truth_item):
    # hitrate
    hitrate = 1 if ground_truth_item in scores else 0

    # mrr
    mrr = 0
    for position, item in enumerate(scores, 1):
        if item in ground_truth_item:
            mrr = 1 / position
            
    return hitrate, mrr

def csgcn_hit(scores, ground_truth):
    #if len(ground_truth) != 1:
    return 1 if ground_truth in scores else 0
    #else:
     #   return 1 if ground_truth[0] in scores else 0

def csgcn_mrr(scores, ground_truth):
    mrr = 0
    for rank, item in enumerate(scores, 1):
        #if len(ground_truth) != 1:
        if item in ground_truth:
            mrr = 1 / rank
            break
        #else:
         #   if item in ground_truth[0]:
          #      mrr = 1 / rank
    
    return mrr


def eval_score_matrix_loo(score_matrix, test_items, test_set_dict, top_k=50, thread_num=None):
    def _eval_one_user(idx):
        scores = score_matrix[idx]  # all scores of the test user
        #test_item = test_items[idx]  # all test items of the test user
        test_item = test_set_dict[idx]
        ranking = argmax_top_k(scores, top_k)  # Top-K items
        result = []
        hrs = []
        mrrs = []
        result.append(csgcn_hit(ranking, test_item))
        #result.extend(ndcg(ranking, test_item))
        result.append(csgcn_mrr(ranking, test_item))

        result = np.array(result, dtype=np.float32).flatten()
        return result

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        batch_result = executor.map(_eval_one_user, range(len(test_items)))

    result = list(batch_result)  # generator to list
    return np.array(result)  # list to ndarray

def eval_score_matrix_loo2(score_matrix, test_items, test_set_dict, top_k=50, thread_num=None):
    batch_result = []
    mrrs = []
    hrs = []
    for i in range(len(test_items)):
        batch_result.append(_eval_one_user2(i, score_matrix, test_items, test_set_dict))
    result = list(batch_result)  # generator to list
    return np.array(result)  # list to ndarray

def _eval_one_user2(idx, score_matrix, test_items, test_set_dict):
    scores = score_matrix[idx]  # all scores of the test user
    test_item = test_items[idx]  # all test items of the test user
    #test_item = test_set_dict[idx]
    ranking = argmax_top_k(scores, 50)  # Top-K items
    result = []
    
    result.append(csgcn_hit(ranking, test_item))
    #result.extend(ndcg(ranking, test_item))
    result.append(csgcn_mrr(ranking, test_item))

    result = np.array(result, dtype=np.float32).flatten()
    return result

