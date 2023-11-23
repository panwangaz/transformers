import numpy as np
import argparse
import json

from scipy.optimize import curve_fit


def func(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d


# def get_recall_2_precision_f1(points_data, args_recall):
#     recalls = points_data["recalls"]
#     precisions = points_data["precisions"]
#     idxs = [i for i in range(len(recalls))]
#     recall_pairs = list(zip(recalls, idxs))
#     sorted_recall_pairs = sorted(recall_pairs, key=lambda x: x[0])
#     sorted_recalls = [pair[0] for pair in sorted_recall_pairs]
#     sorted_precisions = [precisions[pair[1]] for pair in sorted_recall_pairs]
# 
#     sp_next_idx = None
#     for idx, (recall, precision) in enumerate(zip(sorted_recalls, sorted_precisions)):
#         if recall > args_recall:
#             sp_next_idx = idx
#             break
#     if sp_next_idx is not None:
#         head_idx = sp_next_idx - 4 - 1
#         tail_idx = sp_next_idx + 4
#     else:
#         head_idx = len(sorted_recalls) - 5
#         tail_idx = len(sorted_recalls)
# 
#     sorted_recalls = sorted_recalls[head_idx:tail_idx]
#     sorted_precisions = sorted_precisions[head_idx:tail_idx]
# 
#     x_data = np.array(sorted_recalls)
#     y_data = np.array(sorted_precisions)
#     params, covariance = curve_fit(func, x_data, y_data)
#     a_fit, b_fit, c_fit, d_fit  = params
#     precision = func(args_recall, a_fit, b_fit, c_fit, d_fit)
#     f1 = 2 * precision * args_recall / (precision + recall)
#     print("the result of the recall: {}, precision: {}, F1: {}".format(recall, precision, f1))

def get_recall_2_precision_f1(points_data):
    recalls = points_data["recalls"]
    precisions = points_data["precisions"]
    idxs = [i for i in range(len(recalls))]
    recall_pairs = list(zip(recalls, idxs))
    sorted_recall_pairs = sorted(recall_pairs, key=lambda x: x[0])
    sorted_recalls = [pair[0] for pair in sorted_recall_pairs]
    sorted_precisions = [precisions[pair[1]] for pair in sorted_recall_pairs]
    
    recall = np.array(sorted_recalls)
    precision = np.array(sorted_precisions)
    f1 = np.nan_to_num(2*precision*recall/(precision+recall))
    idx = f1.argmax()
    recall = recall[idx]
    precision = precision[idx]
    f1 = f1[idx]
    print("the result of the recall: {}, precision: {}, F1: {}".format(recall, precision, f1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--points_path", help="", type=str, default="../tmp/best_pr_auc_metrics/2023_08_30-22_21_52.json")
    args = parser.parse_args()

    with open(args.points_path, "r") as jf:
        points_data = json.load(jf)

    get_recall_2_precision_f1(points_data)

if __name__ == "__main__":
    main()

