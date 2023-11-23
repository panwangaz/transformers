import os
import json
import re
import random
from scipy.special import softmax
from sklearn.metrics import precision_recall_curve, accuracy_score, auc, precision_recall_fscore_support


def my_metrics(pred_labels, true_labels, probs):
    metric_dict = {}
    accuracy = []
    pr_auc = []
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None)

    for class_label in range(3):
        # 创建二分类标签数组，将当前类别置为正类（1），其他类别置为负类（0）
        y_true_binary = [1 if y == class_label else 0 for y in true_labels]
        y_pred_binary = [1 if y == class_label else 0 for y in pred_labels]
        
        # 计算当前类别的预测准确度
        accuracy.append(accuracy_score(y_true_binary, y_pred_binary))

    pred_probs = softmax(probs, axis=1)
    for class_label in range(3):
        # 将二进制标签创建为每个类别的one-hot编码
        y_true = (true_labels == class_label).astype(int)
    
        # 获取预测属于当前类别的概率
        y_score = pred_probs[:, class_label]
        
        # 计算PR曲线中的精确度和召回率
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_score)
        
        # 计算PR AUC
        pr_auc.append(auc(pr_recall, pr_precision))
    
    metric_dict['precision_negative'] = precision[0]
    metric_dict['recall_negative'] = recall[0]
    metric_dict['f1_score_negative'] = f1_score[0]
    metric_dict['precision_neutral'] = precision[1]
    metric_dict['recall_neutral'] = recall[1]
    metric_dict['f1_score_neutral'] = f1_score[1]
    metric_dict['precision_positive'] = precision[2]
    metric_dict['recall_positive'] = recall[2]
    metric_dict['f1_score_positive'] = f1_score[2]
    metric_dict['accuracy_negative'] = accuracy[0]
    metric_dict['accuracy_neutral'] = accuracy[1]
    metric_dict['accuracy_positive'] = accuracy[2]
    metric_dict['pr_auc_negative'] = pr_auc[0]
    metric_dict['pr_auc_neutral'] = pr_auc[1]
    metric_dict['pr_auc_positive'] = pr_auc[2]
    
    return metric_dict

def format_feature(feature):
    if isinstance(feature, float):
        feature = round(feature, 3)
        return feature

def load_prelabel_output_dir(path):
    files = [os.path.join(path, i) for i in os.listdir(path) if '.json' in i]
    with open(files[0], "r") as jf:
        data = json.load(jf)
    for file in files[1:]:
        with open(file, "r") as jf:
            data_new = json.load(jf)
            data.update(data_new)
    return data

def clean_string(sentence):
    cleaned_string = re.sub(r'\d+\.\s*', '', sentence)
    cleaned_string = cleaned_string.replace("\n", "")
    return cleaned_string

def truncate_dialogue(string, max_len):
    # 截取最后512个单词
    words = string.split()
    if len(words) < max_len:
        return string
    words_to_keep = words[-max_len:]

    # 将截取的单词重新组合成字符串
    truncated_string = " ".join(words_to_keep)

    # 确保对话以[user]或[advisor]开头
    index_user = truncated_string.find("[user]")
    index_advisor = truncated_string.find("[advisor]")

    # 如果找到了[user]或[advisor]开头的索引
    if (index_user != -1) or (index_advisor != -1):
        truncated_string = truncated_string[min(index_user, index_advisor):]

    # 确保截取后的对话以[user]或[advisor]开头
    if truncated_string.startswith("[user]") or truncated_string.startswith("[advisor]"):
        return truncated_string

    # 如果不是以[user]或[advisor]开头，则继续向后截取
    for i in range(len(truncated_string)):
        if truncated_string[i:].startswith("[user]") or truncated_string[i:].startswith("[advisor]"):
            return truncated_string[i:]

    # 如果未找到[user]或[advisor]开头，则返回空字符串
    return ""

def get_data(path,regression=False,max_len=512):
    data = load_prelabel_output_dir(path)
    sentences = []
    scores = []
    order_ids = []
    sample_nums = []
    for order_id in data:
        for sample_num in data[order_id]:
            history = clean_string(data[order_id][sample_num]['history'])
            current = clean_string(data[order_id][sample_num]['current'])
            score = data[order_id][sample_num]['score']
            
            if score > 0.3:
                score = 2
            elif score < -0.3:
                score = 0
            else:
                score = 1
            
            sentence = "{}[SEP] {}".format(history,current)
            sentence = truncate_dialogue(sentence,max_len)
            sentences.append(sentence)
            if regression == True:
                scores.append(score)
            else:
                # scores.append(int(score>0))
                scores.append(score)
            order_ids.append(order_id)
            sample_nums.append(sample_num)
    return sentences,scores,order_ids,sample_nums

def collate_data_deprecated(path, balance=False, regression=False, max_len=512):

    data = {
        "sentence": [],
        "label": [],
        "idx": [],
        "group_id": [] # 0 for raw data 1 for copied data
    }

    # balance means balancing the possitive and negative data
    
    raw_data,labels,order_ids,sample_nums = get_data(path,regression,max_len)

    data_idx = [i for i in range(len(raw_data))]

    label_to_data_idxs = {}

    for idx in data_idx:
        label = labels[idx]
        data["sentence"].append(raw_data[idx])
        data["label"].append(label)
        data["idx"].append(idx)
        data["group_id"].append(0)
        if label not in label_to_data_idxs:
            label_to_data_idxs[label] = []
        label_to_data_idxs[label].append(idx)

    for label, idx_list in label_to_data_idxs.items():
        print("ratio of {} label is: {}".format(label, len(idx_list) / len(data_idx)))

    # if balance:
    #     sample_len = len(label_to_data_idxs[0]) - len(label_to_data_idxs[1])
    #     add_idxs = []
    #     now_idx = 0
    #     while(len(add_idxs) <= sample_len):
    #         if now_idx >= len(label_to_data_idxs[1]):
    #             now_idx = 0
    #         add_idxs.append(label_to_data_idxs[1][now_idx])
    #         now_idx += 1
            
    #     for idx in add_idxs:
    #         data["sentence"].append(data["sentence"][idx])
    #         data["label"].append(data["label"][idx])
    #         data["idx"].append(len(data["idx"]))
    #         data["group_id"].append(1)

    if balance:
        sample_neg_len = len(label_to_data_idxs[1]) - len(label_to_data_idxs[0])
        sample_pos_len = len(label_to_data_idxs[1]) - len(label_to_data_idxs[2])
        add_neg_idxs = []
        add_pos_idxs = []
        now_neg_idx = 0
        now_pos_idx = 0
        
        while(len(add_neg_idxs) <= sample_neg_len):
            if now_neg_idx >= len(label_to_data_idxs[0]):
                now_neg_idx = 0
            add_neg_idxs.append(label_to_data_idxs[0][now_neg_idx])
            now_neg_idx += 1

        while(len(add_pos_idxs) <= sample_pos_len):
            if now_pos_idx >= len(label_to_data_idxs[2]):
                now_pos_idx = 0
            add_pos_idxs.append(label_to_data_idxs[2][now_pos_idx])
            now_pos_idx += 1
            
        for idx in add_neg_idxs:
            data["sentence"].append(data["sentence"][idx])
            data["label"].append(data["label"][idx])
            data["idx"].append(len(data["idx"]))
            data["group_id"].append(1)
        
        for idx in add_pos_idxs:
            data["sentence"].append(data["sentence"][idx])
            data["label"].append(data["label"][idx])
            data["idx"].append(len(data["idx"]))
            data["group_id"].append(1)
    
    return data,order_ids,sample_nums


def test():
    accuracy = []
    data,order_ids,sample_nums = collate_data_deprecated(path = '/data/yangxiaoran/nonrt_init/pay_rate_level_train/data/sentiment_train_new/', balance=False)
    print('len_data: ',len(data['sentence']))
    print('len_order_ids:',len(order_ids))
    print('len_sample_nums:',len(sample_nums))

    import pdb;pdb.set_trace()
    data_len = len(data['label'])
    data_random = [random.randint(0, 2) for _ in range(data_len)]
    # 计算精确度
    precision, recall, f1_score, _ = precision_recall_fscore_support(data['label'], data_random, average=None)

    for class_label in range(3):
        # 创建二分类标签数组，将当前类别置为正类（1），其他类别置为负类（0）
        y_true_binary = [1 if y == class_label else 0 for y in data['label']]
        y_pred_binary = [1 if y == class_label else 0 for y in data_random]
        
        # 计算当前类别的预测准确度
        accuracy.append(accuracy_score(y_true_binary, y_pred_binary))

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print('Accuracy:',accuracy)


if __name__ == "__main__":
    test()
