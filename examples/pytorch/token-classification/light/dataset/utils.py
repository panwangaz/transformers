import json
from tqdm import tqdm
from copy import deepcopy
from mmengine import Registry
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


DATASETS = Registry("datasets")
MODELS = Registry('models')


def classfication_metric(predictions, references, labels=None, pos_label=1, average=None, 
                         sample_weight=None, zero_division="warn", normalize=True):
    f1_scores = f1_score(
        references, predictions, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight
    )
    precision_scores = precision_score(
        references,
        predictions,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    recall_scores = recall_score(
        references,
        predictions,
        labels=labels,
        pos_label=pos_label,
        average=average,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    accuracy_scores = accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)

    return {"f1": float(f1_scores) if f1_scores.size == 1 else f1_scores, 
            "precision": float(precision_scores) if precision_scores.size == 1 else precision_scores,
            "recall": float(recall_scores) if recall_scores.size == 1 else recall_scores,
            "accuracy": float(accuracy_scores),
            }

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    f.close()
    return data

def write_json(data, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False)
    f.close()
    print(f"finish dump data to {path}")

def find(target, source_data):
    for order in source_data:
        for index, words in enumerate(order["order"]):
            if target == words:
                return order["label"][index], order["user_index"]
    return None, None

def merge_labels(date_json_path, name_json_path, save_path):
    date_data = load_json(date_json_path)
    name_data = load_json(name_json_path)
    merge_data = []
    for order in tqdm(name_data, "process all orders"):
        new_order = deepcopy(order)
        new_order["name_label"] = new_order.pop('label')
        new_order["name_index"] = new_order.pop('user_index')
        new_order["date_label"] = [[] for _ in range(len(new_order["order"]))]
        new_order["date_index"] = None
        delete_indexs = []
        for index, sentences in enumerate(order["order"]):
            name, user_index = find(sentences, date_data)
            if name is not None:
                new_order["date_label"][index] = name
                new_order["date_index"] = user_index
            elif name is None and user_index is None:
                delete_indexs.append(index)
        delete_indexs.sort(reverse=True)
        for i in delete_indexs:
            new_order["order"].pop(i)
            new_order["name_label"].pop(i)
            new_order["date_label"].pop(i)
        merge_data.append(new_order)
    merge_data = remove_duplicate_lines(merge_data, is_merge=True)
    write_json(merge_data, save_path)

def remove_duplicate_lines(data, is_merge=False, is_name=False, is_date=False):
    for orders in tqdm(data, "remove duplicate lines"):
        order = orders["order"]
        if is_merge:
            name_label, date_label = orders["name_label"], orders["date_label"]
        if is_name:
            name_label = orders["label"]
        if is_date:
            date_label = orders["label"]

        pre_text, delete_indexs = None, []
        for index, text in enumerate(order):
            if pre_text is not None and pre_text == text:
                delete_indexs.append(index)
            pre_text = text
        delete_indexs.sort(reverse=True)
        for i in delete_indexs:
            order.pop(i)
            if is_merge:
                name_label.pop(i)
                date_label.pop(i)
            elif is_name:
                name_label.pop(i)
            elif is_date:
                date_label.pop(i)
    return data


if __name__ == "__main__":    
    date = ["data/AgeNER/train5000.json", "data/AgeNER/valid.json"]
    date_save_paths = ["data/DATE_NER/train.json", "data/DATE_NER/val.json"]
    name = ["data/NERDataset/tagged_train0000_01.json", "data/NERDataset/tagged_valid0000_01.json"]
    name_save_paths = ["data/NAME_NER/train.json", "data/NAME_NER/val.json"]
    save_paths = ["data/NAME_DATE_NER/train.json", "data/NAME_DATE_NER/val.json"]

    # for date_path, date_save_path in zip(date, date_save_paths):
    #     date_data = load_json(date_path)
    #     remove_duplicate_lines(date_data, is_name=True)
    #     write_json(date_data, date_save_path)

    # for path in date_save_paths:
    #     date_data = load_json(path)

    # for name_path, name_save_path in zip(name, name_save_paths):
    #     name_data = load_json(name_path)
    #     remove_duplicate_lines(name_data, is_date=True)
    #     write_json(name_data, name_save_path)

    # for path in name_save_paths:
    #     name_data = load_json(path)

    # for path in save_paths:
    #     data = load_json(path)

    for dp, np, sp in zip(date_save_paths, name_save_paths, save_paths):
        merge_labels(dp, np, sp)
        