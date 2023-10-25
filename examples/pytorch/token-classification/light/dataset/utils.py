import json
from tqdm import tqdm
from copy import deepcopy
from mmengine import Registry
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


DATASETS = Registry("datasets")


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
                return order["label"][index]
    return None

def merge_labels(date_json_path, name_json_path, save_path):
    date_data = load_json(date_json_path)
    name_data = load_json(name_json_path)
    merge_data = deepcopy(date_data)
    for order in tqdm(merge_data, "process all orders"):
        order["date_label"] = order.pop('label')
        order["name_label"] = [[] for _ in range(len(order["order"]))]
        for index, sentences in enumerate(order["order"]):
            name = find(sentences, name_data)
            if name != [] and name is not None:
                order["name_label"][index] = name

    write_json(merge_data, save_path)

if __name__ == "__main__":    
    date = ["data/AgeNER/train5000.json", "data/AgeNER/valid.json"]
    name = ["data/NERDataset/tagged_train0000_01.json", "data/NERDataset/tagged_train0000_01.json"]
    save_paths = ["data/NAME_DATE_NER/train.json", "data/NAME_DATE_NER/val.json"]

    for dp, np, sp in zip(date, name, save_paths):
        merge_labels(dp, np, sp)
        