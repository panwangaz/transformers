import re
import unicodedata
from tqdm import tqdm
from copy import deepcopy
from dataset.utils import load_json, write_json

ORDER = dict(
    order=[],
    label=[],
    user_index=[],
)
"""targets
    [
        {
            "order": [[False, "sentences"], [], ...],
            "label": [[index in sentences], [], ...],
            "user_index", []
        },
        {
        
        },
    ]
"""

def remove_extra_words(sentence):
    # 使用正则表达式将特殊字符前后添加空格
    sentence = re.sub(r'([\n\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF])', r' \1 ', sentence)
    # 使用正则表达式将连续的多个空格替换为单个空格
    sentence = re.sub(r'\s+', ' ', sentence)
    # 使用 strip() 方法删除句子两边多余的空格
    sentence = sentence.strip()
    return sentence

def separate_emojis_and_text(input_string):
    separated_string = ""

    for char in input_string:
        if unicodedata.category(char) == "So":
            # 表情包特殊字符
            separated_string += " " + char + " "
        else:
            # 其他字符
            separated_string += char

    return separated_string

def extract_order(data):
    orders = []
    for order in data:
        if len(order) < 2:
            continue
        order[1] = separate_emojis_and_text(order[1])
        order[1] = remove_extra_words(order[1])
        orders.append(order[:2])
    return orders

def match(sources, targets):
    source_words = sources.split(" ")
    target_words = targets.split(" ")
    indexs = []
    for target_word in target_words:
        for i, source_word in enumerate(source_words):
            if target_word == source_word:
                indexs.append(i)
    return indexs

def check_indexs(target, sources):
    for source in sources:
        if target == source:
            return False
        else:
            for i in target:
                if i in source:
                    return False
    return True

import re

def check_for_digits(sentence):
    # 使用正则表达式匹配字符串中的数字
    pattern = r"\d"
    matches = re.findall(pattern, sentence)
    if matches:
        return True
    else:
        return False

def extract_label(orders, label):
    sample = lambda x : [i[0] if isinstance(i, list) else i for i in x]
    dob, name = label["dob"], label["name"]
    user_dob, other_dob = dob["user"], sample(dob["other"])
    user_name, other_name = name["user"], sample(name["other"])
    other_name.sort(key=lambda x : len(x), reverse=True)
    user_name.sort(key=lambda x : len(x), reverse=True)
    other_dob.sort(key=lambda x : len(x), reverse=True)
    user_dob.sort(key=lambda x : len(x), reverse=True)
    name_label, dob_label = [[] for _ in range(len(orders))], [[] for _ in range(len(orders))]
    name_user_index, dob_user_index = [], []
    name_counts, dob_counts = 0, 0
    for index, order in enumerate(orders):
        # match dob label
        for ud in user_dob:
            if check_for_digits(ud):
                ud_indexs = match(order[1], ud)
                if ud_indexs and check_indexs(ud_indexs, dob_label[index]):
                    dob_user_index.append(dob_counts)
                    dob_counts += 1
                    dob_label[index].append(ud_indexs)

        for od in other_dob:
            if check_for_digits(od):
                od_indexs = match(order[1], od)
                if od_indexs and check_indexs(od_indexs, dob_label[index]):
                    dob_counts += 1
                    dob_label[index].append(od_indexs)

        # match name label
        for un in user_name:
            un_indexs = match(order[1], un)
            if un_indexs and check_indexs(un_indexs, name_label[index]):
                name_user_index.append(name_counts)
                name_counts += 1
                name_label[index].append(un_indexs)

        for on in other_name:
            on_indexs = match(order[1], on)
            if on_indexs and check_indexs(on_indexs, name_label[index]):
                name_counts += 1
                name_label[index].append(on_indexs)

    return (name_label, dob_label, name_user_index, dob_user_index)

def show_order_and_label(data):
    from rich import print
    for i, orders in tqdm(enumerate(data)):
        words, names, dates  = orders["order"], orders["name_label"], orders["date_label"]
        for word, name, date in zip(words, names, dates):
            print(word)
            print(name)
            print(date)
        # words, names = orders["order"], orders["label"]
        # show order
        # for word, name in zip(words, names):
        #     print(word)
        #     print(name)

def format_json(data, name_path, date_path):
    name_data, date_data = deepcopy(data), deepcopy(data)
    for name_orders, date_orders in zip(name_data, date_data):
        name_orders["label"] = name_orders.pop("name_label")
        name_orders["user_index"] = name_orders.pop("name_index")
        name_orders.pop("date_label")
        name_orders.pop("date_index")

        date_orders["label"] = date_orders.pop("date_label")
        date_orders["user_index"] = date_orders.pop("date_index")
        date_orders.pop("name_label")
        date_orders.pop("name_index")

    write_json(name_data, name_path)
    write_json(date_data, date_path)

def merge(llm_tag, human_tag):
    sample = lambda x : [i[0] for i in x if i]
    llm_dob, llm_name = llm_tag["dob"], llm_tag["name"]
    human_dob, human_name = human_tag["dob"], human_tag["name"]
    dob_user = list(set(llm_dob["user"] + human_dob["user"]))
    dob_other = list(set(sample(llm_dob["other"]) + human_dob["other"]))
    name_user = list(set(llm_name["user"] + human_name["user"]))
    name_other = list(set(sample(llm_name["other"]) + human_name["other"]))

    return (name_user, name_other, dob_user, dob_other)

def merge_llm_human_labels(data):
    msc, llm_tag, human_tag = data["msc"], data["llm_tag"], data["human_tag"]
    order = extract_order(msc)
    merge_tags = merge(llm_tag, human_tag)
    label = dict(
        dob=dict(user=merge_tags[2], other=merge_tags[3]),
        name=dict(user=merge_tags[0], other=merge_tags[1]),
    )
    labels = extract_label(order, label)
    return order, labels

def reorg_train_val(name_path, date_path, save_path):
    date_paths = ["data/DATE_NER/train.json", "data/DATE_NER/val.json", date_path]
    name_paths = ["data/NAME_NER/train.json", "data/NAME_NER/val.json", name_path]
    all_date, all_name = [], []
    for dp in date_paths:
        all_date.extend(load_json(dp))
    for np in name_paths:
        all_name.extend(load_json(np))
    
    import random
    random.shuffle(all_date)
    random.shuffle(all_name)
    all_date_len, all_name_len = int(len(all_date) * 0.7), int(len(all_name) * 0.7)
    all_date_train, all_date_val = all_date[:all_date_len], all_date[all_date_len:]
    all_name_train, all_name_val = all_name[:all_name_len], all_name[all_name_len:]
    write_json(all_date_train, f"{save_path}/date_train.json")
    write_json(all_date_val, f"{save_path}/date_val.json")
    write_json(all_name_train, f"{save_path}/name_train.json")
    write_json(all_name_val, f"{save_path}/name_val.json")

def main(data_path, save_path):
    datas = load_json(data_path)
    new_data = []
    for data in tqdm(datas, desc="processing single order"):
        order = extract_order(data['msc'])
        cur_order = deepcopy(ORDER)
        user_info = data["user_info"]
        cur_order["order"], cur_order["name_label"], cur_order["date_label"] = order, user_info["name"], user_info["dob"]
        new_data.append(cur_order)

    write_json(new_data, save_path)

# def main(data_path, save_path):
#     datas = load_json(data_path)
#     new_data = []
#     for data in tqdm(datas, desc="processing single order"):
#         order, label = merge_llm_human_labels(data)
#         cur_order = deepcopy(ORDER)
#         cur_order["order"], cur_order["name_label"], cur_order["date_label"] = order, label[0], label[1]
#         cur_order["date_index"], cur_order["name_index"] = label[3], label[2]
#         new_data.append(cur_order)

#     write_json(new_data, save_path)

if __name__ == "__main__":
    data_path = "data/label_20231113/all_first_order.json"
    save_path = "data/label_20231113/train.json"
    name_path = "data/label_20231113/name_train.json"
    date_path = "data/label_20231113/date_train.json"
    all_save_path = "data/ALL_DATA_20231113"
    # data_path = "data/label_20231109/user_profile_tagged_data.2023-11-09.json"
    # save_path = "data/label_20231109/train.json"
    # name_path = "data/label_20231109/name_train.json"
    # date_path = "data/label_20231109/date_train.json"
    # all_save_path = "data/ALL_DATA_20231109"

    # step 1: load llm_human_data into data.json
    main(data_path, save_path)

    # step 2: split llm_human_data into name & date dataset
    # data = load_json(save_path)
    # show_order_and_label(data)
    # format_json(data, name_path, date_path)

    # # step 3: reorg new train & val dataset
    # reorg_train_val(name_path, date_path, all_save_path)
