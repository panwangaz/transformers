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


def extract_order(data):
    orders = []
    for order in data:
        if len(order) < 2:
            continue
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

def extract_label(orders, label):
    sample = lambda x : [i[0] for i in x]
    dob, name = label["dob"], label["name"]
    user_dob, other_dob = dob["user"], sample(dob["other"])
    user_name, other_name = name["user"], sample(name["other"])
    name_label, dob_label = [[] for _ in range(len(orders))], [[] for _ in range(len(orders))]
    name_user_index, dob_user_index = [], []
    name_counts, dob_counts = 0, 0
    for index, order in enumerate(orders):
        # match dob label
        for ud in user_dob:
            ud_indexs = match(order[1], ud)
            if ud_indexs:
                dob_user_index.append(dob_counts)
                dob_counts += 1
                dob_label[index].append(ud_indexs)

        for od in other_dob:
            od_indexs = match(order[1], od)
            if od_indexs:
                dob_counts += 1
                dob_label[index].append(od_indexs)

        # match name label
        for un in user_name:
            un_indexs = match(order[1], un)
            if un_indexs:
                name_user_index.append(name_counts)
                name_counts += 1
                name_label[index].append(un_indexs)

        for on in other_name:
            on_indexs = match(order[1], on)
            if on_indexs:
                name_counts += 1
                name_label[index].append(on_indexs)

    return name_label, dob_label, name_user_index, dob_user_index

def show_order_and_label(data):
    from rich import print
    for i, orders in tqdm(enumerate(data)):
        # words, names, dates  = orders["order"], orders["name_label"], orders["date_label"]
        words, names = orders["order"], orders["label"]
        # show order
        for word, name in zip(words, names):
            print(word)
            print(name)

def format_json(data):
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

    write_json(name_data, "data/new_data/name_train.json")
    write_json(date_data, "data/new_data/date_train.json")

def main():
    label_path = "data/new_data/tagged_data.json"
    data_path = "data/new_data/first_order.20230823-1417.json"
    save_path = "data/new_data/train.json"

    labels = load_json(label_path)
    datas = load_json(data_path)

    new_data = []
    for key, data in tqdm(datas.items(), desc="processing single order"):
        if key not in labels.keys():
            continue

        order = extract_order(data)
        name_label, dob_label, name_user_index, dob_user_index = extract_label(order, labels[key]["tag_info"])
        cur_order = deepcopy(ORDER)
        cur_order["order"], cur_order["name_label"], cur_order["date_label"] = order, name_label, dob_label
        cur_order["date_index"], cur_order["name_index"] = dob_user_index, name_user_index
        new_data.append(cur_order)

    write_json(new_data, save_path)


if __name__ == "__main__":
    path = "data/NAME_NER/train.json"
    data = load_json(path)
    show_order_and_label(data)
    # format_json(data)
    main()