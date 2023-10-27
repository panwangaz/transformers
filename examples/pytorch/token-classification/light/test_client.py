import requests
from tqdm import tqdm
from copy import deepcopy
from dataset import load_json, classfication_metric


URL = 'http://192.168.12.223:12765/api/v0/user_profile'

INPUT_DATA = dict(
    log_id = "202cdece-79d8-44c2-8d96-56687d1f002d_4ae1a058-62cd-4dfe-9613-7f45c07b1009_1695089704688",
    user_name = "751198232",
    user_id = "202cdece-79d8-44c2-8d96-56687d1f002d",
    star_id = "4ae1a058-62cd-4dfe-9613-7f45c07b1009",
    gendr = -1,
    history = [],
)

USER_MESSAGE = dict(
    from_user = True,
    msg = "",
)

NON_USER_MESSAGE = dict(
    from_user = False,
    msg = "",
)


"""a example for requset input data
{
    "log_id":"202cdece-79d8-44c2-8d96-56687d1f002d_4ae1a058-62cd-4dfe-9613-7f45c07b1009_1695089704688",
    "user_name":"751198232",
    "user_id":"202cdece-79d8-44c2-8d96-56687d1f002d",
    "star_id":"4ae1a058-62cd-4dfe-9613-7f45c07b1009",
    "gender":-1,
    "history":[
        {"from_user":True,"msg":"Do you see any future for us?"},
        {"from_user":True,"msg":"Okk"},
        {"from_user":False,"msg":"It does show that there is still a very strong connection between the two of you and I do see that he’s going to reach out soon. \
            Unfortunately I cannot provide a timeframe because that is not something that I ever predict however when he does reach out, \
            you’re going to notice that he’s just going to be distant hot and cold back in forward it’s going to feel like he’s going to \
            repeat this pattern over and over again but allow him to reach out when he is ready because he does need to be willing to come forward with his True feelings. He does"},
        {"from_user":True,"msg":"Hi — my name is Aimee and I just wanted to describe the situation I am currently in and I just want the reassurance and \
            understanding of what happened and what is currently happening and what may happen? \\nMe and this guy I recently befriended, a coworker of mine, \
            had gotten close in a matter of three weeks. He would snapchat me regularly as in multiple times of day, and consistently in the afternoon to night every day for as I mentioned, \
            three weeks straight. Him and I work on two different floors, and during the work week he would come to my floor to make sure he’d visit me. I had explained to him it is hard for me \
            in relationships and being vulnerable so I kinda avoid that area. However, he insisted he wanted to be that person for me, so we both opened up to one another. \
            I was very vulnerable with him as he was with me. We had established prior to anything that we were just friends, but the dynamic said other wise. \
            It was the conversations we had and the way we talked and how often. Also he would ask me questions like “if we ended up together” or “if in the future” like he did all the work and \
            asked everything, he set up the dynamic. He would flirt with me and call us hanging out a date. I asked him if he could see us be together in the future for real and he said yes. \
            Then two days later suddenly he said he only wanted to be friends and everything he meant was just as friends!! I was texting him a lot to understand what had happened but \
            he doesn’t visit me anymore and we don’t talk. Nothing makes sense now. What happened??!"},
        {"from_user":True,"msg":"Aimee Lenore - July 2nd 2003"},
        {"from_user":False,"msg":"✨"},
        {"from_user":False,"msg":" Tell me your name and date of birth"},
        {"from_user":False,"msg":"Can I have your name and date of birth please"},
        {"from_user":True,"msg":"When will he/she comeback to me?"}
    ],
}
"""

"""a return example from request
{
    'log_id': '202cdece-79d8-44c2-8d96-56687d1f002d_4ae1a058-62cd-4dfe-9613-7f45c07b1009_1695089704688', 
    'dob': {'year': None, 'month': None, 'day': None}, 
    'age': None, 
    'name': 'Aimee Lenore', 
    'age_confidence': 0.9990234375, 
    'name_confidence': 0.9990234375, 
    'other_dobs': {'0': 'July 2 nd 2003'}, 
    'other_names': {}, 
    'user_dobs': {'0': 'July 2 nd 2003'}, 
    'user_names': {}, 
    'debug_dict': {'bad_dob': None, 'bad_name': None}
}
"""

def compute_metrics(predictions, labels, label_list):
    true_labels_num = [label_list.index(l) for labels in labels for l in labels]
    true_pred_num = [label_list.index(p) for pred in predictions for p in pred]
    results = classfication_metric(predictions=true_pred_num, references=true_labels_num)
    log_res = {
        "precision": results["precision"].tolist(),
        "recall": results["recall"].tolist(),
        "f1": results["f1"].tolist(),
        "accuracy": results["accuracy"],
    }
    print(log_res)

def orgnaize_input(input):
    assert isinstance(input, dict), "not a dict input data"
    tokens = input.get("order", None)
    res, history = deepcopy(INPUT_DATA), []
    if tokens is not None:
        for token in tokens:
            if not token[0]:
                mes = deepcopy(NON_USER_MESSAGE)
            else:
                mes = deepcopy(USER_MESSAGE)
            mes["msg"] = token[1]
            history.append(mes)
        res["history"] = history
        return res
    return None

def reponse_from_web(req_data):
    rsp = requests.post(URL, json=req_data)
    if rsp.status_code ==200:
        rsp_data = rsp.json()
        return rsp_data
    return None

def use_gt_label(ori_data, all_labels):
    orders, date_index = ori_data["order"], ori_data["date_index"]
    name_index = ori_data["name_index"]
    date_labels, name_labels = ori_data["date_label"], ori_data["name_label"]
    name_count, date_count = 0, 0
    for order, date_label, name_label in zip(orders, date_labels, name_labels):
        if len(date_label) == 0 and len(name_label) == 0:
            all_labels.append([])
            continue
        words = order[1].split(" ")
        data = ["O" for _ in range(len(words))]
        # label name
        for nl in name_label:
            name = "USER_NAME" if (isinstance(name_index, (list, tuple)) and name_count in name_index) else "NON_USER_NAME"
            name_count += 1
            for n in nl:
                data[n] = name

        # label date
        for dl in date_label:
            date = "USER_DATE" if (isinstance(date_index, (list, tuple)) and date_count in date_index) else "NON_USER_DATE"
            date_count += 1
            for d in dl:
                data[d] = date
        all_labels.append(data)

def match(source, target):
    index = []
    for uname in target.values():
        target_words = uname.split(" ")
        for tw in target_words:
            if tw in source:
                index.append(source.index(tw))
    return index

def use_pred_label(ori_data, res_data, all_predictions):
    user_name, user_date = res_data["user_names"], res_data["user_dobs"]
    other_name, other_date = res_data["other_names"], res_data["other_dobs"]
    orders = ori_data["order"]
    if not (user_name or user_date or other_date or other_name):
        all_predictions.extend([[] for _ in range(len(orders))])
        return

    for order in orders:
        source_words = order[1].split(" ")
        data = ["O" for _ in range(len(source_words))]

        # label user
        un = match(source_words, user_name)
        for i in un:
            data[i] = "USER_NAME"
        ud = match(source_words, user_date)
        for i in ud:
            data[i] = "USER_DATE"
        # label other
        on = match(source_words, other_name)
        for i in on:
            data[i] = "NON_USER_NAME"
        od = match(source_words, other_date)
        for i in od:
            data[i] = "NON_USER_DATE"
        if data == ["O" for _ in range(len(source_words))]:
            all_predictions.append([])
        else:
            all_predictions.append(data)

def main():
    labels = ["O", "USER_NAME", "USER_DATE", "NON_USER_NAME", "NON_USER_DATE"]
    json_path = "data/NAME_DATE_NER/val.json"
    val_dataset = load_json(json_path)
    all_labels, all_predictions = [], []
    for i in tqdm(range(len(val_dataset))):
        org_data = orgnaize_input(val_dataset[i])
        if org_data is None:
            continue
        reponse = reponse_from_web(org_data)
        use_gt_label(val_dataset[i], all_labels)
        use_pred_label(val_dataset[i], reponse, all_predictions)

    # padding all_predictions or all_labels for compute metrics
    for i, pred in enumerate(all_predictions):
        label = all_labels[i]
        if len(pred) == len(label):
            continue
        elif len(pred) == 0 and len(label) != 0:
            all_predictions[i] = ["O" for _ in range(len(label))]
        elif len(pred) != 0 and len(label) == 0:
            all_labels[i] = ["O" for _ in range(len(pred))]
    compute_metrics(all_predictions, all_labels, labels)


if __name__ == "__main__":
    main()
