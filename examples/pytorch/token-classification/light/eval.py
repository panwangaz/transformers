import json
import torch
import datasets
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from dataset.name_dataset import NAMEDataset
from model.modeling_multihead_tokencls import MultiheadBertForTokenClassification, MultiheadDistilBertForTokenClassification

tokenize_and_align_labels, compute_metrics = NAMEDataset.tokenize_and_align_labels, NAMEDataset.compute_metrics
logger = datasets.logging.get_logger(__name__)

LABEL2IDS = [
    {"O":0, "USER_NAME":1, "NON_USER_NAME":2},
    {"O":0, "USER_DATE":1, "NON_USER_DATE":2}
]

def _get_labels(data, with_prefix_token=None):
    for item in data:
        order = item["order"]
        name_label_list, name_index = item["label"], item["user_index"]
        item["flat_order"], item["flat_label"] = [], []
        for sentence, name_labels in zip(order, name_label_list):
            if with_prefix_token is not None:
                text = f"{with_prefix_token} [USER] " if sentence[0] else f"{with_prefix_token} [ADVISOR] "
            else:
                text = "[USER] " if sentence[0] else "[ADVISOR] "
            text += sentence[1].strip()
            text = text.split()
            item["flat_order"].append(text)
            label_flattened = [ "O" for _ in range(len(text))]
            name_count = 0
            # process name labels
            for label in name_labels:
                name_value = "USER_NAME" if isinstance(name_index, (list, tuple, )) and name_count in name_index else "NON_USER_NAME"
                name_count += 1
                for label_index in range(len(label)):
                    tag_index = label[label_index] + 1 if with_prefix_token is None else label[label_index] + 2
                    if tag_index < len(label_flattened):
                        label_flattened[tag_index] = name_value
            item["flat_label"].append(label_flattened)
    return data

def _get_datas(tokenizer, path, window_size):
    """
    The tokenizer here is for the out of max seq text cutting
    """
    def _merge_lines(lines):
        result = []
        for line in lines:
            result.extend(line)
        return result
    # init data:
    with open(path, "r") as fin:
        data = json.loads(fin.read())
        fin.close()
    data = _get_labels(data)
    input_dict = {"tokens": [] , "labels": []}

    for order in tqdm(data):
        for line_index in range(len(order["order"])):
            if order["order"][line_index][0]:
                temp_tokens, temp_labels = [], []
                # pre-context input
                for i in range(window_size + 1):
                    now_index = line_index - i
                    if now_index >= 0:
                        temp_tokens.insert(0, order["flat_order"][now_index])
                        temp_labels.insert(0, order["flat_label"][now_index])
                    # check max seq
                    now_sentence = " ".join(_merge_lines(temp_tokens))
                    tokenized_now_sentence = tokenizer(now_sentence)
                    input_ids = tokenized_now_sentence["input_ids"]
                    if len(input_ids) > 514:
                        del(temp_tokens[0])
                        del(temp_labels[0])
                        break
                if len(temp_tokens) > 0:
                    input_dict["tokens"].append(_merge_lines(temp_tokens))
                    input_dict["labels"].append(_merge_lines(temp_labels))

    return Dataset.from_pandas(pd.DataFrame({'tokens': input_dict["tokens"], 
                                            'ner_tags': input_dict["labels"]}))

def build_dataset(data_path, 
                  tokenizer_name_or_path, 
                  context_window_size=6,
                  text_column_name="tokens", 
                  label_column_name="ner_tags",
                  use_padding_for_context = False,
                  max_seq_length = 150,
                  cache_dir="test-ner",
                  use_fast=True,
                  revision="main",
                  use_auth_token=False,
                ):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=cache_dir,
        use_fast=use_fast,
        revision=revision,
        use_auth_token=use_auth_token,
        do_basic_tokenize=False,
    )
    # add special tokens
    special_tokens_dict = {'additional_special_tokens': ['[USER]','[ADVISOR]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"add {num_added_toks} new special tokens: {special_tokens_dict.values()}")
    datas = _get_datas(tokenizer, data_path, context_window_size)

    all_test_dataset = dict()
    for task_id in range(2):
        test_dataset = datas.map(lambda x : tokenize_and_align_labels(tokenizer, text_column_name, \
                                    label_column_name, use_padding_for_context, max_seq_length, 
                                    LABEL2IDS[task_id], task_id, x), batched=True)
        all_test_dataset[task_id] = test_dataset

    return test_dataset

def build_model(model_path):
    return

def post_process():
    return

def main():
    pass


if __name__ == "__main__":
    path = "data/ALL_DATA_20231109/date_val.json"
    tokenizer_path = "ckpts/distillbert-base-uncase-ner-multihead"
    test_datasets = build_dataset(path, tokenizer_path)
    main()
