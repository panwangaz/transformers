import json
import datasets
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import ClassLabel, Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from .utils import classfication_metric, DATASETS


logger = datasets.logging.get_logger(__name__)

@DATASETS.register_module()
class NAMEDataset(object):
    """Light's NER Dataset"""
    def __init__(self, 
                 # define dataset
                 ner_tags=("O", "USER_NAME", "NON_USER_NAME"), 
                 context_window=6, 
                 train_file=None, 
                 validation_file=None, 
                 test_file=None, 
                 use_augmented_data=False,
                 # define tokenizer
                 tokenizer_name_or_path=None, 
                 cache_dir="test-ner", 
                 use_fast=True,
                 revision="main",
                 use_auth_token=False,
                 text_column_name="tokens", 
                 label_column_name="ner_tags",
                 use_padding_for_context = False,
                 max_seq_length = 150,
                 with_prefix_token=None,
                 task_id=None,
                 *args, 
                 **kwargs):
        assert len(ner_tags) > 0, "you must define net tags"
        assert tokenizer_name_or_path, "you need to define the correct tokenizer's infos"
        self._ner_tags = ner_tags
        self.context_window = context_window
        self._train_file, self._val_file, self._test_file = train_file, validation_file, test_file
        self.text_column_name, self.label_column_name = text_column_name, label_column_name
        self.use_padding_for_context, self.max_seq_length = use_padding_for_context, max_seq_length
        self._dataset, self.with_prefix_token = {}, with_prefix_token

        # prepare tokenizer
        if tokenizer_name_or_path is not None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_path,
                cache_dir=cache_dir,
                use_fast=use_fast,
                revision=revision,
                use_auth_token=use_auth_token,
            )
            self._data_collator = DataCollatorForTokenClassification(self._tokenizer)
            logger.info("finish initial the data tokenizer and collator!")

        # loading dataset for source data
        for name, file in zip(["train", "validation", "test"], [train_file, validation_file, test_file]):
            if file is not None:
                logger.info(f"start to process {file} data!")
                cur_dataset = self.get_dataset(file, context_window, use_augmented_data)
                self._dataset[name] = cur_dataset.map(lambda x : self.tokenize_and_align_labels(self._tokenizer, \
                        text_column_name, label_column_name, use_padding_for_context, max_seq_length, self.label2id, task_id, x), batched=True)

    @property
    def dataset(self):
        return self._dataset

    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def data_collator(self):
        return self._data_collator
    
    @property
    def labels(self) -> ClassLabel:
        return self._ner_tags

    @property
    def id2label(self):
        return dict(list(enumerate(self.labels)))

    @property
    def label2id(self):
        return {v: k for k, v in self.id2label.items()}

    def train(self):
        return self._dataset['train']
    
    def test(self):
        return self._dataset["test"]

    def val(self):
        return self._dataset["validation"]
    
    def _get_labels(self, data):
        for item in data:
            order = item["order"]
            name_label_list, name_index = item["label"], item["user_index"]
            item["flat_order"], item["flat_label"] = [], []
            for sentence, name_labels in zip(order, name_label_list):
                if self.with_prefix_token is not None:
                    text = f"{self.with_prefix_token} [USER] " if sentence[0] else f"{self.with_prefix_token} [ADVISOR] "
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
                        tag_index = label[label_index] + 1 if self.with_prefix_token is None else label[label_index] + 2
                        if tag_index < len(label_flattened):
                            label_flattened[tag_index] = name_value
                item["flat_label"].append(label_flattened)
        return data

    def get_dataset(self, path, window_size, use_augmented_data=False):
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
        data = self._get_labels(data)
        input_dict = {"tokens": [] , "labels": [], "bottom_len":[]}

        for order in tqdm(data):
            for line_index in range(len(order["order"])):
                # is use_augmented_data is true, the input is already formed in the data file
                if use_augmented_data and line_index != (len(order["order"]) - 1):
                    continue
                if order["order"][line_index][0]:
                    temp_tokens, temp_labels = [], []
                    input_dict["bottom_len"].append(len(order["flat_order"][line_index]))
                    # pre-context input
                    for i in range(window_size + 1):
                        now_index = line_index - i
                        if now_index >= 0:
                            temp_tokens.insert(0, order["flat_order"][now_index])
                            temp_labels.insert(0, order["flat_label"][now_index])
                        # check max seq
                        now_sentence = " ".join(_merge_lines(temp_tokens))
                        tokenized_now_sentence = self._tokenizer(now_sentence)
                        input_ids = tokenized_now_sentence["input_ids"]
                        if len(input_ids) > 1024: # 514
                            del(temp_tokens[0])
                            del(temp_labels[0])
                            break
                    if len(temp_tokens) > 0:
                        input_dict["tokens"].append(_merge_lines(temp_tokens))
                        input_dict["labels"].append(_merge_lines(temp_labels))
                    else:
                        del(input_dict["bottom_len"][-1])
        # with open("data/NAME_NER/tokens_and_labels.txt", "w+") as f:
        #     for token, label in zip(input_dict["tokens"], input_dict["labels"]):
        #         f.write(" ".join(token) + '\n')
        #         f.write(" ".join(label) + '\n')
        # f.close()
        return Dataset.from_pandas(pd.DataFrame({'tokens': input_dict["tokens"], 
                                                'ner_tags': input_dict["labels"],
                                                'bottom_len': input_dict["bottom_len"]}))

    @staticmethod
    def tokenize_and_align_labels(tokenizer, text_column_name, label_column_name, 
                                  padding, max_seq_length, label2id, task_id, examples):           
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            bottom_len = examples["bottom_len"][i]
            tokenized_bottom = tokenizer(
                examples[text_column_name][i][-bottom_len:],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )
            len_tokenized_bottom = len(tokenized_bottom["input_ids"]) - 2

            for j, word_idx in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                elif j <= (len(word_ids) - len_tokenized_bottom):
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label2id[label[word_idx]])
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        if task_id is not None:
            tokenized_inputs["task_ids"] = [task_id] * len(labels)
        return tokenized_inputs

    @staticmethod
    def compute_metrics(p, label_list):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels_num = [label_list.index(l) for labels in true_labels for l in labels]
        true_pred_num = [label_list.index(p) for pred in true_predictions for p in pred]
        results = classfication_metric(predictions=true_pred_num, references=true_labels_num)
        log_res = {
            "precision": results["precision"].tolist(),
            "recall": results["recall"].tolist(),
            "f1": results["f1"].tolist(),
            "accuracy": results["accuracy"],
            # "true_labels_num": true_labels_num,
            # "true_pred_num": true_pred_num,
        }
        logger.info(log_res)
        return log_res
    
    
if __name__ == "__main__":
    train_path = "data/NERDataset/tagged_train0000_01.json"
    val_path = "data/NERDataset/tagged_valid0000_01.json"
    test_path = "data/NERDataset/tagged_valid0000_01.json"
    tokenizer_path = "ckpts/bert-base-ner"
    all_dataset = NAMEDataset(train_file=train_path, val_file=val_path, test_file=test_path, tokenizer_name_or_path=tokenizer_path)
    train_dataset = all_dataset.train()
    val_dataset = all_dataset.val()
    test_dataset = all_dataset.test()
    print(train_dataset)
    print(train_dataset[0])
