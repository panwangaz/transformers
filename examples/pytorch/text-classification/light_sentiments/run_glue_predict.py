#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import sys
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import json
import datasets
import datetime
import numpy as np
from datasets import load_dataset
from datasets import Dataset
import evaluate
import transformers
from tqdm import tqdm
from trainer import Trainer
from scipy.special import softmax
import warnings
warnings.simplefilter("ignore")


# from data_collator import default_data_collator
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    # Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from utils.utils import collate_data_deprecated
from sklearn.metrics import precision_recall_fscore_support

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.25.0.dev0")
check_min_version("4.24.0.dev0")

BEST_PR_AUC_LOG_PATH = os.path.join("./work_dirs/best_pr_auc_metrics/", datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + ".json")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2":("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class CustomerArguments:
    my_train_dir: str = field(
        metadata={
                    "help": 
                    "Path to the taining file from customer."
                 }
    )
    my_validataion_dir: str = field(
        metadata={
                    "help": 
                    "Path to the validation file from customer."
                 }
    )
    save_model_path: str = field(
        metadata={
                    "help":
                    "The save model path."
                 }
    )
    balance: bool = field(
        metadata={
            "help":
            "ballance the data"
        }
    )
    regression: bool = field(
        metadata={
            "help":
            "regression task"
        }
    )
    use_special_token: bool = field(
        default=False,
        metadata={
            "help":
            "ballance the data"
        }
    )
    kick_ratio: float = field(
        default=0.0,
        metadata={
            "help":
            "ratio of the kicked data from the copy part."
        }
    )


def get_my_dataset(customer_args, data_args):
    dataset_dict = {}
    train_data, train_order_id, train_sample_num = collate_data_deprecated(
        path=customer_args.my_train_dir,
        balance=customer_args.balance,
        regression=customer_args.regression,
        max_len=data_args.max_seq_length
    )    
    valid_data,valid_order_id,valid_sample_num = collate_data_deprecated(
        path=customer_args.my_validataion_dir,
        regression=customer_args.regression,
        max_len=data_args.max_seq_length
    )

    dataset_dict["train_pre"] = Dataset.from_pandas(pd.DataFrame(train_data))
    dataset_dict["valid_pre"] = Dataset.from_pandas(pd.DataFrame(valid_data))

    return datasets.DatasetDict(dataset_dict), train_order_id, train_sample_num, valid_order_id, valid_sample_num
    

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, CustomerArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, customer_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, customer_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        raw_datasets,train_order_id,train_sample_num,valid_order_id,valid_sample_num = get_my_dataset(customer_args,data_args)

    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    label_info = {
        "label2id": {
            "negative": 0,
            "neutral": 1,
            "positive": 2
        },
        "id2label": {
            "0": "negative",
            "1": "neutral",
            "2": "positive"
        }
    }

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            # label_list = raw_datasets["train"].features["label"].names
            # use my label list
            label_list = list(label_info["label2id"].keys())
            num_labels = len(label_info["label2id"])
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        customer_args.save_model_path
    )


    tokenizer = AutoTokenizer.from_pretrained(
        customer_args.save_model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        customer_args.save_model_path
    )

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # result = tokenizer(examples[sentence1_key], padding=padding, max_length=max_seq_length, truncation=True)
        result = tokenizer(examples[sentence1_key], padding=padding, truncation=True)
        # result = tokenizer(examples[sentence1_key], examples[sentence2_key], padding=padding, max_length=max_seq_length, truncation=True)
        
        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        result["group_id"] = [group_id for group_id in examples["group_id"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_predict:
        if data_args.task_name is not None or data_args.test_file is not None:
            if "valid_pre" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            train_predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "train_pre"]
            valid_predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "valid_pre"]
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(valid_predict_dataset), data_args.max_predict_samples)
                valid_predict_dataset = valid_predict_dataset.select(range(max_predict_samples))
                train_predict_dataset = train_predict_dataset.select(range(max_predict_samples))

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("/data/yangxiaoran/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--glue/05234ba7acc44554edcca0978db5fa3bc600eeee66229abe79ff9887eacaf3ed/glue.py", data_args.task_name)
    else:
        metric = evaluate.load("accuracy")


    def my_metrics(pred_labels, true_labels): 
        metric_dict = {}
        precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, pred_labels, average=None)
        metric_dict['precision_negative'] = precision[0]
        metric_dict['recall_negative'] = recall[0]
        metric_dict['f1_score_negative'] = f1_score[0]
        metric_dict['precision_neutral'] = precision[1]
        metric_dict['recall_neutral'] = recall[1]
        metric_dict['f1_score_neutral'] = f1_score[1]
        metric_dict['precision_positive'] = precision[2]
        metric_dict['recall_positive'] = recall[2]
        metric_dict['f1_score_positive'] = f1_score[2]
        
        return metric_dict
        

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction, group_id_list: list):
        id2label_tb = {0: "negative", 1: "neutral", 2:"positive"}
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            # compute my metrics here
            my_result = my_metrics(preds, p.label_ids)
            print(my_result)
            for a_key in list(my_result.keys()):
                result[a_key] = my_result[a_key]
            # if len(result) > 1:
            #     result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        kick_ratio=customer_args.kick_ratio
    )

    with open(BEST_PR_AUC_LOG_PATH, "w") as fout:
        dict_item = {
            "best_pr_auc": 0,
            "roc_auc": 0,
            "recalls": [],
            "precisions": [],
            "args": customer_args.__dict__
            }
        dict_item["args"]["model_name"] = model_args.model_name_or_path
        json.dump(dict_item, fout)

    if training_args.do_predict:
        order_id_2_feature={}
        order_id_predict={}
        feature_name=['bert_'+str(i) for i in range(768)]
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name,data_args.task_name]
        predict_datasets = [train_predict_dataset,valid_predict_dataset]
        order_id_list = [train_order_id,valid_order_id]
        sample_num_list = [train_sample_num,valid_sample_num]

        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task, order_id_data, sample_num_data in zip(predict_datasets, tasks, order_id_list, sample_num_list):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset_new = predict_dataset.remove_columns("label")
            predictions,hidden_states = trainer.predict(predict_dataset_new, metric_key_prefix="predict")
            predictions = predictions.predictions
            
            # predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            predictions = np.squeeze(predictions) if is_regression else softmax(predictions, axis=1)

            if trainer.is_world_process_zero():
                for index, item in enumerate(tqdm(predictions)):
                    features = {}
                    
                    if order_id_data[index] not in order_id_predict:
                        order_id_predict[order_id_data[index]] = {}
                    if sample_num_data[index] not in order_id_predict[order_id_data[index]]:
                        order_id_predict[order_id_data[index]][sample_num_data[index]] = {}
                    order_id_predict[order_id_data[index]][sample_num_data[index]]['true'] = int(predict_dataset['label'][index])
                    
                    # order_id_predict[order_id_data[index]][sample_num_data[index]]['predict'] = int(item)
                    order_id_predict[order_id_data[index]][sample_num_data[index]]['predict'] = list(item.astype(np.float))
                    
                    for i in range(768):
                        features[feature_name[i]] = list(hidden_states[index].astype(float))[i]
                    if order_id_data[index] not in order_id_2_feature:
                        order_id_2_feature[order_id_data[index]] = {}
                    if sample_num_data[index] not in order_id_2_feature[order_id_data[index]]:
                        order_id_2_feature[order_id_data[index]][sample_num_data[index]] = features

    with open('./data/sentence_feature_pre.json', 'w') as f:
        json.dump(order_id_2_feature, f)
    with open('./tmp/sst2/order_id_predict_pre.json', 'w') as f:
        json.dump(order_id_predict, f)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"


if __name__ == "__main__":
    main()
