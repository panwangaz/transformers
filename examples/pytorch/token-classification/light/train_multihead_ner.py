import os
import time
import torch
import logging
import argparse
import datasets
import os.path as osp
import numpy as np
from typing import List, Optional
from packaging import version
from torch.utils.data import DataLoader
from mmengine import Config
from transformers import (
    TrainingArguments, 
    Trainer, 
)
from transformers.integrations.deepspeed import deepspeed_init
from transformers.utils import is_accelerate_available
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
    get_last_checkpoint,
    has_length,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
)
from transformers.integrations.deepspeed import deepspeed_init
from transformers.training_args import TrainingArguments
from dataset import DATASETS
from dataset.utils import reorg_output
from model.modeling_multihead_tokencls import MultiheadBertForTokenClassification, MultiheadDistilBertForTokenClassification
from model.configuration_multihead_tokencls import MultiHeadClsConfig

if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

logger = logging.getLogger(__name__)

ALL_DATASET_LABELS = {
    "eval_NAMEDataset": ["O", "USER_NAME", "NON_USER_NAME"],
    "eval_DATEDataset": ["O", "USER_DATE", "NON_USER_DATE"],
}

class MultiheadTrainer(Trainer):

    def evaluation_loop(
    self,
    dataloader: DataLoader,
    description: str,
    prediction_loss_only: Optional[bool] = None,
    ignore_keys: Optional[List[str]] = None,
    metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            if self.is_fsdp_enabled:
                self.model = model
            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model
            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")
        model.eval()
        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
                and (self.accelerator.sync_gradients or version.parse(accelerate_version) > version.parse("0.20.3"))
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs), ALL_DATASET_LABELS[metric_key_prefix]
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels), ALL_DATASET_LABELS[metric_key_prefix])
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def multi_evaluate(self, ignore_keys_for_eval=None):
        if isinstance(self.eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys_for_eval,
                    metric_key_prefix=f"eval_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
        else:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
        return metrics
    

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--train', action='store_true', help='whether to train')
    parser.add_argument('--eval', action='store_true', help='whether to eval')
    parser.add_argument('--test', action='store_true', help='whether to predict and save the results')
    parser.add_argument('--ckpt', help='the path of checkpoint when test or eval')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--epoches', type=int, default=10, help='total training epoches')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--lr', type=float, default=5e-5, help="the learning rate")
    parser.add_argument('--bs', type=int, default=16, help="batch size per gpu")
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.training.output_dir=args.work_dir
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    output_dir_time = osp.join(osp.splitext(osp.basename(args.config))[0], timestamp)
    cfg.training.output_dir = osp.join(cfg.training.output_dir, output_dir_time)
    ckpt_path = args.ckpt

    if args.lr is not None:
        cfg.training.learning_rate=args.lr
    if args.epoches is not None:
        cfg.training.num_train_epochs=args.epoches
    if args.train is not None:
        cfg.training.do_train=args.train
    if args.eval is not None:
        cfg.training.do_eval=args.eval
    if args.test is not None:
        cfg.training.do_predict=args.test
    if args.bs is not None:
        cfg.training.per_device_train_batch_size=args.bs
        cfg.training.per_device_eval_batch_size=args.bs
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # define args and configs
    model_args, data_args, training_args = cfg.model, cfg.data, cfg.training

    def build_single_dataset(args):
        nonlocal compute_metrics, tokenizer, data_collator
        logger.info(f"loading dataset: {args.type}")
        cur_dataset = DATASETS.build(args)
        all_dataset = cur_dataset.dataset
        tokenizer, data_collator = cur_dataset.tokenizer, cur_dataset.data_collator
        train_dataset, val_dataset = all_dataset["train"], all_dataset["validation"]
        compute_metrics = cur_dataset.compute_metrics
        return train_dataset, val_dataset
    
    # define dataset
    compute_metrics, tokenizer, data_collator = None, None, None
    all_train_datasets, all_val_datasets = [], dict()
    for data_arg in data_args:
        train_dataset, val_dataset = build_single_dataset(data_arg)
        all_train_datasets.append(train_dataset)
        all_val_datasets[data_arg.type] = val_dataset

    merge_train_datasets = None
    for train_dataset in all_train_datasets:
        if merge_train_datasets is None:
            merge_train_datasets = train_dataset.to_pandas()
        else:
            merge_train_datasets = merge_train_datasets._append(train_dataset.to_pandas())
    merge_train_datasets = datasets.Dataset.from_pandas(merge_train_datasets)
    merge_train_datasets.shuffle(seed=123)

    raw_datasets = datasets.DatasetDict(
        {"train": merge_train_datasets, "validation": all_val_datasets}
    )

    # define models
    config = MultiHeadClsConfig.from_pretrained(model_args.model_name_or_path)
    model_name = model_args.model_name_or_path.split('/')[1]
    if model_name == "bert-base-ner-multihead":
        model = MultiheadBertForTokenClassification.from_pretrained(model_args.model_name_or_path, config=config)
    elif model_name == "distillbert-base-uncase-ner-multihead":
        model = MultiheadDistilBertForTokenClassification.from_pretrained(model_args.model_name_or_path, config=config)
    model.resize_token_embeddings(len(tokenizer))
    
    # define training args
    args = TrainingArguments(**training_args)
    trainer = MultiheadTrainer(
        model,
        args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

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

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # training begin
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # training end
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # When training ended Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.multi_evaluate()
        for key, value in metrics.items():
            if isinstance(value, list):
                metrics[key] = sum(value) / len(value)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")
        if ckpt_path is not None:
            trainer._load_from_checkpoint(ckpt_path)
        
        start_time = time.time()
        predictions, labels, metrics = trainer.predict(raw_datasets["test"], metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"total inference time: {total_time}, total samples: {predictions.shape[0]}, per sample: {total_time / predictions.shape[0]}")

        for key, value in metrics.items():
            if isinstance(value, list):
                metrics[key] = sum(value) / len(value)

        # Remove ignored index (special tokens)
        true_predictions = [
            [data_args.ner_tags[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        test_datasets = raw_datasets["test"]
        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction, ori_data in zip(true_predictions, test_datasets):
                    tokens = ori_data["tokens"]
                    new_tokens_id = tokenizer(" ".join(tokens))["input_ids"]
                    new_tokens = tokenizer.convert_ids_to_tokens(new_tokens_id)[1:-1]
                    res = reorg_output(prediction, new_tokens, data_args["ner_tags"][1:])
                    writer.write(" ".join(tokens) + "\n")
                    for k, v in res.items():
                        writer.write(f"{k}: {v}" + "\n")
                    writer.write("\n")


if __name__ == "__main__":
    main()
