import os
import time
import logging
import argparse
import os.path as osp
import numpy as np
from mmengine import Config
from transformers import (
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
)
from transformers.trainer_utils import get_last_checkpoint
from dataset import NERDataset, DateNameDataset, DATASETS


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--epoches', type=int, default=10, help='total training epoches')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--lr', default=1e-5, help="the learning rate")
    parser.add_argument('--bs', default=16, help="batch size per gpu")
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
    if args.lr is not None:
        cfg.training.learning_rate=args.lr
    if args.lr is not None:
        cfg.training.num_train_epochs=args.epoches
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

    # define dataset
    # all_dataset = NERDataset(**data_args)
    all_dataset = DATASETS.build(data_args)
    raw_datasets, compute_metrics = all_dataset.dataset, all_dataset.compute_metrics
    tokenizer, data_collator = all_dataset.tokenizer, all_dataset.data_collator

    # define models
    model = AutoModelForTokenClassification.from_pretrained(model_args.model_name_or_path,
                                                            num_labels=len(data_args.ner_tags),
                                                            from_tf=bool(".ckpt" in model_args.model_name_or_path),
                                                            cache_dir=model_args.cache_dir,
                                                            revision=model_args.model_revision,
                                                            use_auth_token=True if model_args.use_auth_token else None,
                                                            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes)
    model.config.id2label = all_dataset.id2label
    model.config.label2id = all_dataset.label2id

    # define training args
    args = TrainingArguments(**training_args)
    trainer = Trainer(
        model,
        args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p=p, label_list=all_dataset.labels)
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

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(raw_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # When training ended
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(raw_datasets["validation"])
        metrics["eval_samples"] = min(max_eval_samples, len(raw_datasets["validation"]))

        for key, value in metrics.items():
            if isinstance(value, list):
                metrics[key] = sum(value) / len(value)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(raw_datasets["test"], metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [data_args.ner_tags[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")


if __name__ == "__main__":
    main()