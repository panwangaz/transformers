"""
train_data: 2497 orders, val_data: 277 orders
labels: ("O", "USER_NAME", "USER_DATE", "NON_USER_NAME", "NON_USER_DATE")
request:
{
    'precision': [0.957582800697269, 0.8333333333333334, 0.8278688524590164, 0.6567164179104478, 0.7325581395348837], 
    'recall': [0.9660023446658851, 0.9433962264150944, 0.6688741721854304, 0.8301886792452831, 0.6774193548387096], 
    'f1': [0.9617741464838051, 0.8849557522123894, 0.73992673992674, 0.7333333333333334, 0.7039106145251396], 
    'accuracy': 0.9270428015564203
}

bert-base-ner finetuning
{
    'eval_NAMEDataset_precision': [0.9977482548975456, 0.6785714285714286, 0.6538461538461539], 
    'eval_NAMEDataset_recall': [1.0, 0.5428571428571428, 0.5862068965517241], 
    'eval_NAMEDataset_f1': [0.9988728584310189, 0.603174603174603, 0.6181818181818182]
    'eval_DATEDataset_precision': [0.9996633374480979, 0.7317073170731707, 0.6904761904761905], 
    'eval_DATEDataset_recall': [0.9992708508609569, 0.9278350515463918, 0.47540983606557374], 
    'eval_DATEDataset_f1': [0.9994670556225632, 0.8181818181818181, 0.5631067961165048],
}

"""
_base_ = [
    "../_base_/dataset.py",
    "../_base_/model.py",
    "../_base_/training.py",
]

tokenizer_name = "ckpts/distillbert-base-uncase-ner-multihead"
output_dir = "work_dirs/"
cache_dir = "work_dirs/test-name-date-ner"

data = [
    dict(
        type="NAMEDataset",
        train_file="data/mini_data/name_train.json",
        validation_file="data/mini_data/name_val.json",
        test_file="data/mini_data/date_val.json",
        text_column_name="tokens",
        label_column_name="ner_tags",

        # ner dataset arguments
        ner_tags=("O", "USER_NAME", "NON_USER_NAME"), 
        use_augmented_data=False,
        context_window=6,
        use_padding_for_context=True,
        tokenizer_name_or_path=tokenizer_name,
        cache_dir=cache_dir,
        with_prefix_token=None,
        task_id=0,
    ),
     dict(
        type="DATEDataset",
        train_file="data/mini_data/date_train.json",
        validation_file="data/mini_data/date_val.json",
        test_file="data/mini_data/date_val.json",
        text_column_name="tokens",
        label_column_name="ner_tags",

        # ner dataset arguments
        ner_tags=("O", "USER_DATE", "NON_USER_DATE"), 
        use_augmented_data=False,
        context_window=6,
        use_padding_for_context=True,
        tokenizer_name_or_path=tokenizer_name,
        cache_dir=cache_dir,
        with_prefix_token=None,
        task_id=1,
    )
]

model = dict(
    model_name_or_path=tokenizer_name,
    cache_dir=cache_dir,
    ignore_mismatched_sizes=True,
)

training = dict(
    output_dir=output_dir,
    do_train=True,
    do_eval=True,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_steps=50,
    eval_steps=2000,
    report_to="tensorboard",
)
