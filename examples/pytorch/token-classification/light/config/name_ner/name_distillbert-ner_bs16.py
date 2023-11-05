"""
train_data: 2497 orders, val_data: 277 orders
labels: ("O", "USER_NAME", "NON_USER_NAME")
{
    'eval_precision': [0.9975236380009005, 0.7333333333333333, 0.6956521739130435], 
    'eval_recall': [1.0, 0.6285714285714286, 0.5517241379310345], 
    'eval_f1': [0.9987602840076637, 0.6769230769230768, 0.6153846153846154]
}
"""
_base_ = [
    "../_base_/dataset.py",
    "../_base_/model.py",
    "../_base_/training.py",
]

tokenizer_name = "ckpts/distillbert-base-uncase-ner"
output_dir = "work_dirs"
cache_dir = "work_dirs/test-name-ner"

data = dict(
    type="NAMEDataset",
    train_file="data/ALL_DATA_20231109/name_train.json",
    validation_file="data/ALL_DATA_20231109/name_val.json",
    test_file="data/ALL_DATA_20231109/name_val.json",
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
)

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
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_steps=50,
    eval_steps=1000,
    report_to="tensorboard",
    warmup_ratio=0.2,
    warmup_steps=0,
)