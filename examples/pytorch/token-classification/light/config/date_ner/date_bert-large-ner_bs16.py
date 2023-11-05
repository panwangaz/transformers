"""
train_data: 5000 orders, val_data: 957 orders
labels: ("O", "USER_DATE", "NON_USER_DATE")
{
    'eval_precision': [0.9997194006397665, 0.7377049180327869, 0.6956521739130435], 
    'eval_recall': [0.9991586740703349, 0.9278350515463918, 0.5245901639344263], 
    'eval_f1': [0.9994389587073608, 0.8219178082191781, 0.5981308411214953]
}
"""
_base_ = [
    "../_base_/dataset.py",
    "../_base_/model.py",
    "../_base_/training.py",
]

tokenizer_name = "ckpts/bert-large-ner"
output_dir = "work_dirs"
cache_dir = "work_dirs/test-name-ner"

data = dict(
    type="DATEDataset",
    train_file="data/ALL_DATA_20231109/date_train.json",
    validation_file="data/ALL_DATA_20231109/date_val.json",
    test_file="data/ALL_DATA_20231109/date_val.json",
    text_column_name="tokens",
    label_column_name="ner_tags",

    # ner dataset arguments
    ner_tags=("O", "USER_DATE", "NON_USER_DATE"), 
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