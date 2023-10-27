"""
train_data: 5000 orders, val_data: 957 orders
labels: ("O", "USER_DATE", "NON_USER_DATE")
{
    'eval_precision': [0.9997755834829444, 0.7904761904761904, 0.7068965517241379], 
    'eval_recall': [0.9994952044422009, 0.8556701030927835, 0.6721311475409836], 
    'eval_f1': [0.9996353743023028, 0.8217821782178217, 0.6890756302521007], 
}
"""
_base_ = [
    "../_base_/dataset.py",
    "../_base_/model.py",
    "../_base_/training.py",
]

tokenizer_name = "ckpts/bert-base-ner"
output_dir = "work_dirs"
cache_dir = "work_dirs/test-date-ner"

data = dict(
    type="DATEDataset",
    train_file="data/DATE_NER/train.json",
    validation_file="data/DATE_NER/val.json",
    test_file="data/DATE_NER/val.json",
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
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_steps=50,
    report_to="tensorboard",
)