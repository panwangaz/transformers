_base_ = [
    "../_base_/dataset.py",
    "../_base_/model.py",
    "../_base_/training.py",
]

tokenizer_name = "ckpts/bert-base-ner"
output_dir = "work_dirs"
cache_dir = "work_dirs/test-ner"

data = dict(
    type="NERDataset",
    train_file="data/NERDataset/tagged_train0000_01.json",
    validation_file="data/NERDataset/tagged_valid0000_01.json",
    test_file="data/NERDataset/tagged_valid0000_01.json",
    text_column_name="tokens",
    label_column_name="ner_tags",

    # ner dataset arguments
    ner_tags=("O", "NAME"), 
    use_augmented_data=False,
    context_window=6,
    use_padding_for_context=True,
    tokenizer_name_or_path=tokenizer_name,
    cache_dir=cache_dir,
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