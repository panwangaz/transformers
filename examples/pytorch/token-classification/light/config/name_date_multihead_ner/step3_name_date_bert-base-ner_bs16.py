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
    'precision': [0.9977766661108332, 0.18181818181818182, 0.7333333333333333, 0.5882352941176471, 0.8831168831168831], 
    'recall': [0.9979985545115917, 0.2857142857142857, 0.8571428571428571, 0.36363636363636365, 0.7640449438202247], 
    'f1': [0.9978875979765411, 0.2222222222222222, 0.7904191616766466, 0.44943820224719105, 0.8192771084337349], 
    'accuracy': 0.9932550998025883, 
}
"""
_base_ = [
    "../_base_/dataset.py",
    "../_base_/model.py",
    "../_base_/training.py",
]

tokenizer_name = "ckpts/bert-base-ner-multihead-step3"
output_dir = "work_dirs/"
cache_dir = "work_dirs/test-name-date-ner"

data = dict(
    type="DateNameDataset",
    train_file="data/NAME_DATE_NER/train.json",
    validation_file="data/NAME_DATE_NER/val.json",
    test_file="data/NAME_DATE_NER/val.json",
    text_column_name="tokens",
    label_column_name="ner_tags",

    # ner dataset arguments
    ner_tags=("O", "USER_NAME", "NON_USER_NAME", "USER_DATE", "NON_USER_DATE"), 
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