from .utils import DATASETS
from .ner_dataset import NERDataset


@DATASETS.register_module()
class DateNameDataset(NERDataset):
    def __init__(self,
                 # define dataset
                 ner_tags=("O", "NAME"), 
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
                 *args, 
                 **kwargs):
        super(DateNameDataset, self).__init__(ner_tags=ner_tags, context_window=context_window, train_file=train_file, 
            validation_file=validation_file, test_file=test_file, use_augmented_data=use_augmented_data, 
            tokenizer_name_or_path=tokenizer_name_or_path, cache_dir=cache_dir, use_fast=use_fast, revision=revision, 
            use_auth_token=use_auth_token, text_column_name=text_column_name, label_column_name=label_column_name, 
            use_padding_for_context = use_padding_for_context, max_seq_length = max_seq_length, *args, **kwargs)
        
    def _get_labels(self, data):
        for item in data:
            order, date_label_list = item["order"], item["date_label"]
            user_index, name_label_list = item["user_index"], item["name_label"]
            item["flat_order"], item["flat_label"] = [], []
            for sentence, date_labels, name_labels in zip(order, date_label_list, name_label_list):
                if sentence[0]:
                    text = "[USER] "
                else:
                    text = "[ADVISOR] "
                text += sentence[1].strip()
                text = text.split()
                item["flat_order"].append(text)
                label_flattened = [ "O" for _ in range(len(text))]
                # process name labels
                for i, label in enumerate(name_labels):
                    if sentence[0]:
                        name_value = "USER_NAME"
                    else:
                        name_value = "NON_USER_NAME"
                    for label_index in range(len(label)):
                        tag_index = label[label_index] + 1
                        label_flattened[tag_index] = name_value
                # process date labels
                for i, label in enumerate(date_labels):
                    if user_index is not None and i in user_index:
                        date_value = "USER_DATE"
                    else:
                        date_value = "NON_USER_DATE"
                    for label_index in range(len(label)):
                        tag_index = label[label_index] + 1
                        label_flattened[tag_index] = date_value
                item["flat_label"].append(label_flattened)
        return data


if __name__ == "__main__":
    train_file="data/NAME_DATE_NER/train.json"
    validation_file="data/NAME_DATE_NER/val.json"
    test_file="data/NAME_DATE_NER/val.json"
    tokenizer_path = "ckpts/bert-base-ner"
    ner_tags=("O", "USER_NAME", "USER_DATE", "ADV_NAME", "ADV_DATE")
    all_dataset = DateNameDataset(ner_tags=ner_tags, train_file=train_file, 
        validation_file=validation_file, test_file=test_file, tokenizer_name_or_path=tokenizer_path)
    train_dataset = all_dataset.train()
    val_dataset = all_dataset.val()
    test_dataset = all_dataset.test()
    print(train_dataset)
    print(train_dataset[0])
