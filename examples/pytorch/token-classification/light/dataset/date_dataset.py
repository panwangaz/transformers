from .utils import DATASETS
from .name_dataset import NAMEDataset


@DATASETS.register_module()
class DATEDataset(NAMEDataset):
    def __init__(self,
                 # define dataset
                 ner_tags=("O", "USER_DATE", "NON_USER_DATE"), 
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
                 with_prefix_token=None,
                 task_id=None,
                 *args, 
                 **kwargs):
        super(DATEDataset, self).__init__(ner_tags=ner_tags, context_window=context_window, train_file=train_file, 
            validation_file=validation_file, test_file=test_file, use_augmented_data=use_augmented_data, 
            tokenizer_name_or_path=tokenizer_name_or_path, cache_dir=cache_dir, use_fast=use_fast, revision=revision, 
            use_auth_token=use_auth_token, text_column_name=text_column_name, label_column_name=label_column_name, 
            use_padding_for_context = use_padding_for_context, max_seq_length = max_seq_length, 
            with_prefix_token=with_prefix_token, task_id=task_id, *args, **kwargs)
        
    def _get_labels(self, data):
        for item in data:
            order, date_label_list = item["order"], item["label"]
            date_index = item["user_index"]
            item["flat_order"], item["flat_label"] = [], []
            for sentence, date_labels in zip(order, date_label_list):
                if self.with_prefix_token is not None:
                    text = f"{self.with_prefix_token} [USER] " if sentence[0] else f"{self.with_prefix_token} [ADVISOR] "
                else:
                    text = "[USER] " if sentence[0] else "[ADVISOR] "
                text += sentence[1].strip()
                text = text.split()
                item["flat_order"].append(text)
                label_flattened = [ "O" for _ in range(len(text))]
                date_count = 0
                # process date labels
                for label in date_labels:
                    date_value = "USER_DATE" if isinstance(date_index, (list, tuple, )) and date_count in date_index else "NON_USER_DATE"
                    date_count += 1
                    for label_index in range(len(label)):
                        tag_index = label[label_index] + 1 if self.with_prefix_token is None else label[label_index] + 2
                        if tag_index < len(label_flattened):
                            label_flattened[tag_index] = date_value
                item["flat_label"].append(label_flattened)
        return data


if __name__ == "__main__":
    train_file="data/NAME_DATE_NER/train.json"
    validation_file="data/NAME_DATE_NER/val.json"
    test_file="data/NAME_DATE_NER/val.json"
    tokenizer_path = "ckpts/bert-base-ner"
    ner_tags=("O", "USER_NAME", "USER_DATE", "NON_USER_NAME", "NON_USER_DATE")
    all_dataset = DATEDataset(ner_tags=ner_tags, train_file=train_file, 
        validation_file=validation_file, test_file=test_file, tokenizer_name_or_path=tokenizer_path)
    train_dataset = all_dataset.train()
    val_dataset = all_dataset.val()
    test_dataset = all_dataset.test()
    print(train_dataset)
    print(train_dataset[0])
