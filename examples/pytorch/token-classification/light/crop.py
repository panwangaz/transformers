import json
from dataset.utils import load_json, write_json

json_path = [
    "data/mini_data/date_train.json",
    "data/mini_data/date_val.json",
    "data/mini_data/name_train.json",
    "data/mini_data/name_val.json"
]

for path in json_path:
    data = load_json(path)
    write_json(data[:100], path)
