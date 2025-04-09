import pandas as pd
import json

data = pd.read_excel("gastric_data.xlsx", sheet_name = 'sheet1').values.tolist()
data_dict = {}

for slide_id, id, text, label, size, path in data:
    label = str(label)
    if "-" in label:
        main_label, sub_label = map(int, label.split("-"))
    else:
        main_label = int(label)
        if main_label >= 3:
            sub_label = main_label - 3
            main_label = 3
        else:
            sub_label = 0
    path_dir = path.split("/")[-2]
    if text[1] == "、" and text[0] in "0123456789":
        text = text[2:]
    data_dict[slide_id.strip("\n")] = {'text': text.strip("\n").replace("\n", ""),
                           'label': main_label,
                           'sub_label': sub_label,
                           "origin_label": label,
                           'size': size,
                           'path': path_dir}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data_dict, f, ensure_ascii=False, indent=4)