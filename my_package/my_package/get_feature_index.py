from feature_index import FEATURE_INDEX, TRAIN_TYPE_DICT, FEATURE_TYPE_DICT
import os
from pathlib import Path

if len(FEATURE_INDEX) > 0:
    current_index = max(FEATURE_INDEX.keys()) + 1
else:
    current_index = 0

new_dict = FEATURE_INDEX.copy()
feature_dir = "/home/chengfangchi/graduate2302/data/wsi_features"
for feature_name in sorted(os.listdir(feature_dir)):
    if feature_name not in new_dict.values():
        new_dict[current_index] = feature_name
        current_index += 1

with open(os.path.join(Path(__file__).resolve().parent, "feature_index.py"), "w") as f:
    f.write("FEATURE_INDEX = {\n")
    for key, value in new_dict.items():
        f.write(f'   {key}: "{value}",\n')
    f.write("}")

    f.write("\n\nFEATURE_TYPE_DICT = {\n")
    for key, value in new_dict.items():
        f.write(f'   {key}: "{value.split("]")[0].split("[")[1]}",\n')
    f.write("}")

    f.write("\n\nTRAIN_TYPE_DICT = {\n")
    for key, value in new_dict.items():
        try:
            train_type = value.split("]")[1].split("_", 1)[1]
        except:
            train_type = "default"
        f.write(f'   {key}: "{train_type}",\n')
    f.write("}")