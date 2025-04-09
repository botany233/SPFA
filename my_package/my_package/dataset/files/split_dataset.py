import json
from sklearn.model_selection import train_test_split
from collections import Counter

with open("data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

invalid_ids = []
with open("invalid.txt", "r") as f:
    for line in f:
        invalid_ids.append(line.strip("\n"))

labels = []
ids = []
for key, value in raw_data.items():
    if key not in invalid_ids:
        labels.append(value["label"]*10 + value["sub_label"])
        ids.append(key)

random_seed = 2333
train_ids, temp_ids, train_labels, temp_labels = train_test_split(ids, labels, test_size=0.3, random_state=random_seed, stratify=labels)
val_ids, test_ids, val_labels, test_labels = train_test_split(temp_ids, temp_labels, test_size=0.5, random_state=random_seed, stratify=temp_labels)

def get_percent(data):
    count = Counter(data)
    return count

train_percent = get_percent(train_labels)
val_percent = get_percent(val_labels)
test_percent = get_percent(test_labels)

for key in sorted(train_percent.keys()):
    if key not in val_percent: val_percent[key] = 0
    if key not in test_percent: test_percent[key] = 0
    total = (train_percent[key] + val_percent[key] + test_percent[key])/100
    print(f"{key}: \t{train_percent[key]} \t{val_percent[key]} \t{test_percent[key]} \t{train_percent[key]/total:.1f}% \t{val_percent[key]/total:.1f}% \t{test_percent[key]/total:.1f}%")

with open("train.txt", "w", encoding="utf-8") as f:
    for id in train_ids:
        f.write(id + "\n")
with open("val.txt", "w", encoding="utf-8") as f:
    for id in val_ids:
        f.write(id + "\n")
with open("test.txt", "w", encoding="utf-8") as f:
    for id in test_ids:
        f.write(id + "\n")
with open("all.txt", "w", encoding="utf-8") as f:
    for id in ids:
        f.write(id + "\n")