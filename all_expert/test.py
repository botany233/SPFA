import pickle
import torch
from my_package.dataset import WSIFeaturesDataset
from my_package import TestRecord, wait_gpu
from my_package import model as Model
from torch.utils.data import DataLoader
from my_package.label import MIX_LABEL_DICT as LABEL_DICT
from my_package.label import NAME_DICT
import os
import csv

models_dir = "/home/chengfangchi/graduate2302/code/cfc/最后的冯如杯/mult_transmil/record/对比实验/transmil_feature_meanpool"
device = wait_gpu(2, 8000)
choose_value = ["val_acc", "val_macro_f1"][0]

for model_name in os.listdir(models_dir):
    model_dir = os.path.join(models_dir, model_name)
    with open(os.path.join(model_dir, "record.csv"), "r") as f:
        data = list(csv.reader(f))
    value_datas, titles = data[1:], data[0]
    titles = [i.strip(" ") for i in titles]
    model_epoch, max_value, tr_acc = -1, 0, 0
    for i in value_datas:
        epoch, val_acc = int(i[titles.index("epoch")]), float(i[titles.index(choose_value)])
        if max_value <= val_acc:
            max_value, model_epoch, tr_acc = val_acc, epoch, float(i[titles.index("train_acc")])

    with open(os.path.join(model_dir, "model", str(model_epoch) + ".pickle"), "rb") as f:
        model = pickle.load(f).to(device)

    with open(os.path.join(model_dir, "paras.txt"), "r") as f:
        for line in f:
            key, value = line.split("=", 1)
            if key.strip() == "feature_names":
                feature_names = [i.strip() for i in value.split(",")]
                break

    class_names = [[] for _ in range(len(set(i for i in LABEL_DICT.values() if i>=0)))]
    scales = ["Medium" for _ in range(len(feature_names))] # type: ignore
    for key, value in LABEL_DICT.items():
        if value >= 0:
            class_names[value].append(NAME_DICT[key])
    record = TestRecord(model_name.replace("[", "(").replace("]", ")"), ["_".join(i) for i in class_names])
    test_set = WSIFeaturesDataset("test", scales, feature_names) # type: ignore
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    num, correct = 0, 0
    for features, text, main_label, sub_label in test_loader:
        label = LABEL_DICT[(main_label.item(), sub_label.item())]
        if label < 0: continue
        with torch.no_grad():
            logits = model(features[0].to(device)).to("cpu")
        result = logits.argmax(dim=-1).item()
        record.record(logits, label)
        num += 1
        correct += int(label == result)
    record.end_record()
    print(f"{model_name}: epoch={model_epoch} tr_acc={tr_acc*100:.1f}% val_acc={max_value*100:.1f}% test_acc={correct/num*100:.1f}%")