import pickle
import torch
from my_package.dataset import FeaturesDataset
from my_package import TestRecord, wait_gpu
from my_package import model as Model
from torch.utils.data import DataLoader
from label import CANCER_5_LABEL_DICT as LABEL_DICT
from label import NAME_DICT
from pathlib import Path
import os

model_dir = "/home/chengfangchi/graduate2302/code/cfc/最后的冯如杯/transmil/record/[UNI_Medium]transmil_cancer_5_plus"
device = wait_gpu(1, 5000)
model_epochs = [10, 17, 33, 34]

for model_epoch in model_epochs:
    with open(os.path.join(model_dir, "model", str(model_epoch) + ".pickle"), "rb") as f:
        model = pickle.load(f).to(device)

    model_dir = Path(model_dir).resolve()
    model_name = model_dir.name
    feature_name, scale = model_name.split("]")[0].split("[")[1].rsplit("_", 1)
    class_names = [[] for _ in range(len(set(i for i in LABEL_DICT.values() if i>=0)))]
    for key, value in LABEL_DICT.items():
        if value >= 0:
            class_names[value].append(NAME_DICT[key])
    record = TestRecord(model_name.replace("[", "(").replace("]", ")"), ["_".join(i) for i in class_names])
    test_set = FeaturesDataset("val", scale, feature_name) # type: ignore
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    num, correct = 0, 0
    for features, locations, text, main_label, sub_label in test_loader:
        label = LABEL_DICT[(main_label.item(), sub_label.item())]
        if label < 0: continue
        with torch.no_grad():
            logits = model(features[0].to(device)).to("cpu")
        result = logits.argmax(dim=-1).item()
        record.record(logits, label)
        num += 1
        correct += int(label == result)
    record.end_record()
    print(f"{model_name}: epoch={model_epoch} test_acc={correct/num*100:.1f}%")