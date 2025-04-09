import pickle
import torch
from my_package.dataset import FeaturesDataset
from my_package import TestRecordFive, wait_gpu
from my_package import model as Model
from tzy_model import S4Model
from torch.utils.data import DataLoader
from my_package import mix_to_five
from my_package.label import FIVE_LABEL_DICT as LABEL_DICT
from my_package.label import NAME_DICT
from tqdm import tqdm
import os

models_dir = "/home/chengfangchi/graduate2302/code/cfc/最后的冯如杯/get_wsi_features/s4_mix"
# device = wait_gpu(0, 8000)
device = "cpu"

for test_model in os.listdir(models_dir):
    with open(os.path.join(models_dir, test_model), "rb") as f:
        model = pickle.load(f).to(device)

    model_name = test_model.rsplit(".", 1)[0]
    feature_name, scale = model_name.split("]")[0].split("[")[1].rsplit("_", 1)
    class_names = [[] for _ in range(len(set(i for i in LABEL_DICT.values() if i>=0)))]
    for key, value in LABEL_DICT.items():
        if value >= 0:
            class_names[value].append(NAME_DICT[key])
    record = TestRecordFive(model_name.replace("[", "(").replace("]", ")"), ["_".join(i) for i in class_names])
    test_set = FeaturesDataset("val", scale, feature_name) # type: ignore
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    num, correct = 0, 0
    for features, locations, text, main_label, sub_label in tqdm(test_loader):
        label = LABEL_DICT[(main_label.item(), sub_label.item())]
        if label < 0: continue
        with torch.no_grad():
            logits = model(features.to(device))["logits"][0].to("cpu")
        logits, result = mix_to_five(logits)
        record.record(logits, label, result)
        num += 1
        correct += int(label == result)
    record.end_record()
    print(f"{model_name}: test_acc={correct/num*100:.1f}%")