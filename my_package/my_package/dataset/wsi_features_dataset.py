import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Literal
import json
import pickle
from concurrent.futures import ThreadPoolExecutor

class WSIFeaturesDataset(Dataset):
    def __init__(self,
                 mode: Literal["train", "val", "test", "all"],
                 scales: list[Literal["Overview", "Small", "Medium"]],
                 feature_names: list[str],
                 return_logits: bool = False,
                 main_path = "/home/chengfangchi/graduate2302/code/cfc/最后的冯如杯/my_package/my_package/dataset/files",
                 data_path = "/home/chengfangchi/graduate2302/data/wsi_features"
                 ):
        assert len(scales) == len(feature_names)
        self.paras = tuple(zip(feature_names, scales))
        self.data_path = data_path
        self.return_logits = return_logits
        with open(os.path.join(main_path, "data.json"), "r", encoding="utf-8") as f:
            self.data = json.load(f)
        with open(os.path.join(main_path, f"{mode}.txt"), "r") as f:
            self.valid_id = [i.strip("\n") for i in f.readlines()]

    def __len__(self):
        return len(self.valid_id)

    def __getitem__(self, index):
        id = self.valid_id[index]
        data = self.data[id]
        text, main_label, sub_label = data["text"], data["label"], data["sub_label"]
        paras = [os.path.join(self.data_path, name, id, scale) + ".pickle" for name, scale in self.paras]
        features, logits = [], []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for feature, logit in executor.map(self.read_pickle, paras):
                features.append(feature)
                logits.append(logit)
        features = torch.stack(features)
        if self.return_logits:
            logits = torch.stack(logits)
            return features, logits, text, main_label, sub_label
        else:
            return features, text, main_label, sub_label

    @staticmethod
    def read_pickle(path):
        with open(path, "rb") as f:
            features, logits = pickle.load(f)
        return features, logits

if __name__=="__main__":
    import time
    import torch.nn.functional as F
    from tqdm import tqdm
    from my_package.feature_index import FEATURE_INDEX
    from my_package.label import MIX_LABEL_DICT as LABEL_DICT

    feature_indexs = [i for i in range(72, 81)]
    names = [FEATURE_INDEX[i] for i in feature_indexs]
    scales = ["Medium" for _ in range(len(names))]
    class_num = len(set(i for i in LABEL_DICT.values() if i>=0))

    dataset = WSIFeaturesDataset("train", scales, names, True) # type: ignore
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    record = [[] for _ in range(class_num)]
    tag = time.perf_counter()
    for features, logits, text, main_label, sub_label in tqdm(dataloader):
        label = LABEL_DICT[(main_label.item(), sub_label.item())]
        if label < 0: continue
        logits /= 5
        record[label].append(torch.mean(F.softmax(logits, dim=-1), dim=1))
    print(f"data num: {len(dataset)}, use_time = {round(time.perf_counter()-tag, 1)}s")
    record1 = [torch.mean(torch.cat(record[i]), dim=0) for i in range(class_num)]

    dataset = WSIFeaturesDataset("val", scales, names, True) # type: ignore
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    record = [[] for _ in range(class_num)]
    tag = time.perf_counter()
    for features, logits, text, main_label, sub_label in tqdm(dataloader):
        label = LABEL_DICT[(main_label.item(), sub_label.item())]
        if label < 0: continue
        record[label].append(torch.mean(F.softmax(logits, dim=-1), dim=1))
    print(f"data num: {len(dataset)}, use_time = {round(time.perf_counter()-tag, 1)}s")
    record2 = [torch.mean(torch.cat(record[i]), dim=0) for i in range(class_num)]
    for i, j in zip(record1, record2):
        print([round(float(t), 4) for t in i])
        print([round(float(t), 4) for t in j])
        print("=======")