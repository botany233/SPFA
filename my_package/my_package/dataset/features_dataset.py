import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Literal
import json
import pickle

class FeaturesDataset(Dataset):
    def __init__(self,
                 mode: Literal["train", "val", "test", "all"],
                 scale: Literal["Overview", "Small", "Medium"],
                 feature_name:str,
                 entropy_limit = -1,
                 main_path = "/home/chengfangchi/graduate2302/code/cfc/最后的冯如杯/my_package/my_package/dataset/files",
                 data_path = "/home/chengfangchi/graduate2302/data/features"
                 ):
        self.scale = scale
        self.data_path = os.path.join(data_path, feature_name)
        self.entropy_limit = entropy_limit
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
        with open(os.path.join(self.data_path, id, self.scale) + ".pickle", "rb") as f:
            imgs, locations, entropys = pickle.load(f)
        if self.entropy_limit > 0:
            masks = entropys > self.entropy_limit
            imgs = imgs[masks]
            locations = locations[masks]
        return imgs, locations, text, main_label, sub_label

if __name__=="__main__":
    import time
    from tqdm import tqdm

    dataset = FeaturesDataset("test", "Overview", "UNI")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    label_record = {}
    tag = time.perf_counter()
    for imgs, locations, text, main_label, sub_label in tqdm(dataloader):
        label = main_label.item()*100 + sub_label.item()
        if label in label_record:
            label_record[label] += 1
        else:
            label_record[label] = 1
    for key in sorted(label_record.keys()):
        print(key, label_record[key])
    print(f"data num: {len(dataset)}, use_time = {round(time.perf_counter()-tag, 1)}s")