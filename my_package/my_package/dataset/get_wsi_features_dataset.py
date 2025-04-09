import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Literal
import json
import pickle

class GetWSIFeaturesDataset(Dataset):
    def __init__(self,
                 feature_name:str,
                 scale: Literal["Overview", "Small", "Medium"],
                 mode: Literal["train", "val", "test", "all"] = "all",
                 main_path = "/home/chengfangchi/graduate2302/code/cfc/最后的冯如杯/my_package/my_package/dataset/files",
                 data_path = "/home/chengfangchi/graduate2302/data/features"
                 ):
        self.scale = scale
        self.data_path = os.path.join(data_path, feature_name)
        with open(os.path.join(main_path, "data.json"), "r", encoding="utf-8") as f:
            self.data = json.load(f)
        with open(os.path.join(main_path, f"{mode}.txt"), "r") as f:
            self.valid_id = [i.strip("\n") for i in f.readlines()]

    def __len__(self):
        return len(self.valid_id)

    def __getitem__(self, index):
        id = self.valid_id[index]
        with open(os.path.join(self.data_path, id, self.scale) + ".pickle", "rb") as f:
            features, locations, entropys = pickle.load(f)
        return id, features, locations, entropys

if __name__=="__main__":
    import time
    from tqdm import tqdm

    dataset = GetWSIFeaturesDataset("Virchow2", "Medium")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    tag = time.perf_counter()
    for id, features, locations, entropys in tqdm(dataloader):
        pass
    print(f"data num: {len(dataset)}, use_time = {round(time.perf_counter()-tag, 1)}s")