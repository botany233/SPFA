import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Literal
import json
import pickle
from typing import Union
from concurrent.futures import ThreadPoolExecutor

class MultFeaturesDataset(Dataset):
    def __init__(self,
                 mode: Literal["train", "val", "test", "all"],
                 scales: list[Literal["Overview", "Small", "Medium"]],
                 feature_names: list[str],
                 entropy_limits: list[Union[int, float]] = [],
                 main_path = "/home/chengfangchi/graduate2302/code/cfc/最后的冯如杯/my_package/my_package/dataset/files",
                 data_path = "/home/chengfangchi/graduate2302/data/features"
                 ):
        assert len(scales) == len(feature_names)
        if len(entropy_limits) > 0:
            assert len(scales) == len(entropy_limits)
        else:
            entropy_limits = [-1 for _ in range(len(scales))]
        self.paras = tuple(zip(feature_names, scales, entropy_limits))
        self.data_path = data_path
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
        paras = [(os.path.join(self.data_path, name, id, scale) + ".pickle", limit) for name, scale, limit in self.paras]
        with ThreadPoolExecutor(max_workers=8) as executor:
            features_and_locations = list(executor.map(lambda x: self.read_pickle(*x), paras))
        return features_and_locations, text, main_label, sub_label

    @staticmethod
    def read_pickle(path, entropy_limit):
        with open(path, "rb") as f:
            imgs, locations, entropys = pickle.load(f)
        if entropy_limit > 0:
            masks = entropys > entropy_limit
            imgs = imgs[masks]
            locations = locations[masks]
        return imgs, locations

if __name__=="__main__":
    import time
    from tqdm import tqdm

    scales = ["Medium", "Medium", "Medium", "Medium", "Medium", "Medium"]
    names = ["UNI", "UNI2", "Virchow2", "gigapath", "CONCH", "CONCH_512"]
    dataset = MultFeaturesDataset("all", scales, names) # type: ignore
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    tag = time.perf_counter()
    for features_and_locations, text, main_label, sub_label in tqdm(dataloader):
        pass
    print(f"data num: {len(dataset)}, use_time = {round(time.perf_counter()-tag, 1)}s")