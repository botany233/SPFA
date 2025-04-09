import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import cv2
import concurrent.futures
from typing import Literal
import json
from scipy.stats import entropy
from itertools import chain
from shutil import copy

class TestDataset(Dataset):
    def __init__(self,
                 main_path = "/home/chengfangchi/graduate2302/code/cfc/最后的冯如杯/my_package/my_package/dataset/files",
                 data_path = "/home/chengfangchi/graduate2302/data/local",
                 output_path = "/home/chengfangchi/graduate2302/data/only_overview"
                 ):
        self.data_path = data_path
        self.output_path = output_path
        with open(os.path.join(main_path, "data.json"), "r", encoding="utf-8") as f:
            self.data = json.load(f)
        with open(os.path.join(main_path, f"all.txt"), "r") as f:
            self.valid_id = [i.strip("\n") for i in f.readlines()]

    def __len__(self):
        return len(self.valid_id)

    def __getitem__(self, index):
        id = self.valid_id[index]
        data = self.data[id]
        text, main_label, sub_label = data["text"], data["label"], data["sub_label"]
        source_path = os.path.join(self.data_path, data["path"], id, "Overview.jpg")
        target_dir = os.path.join(self.output_path, f"{main_label}_{sub_label}")
        os.makedirs(target_dir, exist_ok=True)
        copy(source_path, os.path.join(target_dir, f"{id}.jpg"))
        return 1

if __name__=="__main__":
    import time
    import cv2
    from tqdm import tqdm

    dataset = TestDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    tag = time.perf_counter()
    for i in tqdm(dataloader):
        pass
    print(f"data num: {len(dataset)}, use_time = {round(time.perf_counter()-tag, 1)}s")