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
import random

class GetFeaturesDataset(Dataset):
    def __init__(self,
                 mode: Literal["train", "val", "test", "all"] = "all",
                 scale: Literal["Overview", "Small", "Medium"] = "Overview",
                 size: Literal[256, 512] = 256,
                 transformer = None,
                 max_patch: int = -1,
                 main_path = "/home/chengfangchi/graduate2302/code/cfc/最后的冯如杯/my_package/my_package/dataset/files",
                 data_path = "/home/chengfangchi/graduate2302/data/local"
                 ):
        self.scale = scale
        self.data_path = data_path
        self.size = size
        self.transformer = transformer
        self.max_patch = max_patch
        with open(os.path.join(main_path, "data.json"), "r", encoding="utf-8") as f:
            self.data = json.load(f)
        with open(os.path.join(main_path, f"{mode}.txt"), "r") as f:
            self.valid_id = [i.strip("\n") for i in f.readlines()]

    def __len__(self):
        return len(self.valid_id)

    def __getitem__(self, index):
        id = self.valid_id[index]
        data = self.data[id]
        imgs, locations, entropys = self.read_img(os.path.join(self.data_path, data["path"], id, self.scale))
        return id, imgs, locations, entropys

    def read_img(self, img_path):
        paras = []
        for patch_name in os.listdir(img_path):
            patch_path = os.path.join(img_path, patch_name)
            location = np.array(tuple(map(int, patch_name.split(".")[0].split("_"))))
            paras.append((patch_path, location))
        if self.max_patch >= 0:
            paras = random.sample(paras, min(len(paras), self.max_patch))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            result = list(chain.from_iterable(executor.map(self.read_patch, paras)))
            imgs, locations, entropys = zip(*result)
            if self.transformer is not None:
                imgs = list(executor.map(self.apply_transformer, imgs)) # type: ignore
        return np.array(imgs, dtype=np.float32), np.array(locations, dtype=np.int32), np.array(entropys, dtype=np.float32)

    def apply_transformer(self, img: np.ndarray):
        img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8)) # type: ignore
        return self.transformer(img) # type: ignore

    def read_patch(self, para):
        patch_path, location = para
        img = np.array(cv2.imread(patch_path))
        if len(img.shape) != 3:
            print(f"读取patch失败：{patch_path}")
            return []
        img = img[:,:,::-1].copy().transpose(2,0,1).copy()
        if self.size == 256:
            img = img.reshape(3, 2, 256, 2, 256).transpose(1, 3, 0, 2, 4).reshape(4, 3, 256, 256)
            offset = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            location = location[np.newaxis, :] * 2 + offset
        else:
            img = img[np.newaxis, :, :, :]
            location = location[np.newaxis, :]
        entropys = np.array([self.get_entrops(i) for i in img])
        return [(i, j, k) for i, j, k in zip(img, location, entropys)]

    def get_entrops(self, img):
        histogram, _ = np.histogram(img, bins=range(257))
        probabilities = histogram / np.sum(histogram)
        image_entropy = float(entropy(probabilities))
        return image_entropy

if __name__=="__main__":
    import time
    import cv2
    from tqdm import tqdm

    img_size = 256
    dataset = GetFeaturesDataset("all", "Overview", img_size, max_patch=-1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    tag = time.perf_counter()
    for id, imgs, locations, entrops in tqdm(dataloader):
        pass
    print(f"data num: {len(dataset)}, use_time = {round(time.perf_counter()-tag, 1)}s")