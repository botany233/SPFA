from my_package.dataset import GetFeaturesDataset
from my_package import wait_gpu
import torch
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
import os
os.environ["HF_HOME"] = "/home/chengfangchi/graduate2302/model"
import timm
from timm.data import resolve_data_config # type: ignore
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked # type: ignore
from tqdm import tqdm

img_size = 256
level = "Medium"
device = wait_gpu(1)
output_path = "/home/chengfangchi/graduate2302/data/features/Virchow2"

model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).to(device)
model.eval()

transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
dataset = GetFeaturesDataset("all", level, img_size, transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)

model_name = Path(__file__).resolve().parent.name
for id, imgs, locations, entrops in tqdm(data_loader):
    with torch.no_grad():
        chunk_imgs = torch.split(imgs[0], 64, dim=0)
        features = []
        for chunk_img in chunk_imgs:
            features.append(model(chunk_img.to(device)).to("cpu"))
        features = torch.cat(features, dim=0)
        features = torch.cat([features[:, 0], features[:, 1:].mean(1)], dim=-1).to(dtype = torch.float32)

    dir_path = os.path.join(output_path, model_name, id[0])
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, level) + ".pickle"
    with open(file_path, "wb") as f:
        pickle.dump((features, locations[0], entrops[0]), f)