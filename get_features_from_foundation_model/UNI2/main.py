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
from tqdm import tqdm

img_size = 256
device = wait_gpu(0, 11000)
output_path = "/home/chengfangchi/graduate2302/data/features"

timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,  # type: ignore
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
model.eval().to(device)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

model_name = Path(__file__).resolve().parent.name
for level in ["Overview", "Small", "Medium"]:
    dataset = GetFeaturesDataset("all", level, img_size, transform) # type: ignore
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    for id, imgs, locations, entrops in tqdm(data_loader):
        with torch.no_grad():
            chunk_imgs = torch.split(imgs[0], 64, dim=0)
            features = []
            for chunk_img in chunk_imgs:
                features.append(model(chunk_img.to(device)).to("cpu"))
            features = torch.cat(features, dim=0)

        dir_path = os.path.join(output_path, model_name, id[0])
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, level) + ".pickle"
        with open(file_path, "wb") as f:
            pickle.dump((features, locations[0], entrops[0]), f)