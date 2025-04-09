from my_package.dataset import GetFeaturesDataset
from my_package import wait_gpu
import torch
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
import os
os.environ["HF_HOME"] = "/home/chengfangchi/graduate2302/model"
from conch.open_clip_custom import create_model_from_pretrained
from tqdm import tqdm

img_size = 256
device = wait_gpu(3, 11000)
output_path = "/home/chengfangchi/graduate2302/data/features"

model, transform = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch") # type: ignore
model.eval().to(device)

model_name = Path(__file__).resolve().parent.name
if img_size == 512: model_name += f"_{img_size}"
# for level in ["Overview", "Small", "Medium"]:
if True:
    level = "Medium"
    dataset = GetFeaturesDataset("all", level, img_size, transform) # type: ignore
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for id, imgs, locations, entrops in tqdm(data_loader):
        dir_path = os.path.join(output_path, model_name, id[0])
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, level) + ".pickle"
        if not os.path.exists(file_path):
            with torch.no_grad():
                chunk_imgs = torch.split(imgs[0], 64, dim=0)
                features = []
                for chunk_img in chunk_imgs:
                    features.append(model.encode_image(chunk_img.to(device), proj_contrast=False, normalize=False).to("cpu"))
                features = torch.cat(features, dim=0)

            with open(file_path, "wb") as f:
                pickle.dump((features, locations[0], entrops[0]), f)