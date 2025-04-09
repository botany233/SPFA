from my_package.dataset import GetFeaturesDataset
from my_package import wait_gpu
import torch
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
import os
os.environ["HF_HOME"] = "/home/chengfangchi/graduate2302/model"
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from transformer import RealTransformer

img_size = 256
device = wait_gpu(1, 11000)
output_path = "/home/chengfangchi/graduate2302/data/features"

model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True).to(device)
model.eval()

transform = AutoImageProcessor.from_pretrained("histai/hibou-L", trust_remote_code=True)
transform = RealTransformer(transform)

model_name = Path(__file__).resolve().parent.name
for level in ["Overview", "Small", "Medium"]:
    dataset = GetFeaturesDataset("all", level, img_size, transform) # type: ignore
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    for id, imgs, locations, entrops in tqdm(data_loader):
        with torch.no_grad():
            chunk_imgs = torch.split(imgs[0], 64, dim=0)
            features = []
            for chunk_img in chunk_imgs:
                features.append(model(chunk_img.to(device)).pooler_output.to("cpu"))
            features = torch.cat(features, dim=0)

        dir_path = os.path.join(output_path, model_name, id[0])
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, level) + ".pickle"
        with open(file_path, "wb") as f:
            pickle.dump((features, locations[0], entrops[0]), f)