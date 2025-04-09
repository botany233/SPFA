from my_package.dataset import GetFeaturesDataset
from my_package import wait_gpu
import torch
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
import os
os.environ["HF_HOME"] = "/home/chengfangchi/graduate2302/model"
from torchvision import transforms
import timm
from tqdm import tqdm

img_size = 256
device = wait_gpu(1, 11000)
output_path = "/home/chengfangchi/graduate2302/data/features"

model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True).to(device)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

model_name = Path(__file__).resolve().parent.name
if img_size == 512: model_name += f"_{img_size}"
for level in ["Overview", "Small", "Medium"]:
    dataset = GetFeaturesDataset("all", level, img_size, transform) # type: ignore
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    for id, imgs, locations, entrops in tqdm(data_loader):
        dir_path = os.path.join(output_path, model_name, id[0])
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, level) + ".pickle"
        if not os.path.exists(file_path):
            with torch.no_grad():
                chunk_imgs = torch.split(imgs[0], 64, dim=0)
                features = []
                for chunk_img in chunk_imgs:
                    features.append(model(chunk_img.to(device)).to("cpu"))
                features = torch.cat(features, dim=0)

            with open(file_path, "wb") as f:
                pickle.dump((features, locations[0], entrops[0]), f)