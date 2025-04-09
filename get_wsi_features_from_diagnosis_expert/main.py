import pickle
import torch
from my_package.dataset import GetWSIFeaturesDataset
from my_package import wait_gpu
from my_package import model as Model
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import os
from pathlib import Path

output_path = "/home/chengfangchi/graduate2302/data/wsi_features"
device = wait_gpu(1, 10000)

models_path = Path(__file__).resolve().parent / "models"

# for group_name in os.listdir(models_path):
for group_name in ["simple_mix"]:
    group_path = os.path.join(models_path, group_name)
    for model_name in os.listdir(group_path):
        model_path = os.path.join(group_path, model_name)
        model_name = model_name.rsplit(".")[0]
        feature_name, scale = model_name[1:].rsplit("]")[0].rsplit("_", 1)

        with open(model_path, "rb") as f:
            model = pickle.load(f).to(device)
        classification_head = model._fc2.to("cpu")
        model._fc2 = nn.Sequential()
        model.eval()

        dataset = GetWSIFeaturesDataset(feature_name, scale) # type: ignore
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        for id, features, _, _ in data_loader:
            dir_path = os.path.join(output_path, model_name, id[0])
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, scale) + ".pickle"
            if not os.path.exists(file_path):
                with torch.no_grad():
                    wsi_feature = model(features[0].to(device)).to("cpu")
                    logits = classification_head(wsi_feature)

                with open(file_path, "wb") as f:
                    pickle.dump((wsi_feature, logits), f)
        print(f"{model_name} done!")
print("all model done!")