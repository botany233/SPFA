import pickle
import torch
from my_package.dataset import GetWSIFeaturesDataset
from my_package import wait_gpu
from my_package import model as Model
# from my_package.model import S4Model
from tzy_model import S4Model
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from tqdm import tqdm

output_path = "/home/chengfangchi/graduate2302/data/wsi_features"
device = wait_gpu(1, 8000)
# device = "cpu"

main_path = "/home/chengfangchi/graduate2302/code/cfc/最后的冯如杯/get_wsi_features/s4_mix"
for file in os.listdir(main_path):
    model_path = os.path.join(main_path, file)
    model_name = os.path.basename(model_path).rsplit(".")[0]
    feature_name, scale = model_name[1:].rsplit("]")[0].rsplit("_", 1)

    with open(model_path, "rb") as f:
        model = pickle.load(f).to(device)
    classification_head = nn.Sequential(model.relu, model.fc3).to("cpu")
    model.fc3 = nn.Sequential()
    model.relu = nn.Sequential()
    model.eval()

    dataset = GetWSIFeaturesDataset(feature_name, scale) # type: ignore
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    for id, features, _, _ in tqdm(data_loader):
        dir_path = os.path.join(output_path, model_name, id[0])
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, scale) + ".pickle"
        if not os.path.exists(file_path):
            try:
                with torch.no_grad():
                    wsi_feature = model(features.to(device))["logits"][0].to("cpu")
                    logits = classification_head(wsi_feature)
            except Exception as error:
                print(f"id: {error}")
            else:
                with open(file_path, "wb") as f:
                    pickle.dump((wsi_feature, logits), f)
    print(f"{model_name} done!")
print("all model done!")