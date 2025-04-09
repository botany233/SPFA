from my_package.dataset import FeaturesDataset
from my_package import Record, wait_gpu, DIM
import argparse
from my_package import model as Model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import time
from label import MIX_LABEL_DICT as LABEL_DICT
from tqdm import tqdm
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=5e-4, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr_gamma", default=0.975, type=float)
parser.add_argument("--lr_step_size", default=parser.parse_args().epochs // 100, type=int)
args = parser.parse_args()

device = wait_gpu(0, 8000)
feature_name = "CONCH"
scale = ["Overview", "Small", "Medium"][2]
model_name = f"[{feature_name}_{scale}]transmil2_mix_2"
record_path = os.path.join(Path(__file__).resolve().parent, "record")

class_num = len(set(i for i in LABEL_DICT.values() if i>=0))
model = Model.TransMIL2(class_num, DIM[feature_name]).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5) # type: ignore
scheduler = StepLR(optimizer, args.lr_step_size, args.lr_gamma)

train_set = FeaturesDataset("train", scale, feature_name) # type: ignore
val_set = FeaturesDataset("val", scale, feature_name) # type: ignore
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)

key_list = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
curve_paras = [["train_loss", "val_loss"], ["train_acc", "val_acc"]]
record = Record(os.path.join(record_path, model_name), key_list, curve_paras, args)

for epoch in range(args.epochs):
    tag_time = time.perf_counter()

    model.train()
    tr_num, tr_acc, tr_loss = 0, 0, 0
    for features, locations, text, main_label, sub_label in train_loader:
        label = torch.tensor(LABEL_DICT[(main_label.item(), sub_label.item())], dtype=torch.int64)
        if label < 0: continue

        logits = model(features[0].to(device), locations[0].to(device))
        loss = loss_fn(logits, label.to(device))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        tr_acc += int(logits.argmax(dim=-1).item() == label.item())
        tr_loss += loss.item()
        tr_num += 1
    tr_loss, tr_acc = tr_loss / tr_num, tr_acc / tr_num

    model.eval()
    val_num, val_acc, val_loss = 0, 0, 0
    for features, locations, text, main_label, sub_label in val_loader:
        label = torch.tensor(LABEL_DICT[(main_label.item(), sub_label.item())], dtype=torch.int64)
        if label < 0: continue

        with torch.no_grad():
            logits = model(features[0].to(device), locations[0].to(device))
            loss = loss_fn(logits, label.to(device))

        val_acc += int(logits.argmax(dim=-1).item() == label.item())
        val_loss += loss.item()
        val_num += 1
    val_loss, val_acc = val_loss / val_num, val_acc / val_num

    record.record(epoch, tr_loss, tr_acc, val_loss, val_acc)
    record.save_model(model, device)
    print(f"epoch:{epoch+1} train_loss={tr_loss:.5f} train_acc={tr_acc*100:.1f}% val_loss={val_loss:.5f} val_acc={val_acc*100:.1f}% use_time={time.perf_counter()-tag_time:.1f}s")