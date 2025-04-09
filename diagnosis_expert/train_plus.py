from my_package.dataset import FeaturesDataset
from my_package import Record, wait_gpu, DIM
import argparse
from my_package import model as Model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import time
from label import CANCER_5_LABEL_DICT as LABEL_DICT
import os
from pathlib import Path
from torch.nn import functional as F
from sklearn.metrics import f1_score

def focal_loss(logits, targets, gamma=2, alpha=0.25):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=5e-4, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--lr_gamma", default=0.975, type=float)
parser.add_argument("--lr_step_size", default=parser.parse_args().epochs // 100, type=int)
args = parser.parse_args()

device = wait_gpu(0, 6000)
feature_name = "hibou_l"
scale = ["Overview", "Small", "Medium"][2]
model_name = f"[{feature_name}_{scale}]transmil_cancer_5_focal"
record_path = os.path.join(Path(__file__).resolve().parent, "record")

class_num = len(set(i for i in LABEL_DICT.values() if i>=0))
model = Model.TransMIL(class_num, DIM[feature_name]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5) # type: ignore
scheduler = StepLR(optimizer, args.lr_step_size, args.lr_gamma)

train_set = FeaturesDataset("train", scale, feature_name) # type: ignore
val_set = FeaturesDataset("val", scale, feature_name) # type: ignore
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)

key_list = ["epoch", "train_loss", "train_acc", "tr_macro_f1", "val_loss", "val_acc", "val_macro_f1"]
curve_paras = [["train_loss", "val_loss"], ["train_acc", "val_acc"], ["tr_macro_f1", "val_macro_f1"]]
record = Record(os.path.join(record_path, model_name), key_list, curve_paras, args)

for epoch in range(args.epochs):
    tag_time = time.perf_counter()

    model.train()
    labels, results, tr_loss = [], [], 0
    for features, locations, text, main_label, sub_label in train_loader:
        label = torch.tensor(LABEL_DICT[(main_label.item(), sub_label.item())], dtype=torch.int64)
        if label < 0: continue

        logits = model(features[0].to(device))
        loss = focal_loss(logits, label.to(device))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()

        tr_loss += loss.item()
        labels.append(label.item())
        results.append(logits.argmax(dim=-1).item())
    tr_loss /= len(labels)
    tr_acc = sum(result == label for result, label in zip(results, labels)) / len(labels)
    tr_macro_f1 = f1_score(labels, results, average="macro")

    model.eval()
    labels, results, val_loss = [], [], 0
    for features, locations, text, main_label, sub_label in val_loader:
        label = torch.tensor(LABEL_DICT[(main_label.item(), sub_label.item())], dtype=torch.int64)
        if label < 0: continue

        with torch.no_grad():
            logits = model(features[0].to(device))
            loss = focal_loss(logits, label.to(device))

        val_loss += loss.item()
        labels.append(label.item())
        results.append(logits.argmax(dim=-1).item())
    val_loss /= len(labels)
    val_acc = sum(result == label for result, label in zip(results, labels)) / len(labels)
    val_macro_f1 = f1_score(labels, results, average="macro")

    record.record(epoch, tr_loss, tr_acc, tr_macro_f1, val_loss, val_acc, val_macro_f1)
    record.save_model(model, device)
    print(f"epoch:{epoch+1} loss=[{tr_loss:.4f}, {val_loss:.4f}] acc=[{tr_acc*100:.1f}%, {val_acc*100:.1f}%] macro_f1=[{tr_macro_f1:.3f}, {val_macro_f1:.3f}] use_time={time.perf_counter()-tag_time:.1f}s")