import pickle
import torch
import torch.nn.functional as F
from my_package.dataset import WSIFeaturesDataset
from my_package import TestRecord, wait_gpu
from my_package import model as Model
from torch.utils.data import DataLoader
from my_package.label import MIX_LABEL_DICT as LABEL_DICT
from my_package.value import FLPOPS
from my_package.label import NAME_DICT
from environment_with_logits import Environmentv2
from get_logits_feature import get_logits_feature
import os
import csv
import seaborn as sns
from matplotlib import pyplot as plt

models_dir = "/home/chengfangchi/graduate2302/code/cfc/最后的冯如杯/actor_critic/record/linear_soft_logits"
device = wait_gpu(1, 8000)
choose_value = ["val_acc", "val_macro_f1"][0]

for model_name in os.listdir(models_dir):
    model_dir = os.path.join(models_dir, model_name)
    with open(os.path.join(model_dir, "record.csv"), "r") as f:
        data = list(csv.reader(f))
    value_datas, titles = data[1:], data[0]
    titles = [i.strip(" ") for i in titles]
    model_epoch, max_value, tr_acc = -1, 0, 0
    for i in value_datas:
        epoch, val_acc = int(i[titles.index("epoch")]), float(i[titles.index(choose_value)])
        if max_value <= val_acc:
            max_value, model_epoch, tr_acc = val_acc, epoch, float(i[titles.index("train_acc")])

    if os.path.exists(os.path.join(model_dir, "model", f"{model_epoch}.pickle")):
        with open(os.path.join(model_dir, "model", f"{model_epoch}.pickle"), "rb") as f:
            actor, critic = pickle.load(f)
    else:
        with open(os.path.join(model_dir, "model", f"{model_epoch}_0.pickle"), "rb") as f:
            actor = pickle.load(f)
        with open(os.path.join(model_dir, "model", f"{model_epoch}_1.pickle"), "rb") as f:
            critic = pickle.load(f)
    actor.eval().to(device)
    critic.eval().to(device)

    paras_dict = {}
    with open(os.path.join(model_dir, "paras.txt"), "r") as f:
        for line in f:
            key, value = line.split("=", 1)
            paras_dict[key.strip()] = value.strip()
    feature_names = [i.strip() for i in paras_dict["feature_names"].split(",")]
    initial_feature_names = [i.strip() for i in paras_dict["initial_feature_names"].split(",")]
    end_choice = bool(paras_dict["end_choice"])
    max_agent = int(paras_dict["max_agent"])
    flops_scale = float(paras_dict["flops_scale"])

    class_num = len(set(i for i in LABEL_DICT.values() if i>=0))
    class_names = [[] for _ in range(class_num)]
    scales = ["Medium" for _ in range(len(feature_names))] # type: ignore
    for key, value in LABEL_DICT.items():
        if value >= 0:
            class_names[value].append(NAME_DICT[key])
    record = TestRecord(model_name.replace("[", "(").replace("]", ")"), ["_".join(i) for i in class_names])
    test_set = WSIFeaturesDataset("val", scales, feature_names, return_logits=True) # type: ignore
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    num, correct = 0, 0
    initial_indexes = [feature_names.index(i) for i in initial_feature_names]
    foundation_names = [i[1:].split("]")[0].rsplit("_", 1)[0] for i in feature_names]
    flops_list = [FLPOPS[i] for i in foundation_names]
    env = Environmentv2(initial_indexes, flops_list, end_choice, max_agent, flops_scale)
    action_record = torch.zeros((class_num, max_agent+len(initial_indexes), len(feature_names)+int(end_choice)), dtype=torch.float32)
    for features, logits, text, main_label, sub_label in test_loader:
        label = LABEL_DICT[(main_label.item(), sub_label.item())]
        if label < 0: continue
        logits = torch.stack([get_logits_feature(i, LABEL_DICT) for i in logits[0]])
        # feature_state, logits_state = env.reset(features[0], logits)
        feature_state, logits_state = env.reset(logits, logits)

        with torch.no_grad():
            action_list = []
            critic_logits = critic(feature_state.to(device))
            while True:
                actor_logits = actor(logits_state.to(device), get_logits_feature(critic_logits.to("cpu"), LABEL_DICT).to(device))
                actor_probs = F.softmax(actor_logits, dim=-1)
                actor_probs[torch.tensor(action_list + initial_indexes, dtype=torch.int64)] = 0
                action = int(actor_probs.argmax(dim=-1))
                feature_next_state, logits_next_state, _, done, end = env.step(action)
                action_list.append(action)
                feature_state, logits_state = feature_next_state, logits_next_state
                critic_logits = critic(feature_state.to(device)).to("cpu")
                if done or end: break
        for index, action in enumerate(initial_indexes+action_list): action_record[label, index, action] += 1
        result = critic_logits.argmax(dim=-1).item()
        record.record(critic_logits, label)
        num += 1
        correct += int(label == result)
    record.end_record()
    print(f"{model_name}: epoch={model_epoch} tr_acc={tr_acc*100:.1f}% val_acc={max_value*100:.1f}% test_acc={correct/num*100:.1f}%")

    output_dir = "/home/chengfangchi/graduate2302/code/cfc/output"
    os.makedirs(output_dir, exist_ok=True)
    for index, class_action_record in enumerate(action_record):
        plt.figure(figsize=(class_action_record.shape[1], class_action_record.shape[0]))
        sns.heatmap(class_action_record / torch.sum(class_action_record[0]), annot=True, fmt=".2f", cmap='viridis', vmin=0, vmax=1)
        plt.title("action_record")
        plt.xlabel("options")
        plt.ylabel("round")
        plt.savefig(os.path.join(output_dir, f"{model_name}_{index}.png"))
        plt.close()
    total_action_record = torch.sum(action_record, dim=0)
    plt.figure(figsize=(total_action_record.shape[1], total_action_record.shape[0]))
    sns.heatmap(total_action_record / torch.sum(total_action_record[0]), annot=True, fmt=".2f", cmap='viridis', vmin=0, vmax=1)
    plt.title("action_record")
    plt.xlabel("options")
    plt.ylabel("round")
    plt.savefig(os.path.join(output_dir, f"{model_name}_all.png"))
    plt.close()