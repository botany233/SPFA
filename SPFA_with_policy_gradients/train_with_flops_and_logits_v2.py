from my_package.dataset import WSIFeaturesDataset
from my_package import Record, wait_gpu
from my_package.label import MIX_LABEL_DICT as LABEL_DICT
from my_package.value import FLPOPS
from my_package.feature_index import FEATURE_INDEX
from my_package import model as Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import time
import argparse
import os
from sklearn.metrics import f1_score
from environment_with_logits import Environmentv2
from get_logits_feature import get_logits_feature
from norm_grad import scale_grad
import seaborn as sns
from matplotlib import pyplot as plt

feature_indexs = [i for i in range(72, 81)]
feature_indexs += [7, 15, 23, 31, 39, 47, 55, 63, 71]
initial_indexes = [73]
# initial_indexes = [15]
end_choice = True
model_name = "mix_feature_10"

parser = argparse.ArgumentParser()
parser.add_argument("--actor_lr", default=5e-5, type=float)
parser.add_argument("--critic_lr", default=5e-4, type=float)
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--lr_gamma", default=0.88, type=float)
parser.add_argument("--max_agent", default=min(6, len(feature_indexs)-len(initial_indexes)), type=int)
parser.add_argument("--end_choice", default=end_choice, type=bool)
parser.add_argument("--rand_var", default=0.05, type=float)
parser.add_argument("--flops_scale", default=0.02, type=float)
parser.add_argument("--tr_temperature", default=1, type=float)
parser.add_argument("--lr_step_size", default=max(1, parser.parse_args().epochs // 50), type=int)
parser.add_argument("--feature_names", default=", ".join(FEATURE_INDEX[i] for i in feature_indexs), type=str)
parser.add_argument("--initial_feature_names", default=", ".join(FEATURE_INDEX[i] for i in initial_indexes), type=str)
args = parser.parse_args()

device = wait_gpu(3, 1000)
record_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "record")

feature_names = [FEATURE_INDEX[i] for i in feature_indexs]
scales = [i.split("]")[0].rsplit("_", 1)[1] for i in feature_names]
class_num = len(set(i for i in LABEL_DICT.values() if i>=0))
initial_indexes = [feature_indexs.index(i) for i in initial_indexes]
foundation_names = [i[1:].split("]")[0].rsplit("_", 1)[0] for i in feature_names]
flops_list = [FLPOPS[i] for i in foundation_names]
class_names = [str(i) for i in range(class_num)]
class_names.append("all")

# actor = Model.LinearPool(len(feature_indexs), len(feature_indexs) + int(end_choice), 16).to(device)
# actor = Model.S4Simple(16, len(feature_indexs) + int(end_choice)).to(device)
actor = Model.TransMIL(len(feature_indexs) + int(end_choice), 16, 512).to(device)
# actor = Model.Actorv2(16, len(feature_indexs) + int(end_choice), 512).to(device)
# critic = Model.TransMIL(class_num, 16).to(device)
# critic = Model.S4Simple(16, class_num).to(device)
critic = Model.LinearPool(len(feature_indexs), class_num, 16).to(device)
critic_loss_fn = nn.CrossEntropyLoss()

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, weight_decay=1e-5) # type: ignore
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr, weight_decay=1e-5) # type: ignore
actor_scheduler = StepLR(actor_optimizer, args.lr_step_size, args.lr_gamma)
critic_scheduler = StepLR(critic_optimizer, args.lr_step_size, args.lr_gamma)

train_set = WSIFeaturesDataset("train", scales, feature_names, return_logits=True) # type: ignore
val_set = WSIFeaturesDataset("val", scales, feature_names, return_logits=True) # type: ignore
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)

env = Environmentv2(initial_indexes, flops_list, end_choice, args.max_agent, args.flops_scale)
# env = Environmentv2(initial_indexes, None, end_choice, args.max_agent, args.flops_scale)

key_list = ["epoch", "tr_actor_loss", "tr_critic_loss", "train_acc", "tr_macro_f1", "val_critic_loss", "val_acc", "val_macro_f1", "tr_mean_flops", "val_mean_flops"]
curve_paras = [["tr_actor_loss"], ["tr_critic_loss", "val_critic_loss"], ["train_acc", "val_acc"], ["tr_macro_f1", "val_macro_f1"], ["tr_mean_flops", "val_mean_flops"]]
record = Record(os.path.join(record_path, model_name), key_list, curve_paras, args)

tr_acc = 0.95
for epoch in range(args.epochs):
    tr_action_record = torch.zeros((class_num+1, args.max_agent+len(initial_indexes), len(feature_indexs)+int(end_choice)), dtype=torch.float32)
    val_action_record = torch.zeros((class_num+1, args.max_agent+len(initial_indexes), len(feature_indexs)+int(end_choice)), dtype=torch.float32)
    tag_time = time.perf_counter()

    actor.train()
    critic.train()
    labels, results, tr_actor_loss, tr_critic_loss, tr_mean_flops = [], [], 0, 0, 0
    for features, logits, text, main_label, sub_label in train_loader:
        label = torch.tensor(LABEL_DICT[(main_label.item(), sub_label.item())], dtype=torch.int64)
        if label < 0: continue
        # soft_logits = torch.stack([get_logits_feature(i, LABEL_DICT, args.rand_var) for i in logits[0]/args.tr_temperature])
        logits = torch.stack([get_logits_feature(i, LABEL_DICT, args.rand_var) for i in logits[0]/args.tr_temperature])

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        # feature_state, logits_state = env.reset(features[0], features[0])
        # feature_state, logits_state = env.reset(features[0], logits)
        feature_state, logits_state = env.reset(logits, logits)

        critic_logits = critic(feature_state.to(device))
        critic_loss = critic_loss_fn(critic_logits, label.to(device))
        critic_loss.backward()
        action_list, actor_loss_list, critic_loss_list = [], [], []
        while True:
            actor_logits = actor(logits_state.to(device))
            actor_probs = F.softmax(actor_logits, dim=-1).clone()
            actor_probs[torch.tensor(action_list + initial_indexes, dtype=torch.int64)] = 0
            actor_probs = actor_probs / torch.sum(actor_probs)
            action = int(torch.distributions.Categorical(actor_probs).sample())

            next_feature_state, next_logits_state, reward, done, end = env.step(action)

            if not done:
                next_critic_logits = critic(next_feature_state.to(device))
                next_critic_loss = critic_loss_fn(next_critic_logits, label.to(device))
                next_critic_loss.backward()
                tr_mean_flops += flops_list[action]

                actor_loss = torch.log(actor_probs[action]) * (next_critic_loss - critic_loss).detach() * reward
                actor_loss.backward()

                critic_logits, critic_loss = next_critic_logits, next_critic_loss
                feature_state, logits_state = next_feature_state, next_logits_state
            else:
                result = critic_logits.argmax(dim=-1)
                if result.item() == label:
                    actor_loss = -torch.log(actor_probs[action]) * (1 - tr_acc) * 2
                    # actor_loss = -torch.log(actor_probs[action])
                else:
                    actor_loss = torch.log(actor_probs[action]) * tr_acc * 2
                    # actor_loss = torch.log(actor_probs[action])
                actor_loss.backward()

            critic_loss_list.append(critic_loss.item())
            actor_loss_list.append(actor_loss.item())
            action_list.append(action)
            if end or done: break
        for index, action in enumerate(initial_indexes+action_list): tr_action_record[label, index, action] += 1
        scale_grad(actor.parameters(), 1 / len(action_list))
        scale_grad(critic.parameters(), 1 / len(action_list))
        actor_optimizer.step()
        critic_optimizer.step()

        labels.append(label.item())
        results.append(critic_logits.argmax(dim=-1).item()) # type: ignore
        tr_actor_loss += sum(actor_loss_list) / len(actor_loss_list)
        tr_critic_loss += sum(critic_loss_list) / len(critic_loss_list)
    actor_scheduler.step()
    critic_scheduler.step()
    tr_critic_loss /= len(labels)
    tr_actor_loss /= len(labels)
    tr_mean_flops /= len(labels)
    tr_acc = sum(result == label for result, label in zip(results, labels)) / len(labels)
    tr_macro_f1 = f1_score(labels, results, average="macro")

    actor.eval()
    critic.eval()
    labels, results, val_actor_loss, val_critic_loss, val_mean_flops = [], [], 0, 0, 0
    for features, logits, text, main_label, sub_label in val_loader:
        label = torch.tensor(LABEL_DICT[(main_label.item(), sub_label.item())], dtype=torch.int64)
        if label < 0: continue
        logits = torch.stack([get_logits_feature(i, LABEL_DICT) for i in logits[0]])

        # feature_state, logits_state = env.reset(features[0], logits)
        # feature_state, logits_state = env.reset(features[0], features[0])
        feature_state, logits_state = env.reset(logits, logits)

        action_list = []
        with torch.no_grad():
            critic_logits = critic(feature_state.to(device))
            while True:
                actor_logits = actor(logits_state.to(device))
                actor_probs = F.softmax(actor_logits, dim=-1)
                actor_probs[torch.tensor(action_list + initial_indexes, dtype=torch.int64)] = 0
                action = int(actor_probs.argmax(dim=-1))

                next_feature_state, next_logits_state, reward, done, end = env.step(action)

                action_list.append(action)
                feature_state, logits_state = next_feature_state, next_logits_state

                critic_logits = critic(feature_state.to(device))
                if not done: val_mean_flops += flops_list[action]
                if done or end: break
            critic_loss = critic_loss_fn(critic_logits, label.to(device))
        val_critic_loss += critic_loss.item()
        labels.append(label.item())
        results.append(critic_logits.argmax(dim=-1).item())
        for index, action in enumerate(initial_indexes+action_list): val_action_record[label, index, action] += 1
    val_critic_loss /= len(labels)
    val_mean_flops /= len(labels)
    val_acc = sum(result == label for result, label in zip(results, labels)) / len(labels)
    val_macro_f1 = f1_score(labels, results, average="macro")

    record.record(epoch, tr_actor_loss, tr_critic_loss, tr_acc, tr_macro_f1, val_critic_loss, val_acc, val_macro_f1, tr_mean_flops, val_mean_flops)
    record.save_models([actor, critic], device)
    print(f"epoch:{epoch+1} tr_actor_loss={tr_actor_loss:.4f} critic_loss=[{tr_critic_loss:.4f} {val_critic_loss:.4f}] acc=[{tr_acc*100:.1f}% {val_acc*100:.1f}%] flops=[{tr_mean_flops:.1f} {val_mean_flops:.1f}] use_time={time.perf_counter()-tag_time:.1f}s")

    os.makedirs(os.path.join(record_path, model_name, "choose", str(epoch)))
    tr_action_record[-1] = torch.sum(tr_action_record[:-1], dim=0)
    for data, name in zip(tr_action_record, class_names):
        plt.figure(figsize=(data.shape[1], data.shape[0]))
        sns.heatmap(data / torch.sum(data[0]), annot=True, fmt=".2f", cmap='viridis', vmin=0, vmax=1)
        plt.title("action_record")
        plt.xlabel("options")
        plt.ylabel("round")
        plt.savefig(os.path.join(record_path, model_name, "choose", str(epoch), f"{name}_tr.png"))
        plt.close()
    val_action_record[-1] = torch.sum(val_action_record[:-1], dim=0)
    for data, name in zip(val_action_record, class_names):
        plt.figure(figsize=(data.shape[1], data.shape[0]))
        sns.heatmap(data / torch.sum(data[0]), annot=True, fmt=".2f", cmap='viridis', vmin=0, vmax=1)
        plt.title("action_record")
        plt.xlabel("options")
        plt.ylabel("round")
        plt.savefig(os.path.join(record_path, model_name, "choose", str(epoch), f"{name}_val.png"))
        plt.close()