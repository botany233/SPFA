import torch
import torch.nn.functional as F
from .label import NUM_DICT

def get_logits_feature(logits: torch.Tensor, label_dict: dict):
    probs = F.softmax(logits, dim=-1)
    result = probs.argmax(dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=-1)
    sorted_probs, sorted_index = torch.sort(probs, descending=True)
    top2gap = sorted_probs[0] - sorted_probs[1]
    var = torch.var(probs)

    num_class = len(set(label_dict.values()))
    target_prob = torch.zeros((num_class, ), dtype=torch.float32)
    prob_dict = {key: value/sum(NUM_DICT.values()) for key, value in NUM_DICT.items()}
    for key, value in prob_dict.items():
        target_prob[label_dict[key]] += value
    kl_divergence = F.kl_div(torch.log(probs + 1e-7), target_prob, reduction="batchmean")
    return torch.tensor([*list(probs), result, entropy, top2gap, var, kl_divergence])

def get_logits_featurev2(logits: torch.Tensor, label_dict: dict, rand_var = 0, temperature = 1):
    logits += torch.randn_like(logits) * torch.var(logits, dim=-1, keepdim=True) * rand_var
    probs = F.softmax(logits / temperature, dim=-1)
    result = probs.argmax(dim=-1)
    result_one_hot = torch.zeros_like(probs)
    result_one_hot[result] = 1
    entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=-1)
    sorted_probs, sorted_index = torch.sort(probs, descending=True)
    top2gap = sorted_probs[0] - sorted_probs[1]
    var = torch.var(probs)

    num_class = len(set(label_dict.values()))
    target_prob = torch.zeros((num_class, ), dtype=torch.float32)
    prob_dict = {key: value/sum(NUM_DICT.values()) for key, value in NUM_DICT.items()}
    for key, value in prob_dict.items():
        target_prob[label_dict[key]] += value
    kl_divergence = F.kl_div(torch.log(probs + 1e-7), target_prob, reduction="batchmean")
    return torch.tensor([*list(probs), *list(result_one_hot), entropy, top2gap, var, kl_divergence])

if __name__ == "__main__":
    from my_package.label import MIX_LABEL_DICT
    logits = torch.tensor([12, 23, 0, 7, 12, -114], dtype=torch.float32)
    logits_feature = get_logits_feature(logits, MIX_LABEL_DICT)
    print(logits_feature.shape)