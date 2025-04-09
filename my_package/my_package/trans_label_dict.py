import torch

def mix_to_five(logits):
    result = logits.argmax(dim=-1).item()
    new_result = label_trans_dict[result]
    new_logits = torch.zeros([5], dtype=torch.float32)
    new_logits[:3] += logits[:3]
    new_logits[3] = torch.max(logits[3:5])
    new_logits[4] += logits[5]
    return new_logits, new_result

label_trans_dict = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 4
}