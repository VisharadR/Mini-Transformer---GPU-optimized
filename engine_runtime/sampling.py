import torch

def sample_greedy(logits): 
    return torch.argmax(logits, dim=-1)


def sample_topk(logits, k=50, temperature=1.0):
    probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
    topp = torch.topk(probs, k, dim=-1)
    idx = torch.multinomial(topk:= topp.values, 1)
    return torch.gather(topp.indices, -1, idx).squeeze(-1)