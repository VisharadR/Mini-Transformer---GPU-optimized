import torch

def sample_greedy(logits): 
    return torch.argmax(logits, dim=-1)


def sample_topk(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    if temperature <= 0: temperature = 1.0
    probs = torch.softmax(logits / temperature, dim=-1)
    topv, topi = torch.topk(probs, k, dim=-1)
    idx = torch.multinomial(topv, 1)                # (B,1)
    return torch.gather(topi, -1, idx).squeeze(-1)  # (B,)