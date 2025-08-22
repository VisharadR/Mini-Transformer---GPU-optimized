import torch

class KVMem:
    def __init__(self, n_layers, n_heads, head_dim, max_seq, batch, device):
        self.K = [torch.zeros(batch, n_heads, max_seq, head_dim, device=device, dtype=torch.float16)
                  for _ in range(n_layers)]
        self.V = [torch.zeros(batch, n_heads, max_seq, head_dim, device=device, dtype=torch.float16)
                  for _ in range(n_layers)]
        self.lens = torch.zeros(batch, dtype=torch.int32, device=device)


    def append(self, layer, b_idx, k_new, v_new):
        t = int(self.lens[b_idx].item())
        L = k_new.shape[-2]
        self.K[layer][b_idx, :, t:t+L, :].copy_(k_new)
        self.V[layer][b_idx, :, t:t+L, :].copy_(v_new)
        self.lens[b_idx] += L