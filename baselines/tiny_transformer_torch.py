import math
import torch 
import torch.nn as nn


# This code defines a tiny transformer model in PyTorch, including self-attention, MLP, and transformer blocks.

class TinySelfAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)


    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x) # (B, L, 3 * D)
        q, k, v = qkv.chunk(3, dim=-1) # (B, L, D), (B, L, D), (B, L, D)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2) # (B, n_heads, L, d_head)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2) # (B, n_heads, L, d_head)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2) # (B, n_heads, L, d_head)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = att.softmax(dim=-1) # (B, n_heads, L, L)
        out = att @ v # (B, n_heads, L, d_head)
        out = out.transpose(1, 2).contiguous().view(B, L, D) # (B, L, D)
        return self.proj(out) # (B, L, D)


class TinyMLP(nn.Module):
    def __init__(self, d_model=256, mlp_ratio=4):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x))) # (B, L, D)
    
class TinyTransformerBlock(nn.Module):
    def __init__(self,d_model=256, n_heads=4, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TinySelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = TinyMLP(d_model, mlp_ratio)
    def forward(self, x, mask=None):
        x = x +self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x
    

class TinyTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, depth=4, vocab_size=32000):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        # self.pos = nn.Parameter(torch.zeros(1, 1024, d_model))  # max sequence length of 1024
        self.blocks = nn.ModuleList([TinyTransformerBlock(d_model, n_heads) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def _sinusoidal_pos(self, L, d_model, device):
        # Generate sinusoidal positional encodings
        pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)   # [L,1]
        i = torch.arange(d_model, device=device, dtype=torch.float32).unsqueeze(0)  # [1,d]
        denom = torch.exp((i//2)*(-math.log(10000.0)/(d_model//2)))
        angles = pos * denom                                                 # [L, d]
        # even -> sin, odd -> cos
        pe = torch.zeros(L, d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return pe.unsqueeze(0)  # [1, L, d]
    
    def forward(self, idx):
        B, L = idx.shape
        device = idx.device
        x = self.tok(idx)
        pos = self._sinusoidal_pos(L, x.size(-1), device) # [1, L, d_model]
        x = x + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)
    

if __name__ == "__main__":
    model = TinyTransformer().cuda()
    idx = torch.randint(0, 32000, (2, 128), device='cuda')
    with torch.inference_mode():
        y = model(idx)
    print(y.shape)  # Should output: torch.Size([2, 128, 32000])

