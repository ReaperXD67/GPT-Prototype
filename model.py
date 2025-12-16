# FILE: model_phase3.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CONFIGURATION (124M "Student") ---
class Phase3Config:
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
        self.d_model = 768        # Standard 124M Width
        self.n_layer = 12         # Standard 124M Depth
        self.n_head = 12          # 12 Heads (64 dim per head)
        self.block_size = 1024    # Trained Context (Fast)
        self.dropout = 0.0
        self.ffn_type = "gated_deep_mlp"

# --- DYNAMIC YaRN RoPE (Zero-Shot 8k Extension) ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024, original_max_seq_len=1024, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.original_max_seq_len = original_max_seq_len
        self.max_seq_len = max_seq_len
        self.register_buffer("inv_freq", self._compute_inv_freq(base), persistent=False)

    def _compute_inv_freq(self, base):
        return 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))

    def _yarn_linear_ramp_mask(self, min_val, max_val, num_rotations):
        if min_val == max_val:
            return torch.ones_like(num_rotations)
        mask = (num_rotations - min_val) / (max_val - min_val)
        return torch.clamp(mask, 0, 1)

    def forward(self, x, seq_len=None):
        # 1. Determine Scale (Dynamic)
        current_seq_len = seq_len if seq_len is not None else x.shape[1]
        scale = max(1.0, current_seq_len / self.original_max_seq_len)

        # 2. Standard RoPE (Short Context - No Change)
        if scale <= 1.0:
            t = torch.arange(current_seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

        # 3. Dynamic YaRN (Long Context Extension)
        # Calculate new "stretched" base
        new_base = self.base * (scale ** (self.dim / (self.dim - 2)))
        inv_freq_yarn = self._compute_inv_freq(new_base).to(x.device)
        
        # High/Low Frequency Ramp (Protects local grammar)
        beta_fast = 32
        beta_slow = 1
        dim_indices = torch.arange(0, self.dim, 2, device=x.device).float()
        ramp = self._yarn_linear_ramp_mask(beta_slow, beta_fast, dim_indices / self.dim)
        inv_freq_yarn = inv_freq_yarn * (1 - ramp) + (self.inv_freq.to(x.device) / scale) * ramp

        # Temperature Scaling (Entropy Fix)
        mscale = 0.1 * math.log(scale) + 1.0
        
        t = torch.arange(current_seq_len, device=x.device).type_as(inv_freq_yarn)
        freqs = torch.einsum('i,j->ij', t, inv_freq_yarn)
        emb = torch.cat((freqs, freqs), dim=-1)

        return (emb.cos() * mscale)[None, None, :, :], (emb.sin() * mscale)[None, None, :, :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# --- LAYERS ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)

class GatedDeepMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = dim * 4
        self.in_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.gate1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.val1  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.val2  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x):
        x = self.in_proj(x)
        # SiLU (Swish) is correct for SwiGLU. No GeLU.
        x = x + F.silu(self.gate1(x)) * self.val1(x)
        x = x + F.silu(self.gate2(x)) * self.val2(x)
        return self.out_proj(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = RotaryEmbedding(self.d_head, max_seq_len=config.block_size)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        cos, sin = self.rotary(v, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = GatedDeepMLP(config.d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_embedding(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, C), targets.reshape(B*T))
        return logits, loss
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())