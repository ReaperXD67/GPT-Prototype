# FILE: model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. CONFIGURATION ---

class Phase1Config:
    """Config for the 2M 'Baby' Model (Fast Testing)"""
    def __init__(self, ffn_type="gated_deep_mlp"):
        self.vocab_size = 4096
        self.d_model = 128
        self.n_layer = 4
        self.n_head = 4
        self.block_size = 256
        self.dropout = 0.0
        self.ffn_type = ffn_type 

class Phase2Config:
    """Config for the 50M 'Tank' Model (The Big Bet)"""
    def __init__(self, ffn_type="gated_deep_mlp"):
        self.vocab_size = 4096    # Keep small for stability
        self.d_model = 512        # Width: 4x larger
        self.n_layer = 6          # Depth: Deeper
        self.n_head = 8           # Heads: More parallel processing
        self.block_size = 512     # Context: 2x longer
        self.dropout = 0.0
        self.ffn_type = ffn_type

# Default generic config
ModelConfig = Phase1Config 

# --- 2. LAYERS ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed

# OPTION A: SwiGLU (The Standard Baseline)
class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = int(4 * dim) 
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_val  = nn.Linear(dim, hidden_dim, bias=False)
        self.w_out  = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w_out(F.silu(self.w_gate(x)) * self.w_val(x))

# OPTION B: GatedDeepMLP (The 'High-Dim Highway')
# This is the "Best" custom architecture we designed.
# It keeps data in high dimensions (512 -> 2048) while processing, avoiding bottlenecks.
class GatedDeepMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = dim * 4 # Tank Width (e.g., 512 -> 2048)
        
        # 1. Expand
        self.in_proj = nn.Linear(dim, hidden_dim)
        
        # 2. Deep Gated Processing Loop
        # We process the data TWICE in the high-dimensional space
        self.gate1 = nn.Linear(hidden_dim, hidden_dim)
        self.val1  = nn.Linear(hidden_dim, hidden_dim)
        
        self.gate2 = nn.Linear(hidden_dim, hidden_dim)
        self.val2  = nn.Linear(hidden_dim, hidden_dim)
        
        # 3. Contract
        self.out_proj = nn.Linear(hidden_dim, dim)
        
        # ZERO-INIT TRICK: 
        # Initialize output to zero so the block starts as an identity function.
        # This speeds up convergence massively for deep networks.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        # 1. Expand
        x = self.in_proj(x)
        
        # 2. Deep Processing
        residual = x
        
        # Gated Operation 1
        x = F.silu(self.gate1(x)) * self.val1(x)
        
        # Gated Operation 2
        x = F.silu(self.gate2(x)) * self.val2(x)
        
        # Residual connection inside the high-dim space
        x = x + residual
        
        # 3. Contract
        x = self.out_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape for Attention: (B, T, n_head, d_head) -> (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        # --- FLASH ATTENTION OPTIMIZATION ---
        # PyTorch 2.0+ automatically uses FlashAttention-2 on RTX 4070 if available
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)
        
        # SELECTOR
        if config.ffn_type == "swiglu":
            self.ffn = SwiGLU(config.d_model)
        elif config.ffn_type == "gated_deep_mlp":
            self.ffn = GatedDeepMLP(config.d_model)
        else:
            raise ValueError(f"Unknown ffn_type: {config.ffn_type}")

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class BabyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight Tying (Standard Practice)
        self.token_embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())