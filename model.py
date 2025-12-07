# FILE: model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. CONFIGURATION ---
class ModelConfig:
    def __init__(self, ffn_type="swiglu"):
        self.vocab_size = 4096
        self.d_model = 128
        self.n_layer = 4
        self.n_head = 4
        self.block_size = 256
        self.dropout = 0.0
        self.ffn_type = ffn_type # 'swiglu' or 'deep_res_mlp'

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

# OPTION A: SwiGLU (Standard)
class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = int(4 * dim) 
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_val  = nn.Linear(dim, hidden_dim, bias=False)
        self.w_out  = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w_out(F.silu(self.w_gate(x)) * self.w_val(x))

# OPTION B: DeepResMLP (Custom 4-Layer ResNet)
class DeepResMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Configured for 4x width expansion
        hidden_dim = dim * 4 
        
        # Layer 1: Expand
        self.in_proj = nn.Linear(dim, hidden_dim)
        
        # Internal Residual Loop
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.GELU()
        
        # Layer 4: Contract
        self.out_proj = nn.Linear(hidden_dim, dim)
        
        # Scale weights to prevent instability in deep residuals
        self.apply(self._init_res_weights)

    def _init_res_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        # 1. Expand
        x = self.in_proj(x)
        
        # 2. Residual Block
        residual = x
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        
        # Residual Connection
        x = x + residual 
        x = self.act2(x)

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
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        
        # Select FFN type based on config
        if config.ffn_type == "swiglu":
            self.ffn = SwiGLU(config.d_model)
        elif config.ffn_type == "deep_res_mlp":
            self.ffn = DeepResMLP(config.d_model)
            
        self.ln1 = RMSNorm(config.d_model)
        self.ln2 = RMSNorm(config.d_model)

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
        self.token_embedding.weight = self.lm_head.weight # Weight Tying
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