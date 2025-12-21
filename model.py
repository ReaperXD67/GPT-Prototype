import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CONFIGURATION ---
class GPTConfig:
    def __init__(self, vocab_size=4096, block_size=1024):
        self.vocab_size = vocab_size
        self.d_model = 768
        self.n_layer = 12
        self.n_head = 12
        self.block_size = block_size
        self.dropout = 0.0
        self.bias = False 
        self.untie_embeddings = True 

# --- ROTARY EMBEDDING ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # Reads the sequence length from the input tensor dimensions
        seq_len = x.shape[1] 
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    # Ensure cos/sin broadcast correctly across the batch and head dimensions
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# --- LAYERS ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class GatedDeepMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.d_model 
        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.up_proj   = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x)) 
        val  = self.up_proj(x)
        return self.down_proj(gate * val)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.c_proj.weight.data.zero_() 
        self.rotary = RotaryEmbedding(self.d_head, max_seq_len=config.block_size)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # QK-Norm (Critical for stability with Muon)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        # Apply RoPE
        # We pass 'x' to get the correct sequence length (T)
        cos, sin = self.rotary(x) 
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
        self.mlp = GatedDeepMLP(config) 

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# --- MAIN MODEL ---
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if not config.untie_embeddings:
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
        
        # Logit Softcapping (Prevents loss spikes)
        logits = 30.0 * torch.tanh(logits / 30.0)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates new tokens using the trained model.
        """
        for _ in range(max_new_tokens):
            # Crop context if it's too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get last step logits & scale by temperature
            logits = logits[:, -1, :] / temperature
            
            # Top-K Sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())