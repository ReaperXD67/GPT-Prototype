import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

# --- IMPORT YOUR MODEL & TOKENIZER ---
from model import GPT, GPTConfig 

# Try to import tokenizer
try:
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file("tokenizer_phase3_new.json")
    print("✅ Custom tokenizer loaded successfully.")
except Exception as e:
    print(f"⚠️ Could not load tokenizer: {e}")
    tokenizer = None

# -----------------------------------------------------------------------------
# 1. THE MUON OPTIMIZER CLASS
# -----------------------------------------------------------------------------
class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-Schulz
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                
                # Muon only works on 2D matrices (Linear Layers)
                if g.ndim != 2: continue 

                state = self.state[p]
                
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Newton-Schulz Orthogonalization
                g_norm = g.norm() + 1e-8
                X = g / g_norm
                
                for _ in range(ns_steps):
                    A = X @ X.T
                    B = 3.4445 * A - 4.775 * (A @ A) + 2.0315 * (A @ A @ A)
                    X = B @ X
                
                update = X * 0.2 
                p.data.add_(update, alpha=-lr)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# SETTINGS FOR WINDOWS TESTING:
batch_size = 1          # Keep 1 to prevent OOM on laptop
eval_interval = 20      # Check often to see progress quickly
block_size = 1024       
max_iters = 50000       
muon_lr = 0.02          
adam_lr = 0.0006        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
grad_clip = 1.0         

# -----------------------------------------------------------------------------
# PACKED DATASET CLASS
# -----------------------------------------------------------------------------
class PackedDataset(Dataset):
    def __init__(self, bin_file, block_size):
        self.block_size = block_size
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"CRITICAL: {bin_file} not found!")
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        print(f"📂 Loaded dataset '{bin_file}' with {len(self.data)} tokens.")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        i = torch.randint(len(self.data) - self.block_size, (1,)).item()
        chunk = torch.from_numpy(self.data[i : i + self.block_size + 1].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

@torch.no_grad()
def estimate_loss(model, dataloader):
    model.eval()
    losses = torch.zeros(20) # Check 20 batches
    data_iter = iter(dataloader)
    for k in range(20):
        try:
            X, Y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            X, Y = next(data_iter)
        X, Y = X.to(device), Y.to(device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

# -----------------------------------------------------------------------------
# MAIN TRAINING SETUP
# -----------------------------------------------------------------------------
print("⚙️ Setting up data...")
dataset = PackedDataset("train.bin", block_size)
train_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=0)

print("🧠 Initializing model...")
config = GPTConfig(block_size=block_size) 
model = GPT(config)
model.to(device)

print("🌪️ Splitting parameters for Muon + AdamW...")
muon_params = []
adamw_params = []

for name, p in model.named_parameters():
    # CRITICAL: lm_head must go to AdamW for untied embeddings
    if p.ndim < 2 or "embedding" in name or "ln" in name or "bias" in name or "lm_head" in name:
        adamw_params.append(p)
    else:
        muon_params.append(p)

print(f"   - Muon Params: {len(muon_params)} tensors")
print(f"   - AdamW Params: {len(adamw_params)} tensors")

opt_muon = Muon(muon_params, lr=muon_lr, momentum=0.95)
opt_adam = torch.optim.AdamW(adamw_params, lr=adam_lr, betas=(0.9, 0.95), weight_decay=0.01)

# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------
print(f"🔥 Training Started on {device}...")
iter_num = 0
t0 = time.time()
data_iter = iter(train_loader)

while iter_num < max_iters:
    # 1. Get Batch
    try:
        X, Y = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        X, Y = next(data_iter)
    X, Y = X.to(device), Y.to(device)

    # 2. Forward Pass
    logits, loss = model(X, Y)
    
    # 3. Backward Pass
    opt_muon.zero_grad(set_to_none=True)
    opt_adam.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # 4. Step Optimizers
    opt_muon.step()
    opt_adam.step()

    # 5. Logging
    if iter_num % eval_interval == 0:
        val_loss = estimate_loss(model, train_loader)
        dt = time.time() - t0
        print(f"Step {iter_num}: loss {val_loss:.4f} | time {dt*1000:.2f}ms")
        
        # Save
        if iter_num > 0:
            ckpt = {
                'model': model.state_dict(),
                'opt_muon': opt_muon.state_dict(),
                'opt_adam': opt_adam.state_dict(),
                'iter_num': iter_num,
            }
            torch.save(ckpt, 'checkpoint_latest.pth')
            print("💾 Saved checkpoint.")

        # Print Sample (Now Works!)
        if tokenizer:
            try:
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                print("--- Sample ---")
                # Now model.generate() actually exists!
                print(tokenizer.decode(model.generate(context, max_new_tokens=50)[0].tolist()))
                print("--------------")
            except Exception as e:
                print(f"Sample generation failed: {e}")
        t0 = time.time()

    iter_num += 1