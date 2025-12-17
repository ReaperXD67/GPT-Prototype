import os
import time
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

# --- IMPORT YOUR MODEL & TOKENIZER ---
# We assume model.py is in the same folder
from model import GPT, GPTConfig 

# Try to import your custom tokenizer for printing samples. 
# If it fails, we just won't print text samples during training.
try:
    from train_tokenizer import Tokenizer
    tokenizer = Tokenizer("tokenizer_phase3_new.model") # Adjust filename if needed
    print("✅ Custom tokenizer loaded successfully.")
except Exception as e:
    print(f"⚠️ Could not load tokenizer: {e}")
    print("Training will run, but we won't print sample text.")
    tokenizer = None

# -----------------------------------------------------------------------------
# CONFIGURATION (Edit these settings for your B300/Strong GPU)
# -----------------------------------------------------------------------------
batch_size = 64        # B300 is strong, start with 64. If it crashes, try 32.
block_size = 1024      # Context length (how far back it looks)
max_iters = 50000      # Total training steps
eval_interval = 500    # How often to check validation loss
learning_rate = 3e-4   # Standard starting rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200       # How many steps to average for "clean" loss reporting
grad_clip = 1.0        # Prevents gradients from exploding

# Data File (Created by your prepare_data.py script)
train_data_path = 'train.bin' 

# -----------------------------------------------------------------------------
# PACKED DATASET CLASS (The Fix for Spikes)
# -----------------------------------------------------------------------------
class PackedDataset(Dataset):
    def __init__(self, bin_file, block_size):
        self.block_size = block_size
        # memmap reads the giant file from disk without eating all your RAM
        # We assume data was saved as uint16 (standard for vocab < 65k)
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"CRITICAL: {bin_file} not found! Run prepare_data.py first.")
            
        self.data = np.memmap(bin_file, dtype=np.uint16, mode='r')
        print(f"📂 Loaded dataset '{bin_file}' with {len(self.data)} tokens.")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Random sampling logic (Great for stable training)
        # We pick a random spot in the file
        i = torch.randint(len(self.data) - self.block_size, (1,)).item()
        
        # Grab a chunk of data
        chunk = torch.from_numpy(self.data[i : i + self.block_size + 1].astype(np.int64))
        
        # x is input, y is target (next token)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, dataloader):
    """Calculates a clean loss number to check progress."""
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    
    # We just run a few batches to get an average
    data_iter = iter(dataloader)
    for k in range(eval_iters):
        try:
            X, Y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            X, Y = next(data_iter)
            
        X, Y = X.to(device), Y.to(device)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
        
    model.train()
    return losses.mean()

# -----------------------------------------------------------------------------
# MAIN TRAINING SETUP
# -----------------------------------------------------------------------------
# 1. Setup Data
print("⚙️ Setting up data...")
dataset = PackedDataset(train_data_path, block_size)
train_loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    pin_memory=True, 
    num_workers=0 # Keep 0 for memmap stability
)

# 2. Setup Model
print("🧠 Initializing model...")
# Note: Ensure your model.py GPTConfig defaults match this, or pass arguments here
config = GPTConfig(block_size=block_size) 
model = GPT(config)
model.to(device)

# Enable torch.compile for speed (Works great on modern Linux/GPUs)
print("🚀 Compiling model (this takes a minute at the start)...")
try:
    model = torch.compile(model)
except Exception as e:
    print(f"Note: torch.compile skipped ({e}). Running in standard mode.")

# 3. Setup Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------
print(f"🔥 Training Started on {device}...")
iter_num = 0
t0 = time.time()

# Create an infinite iterator so we don't have to restart the loop
data_iter = iter(train_loader)

while iter_num < max_iters:
    # A. Get Batch
    try:
        X, Y = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        X, Y = next(data_iter)

    X, Y = X.to(device), Y.to(device)

    # B. Forward & Backward Pass
    logits, loss = model(X, Y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Clip gradients (Safety belt for training)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    optimizer.step()

    # C. Logging & Evaluation
    if iter_num % eval_interval == 0:
        # Calculate accurate loss
        val_loss = estimate_loss(model, train_loader)
        dt = time.time() - t0
        print(f"Step {iter_num}: loss {val_loss:.4f} | time {dt*1000:.2f}ms")
        
        # Save Checkpoint (Safety)
        if iter_num > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'config': config,
            }
            print(f"💾 Saving checkpoint to checkpoint_latest.pth")
            torch.save(checkpoint, 'checkpoint_latest.pth')
            
        # Optional: Print generation if tokenizer exists
        if tokenizer:
            try:
                # Generate a short sample (max_new_tokens=50)
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                print("--- Generating Sample ---")
                print(tokenizer.decode(model.generate(context, max_new_tokens=50)[0].tolist()))
                print("-------------------------")
            except Exception as e:
                pass # Don't crash if generation fails

        t0 = time.time()

    iter_num += 1