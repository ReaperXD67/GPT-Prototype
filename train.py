# FILE: train.py
import torch
import math
from datasets import load_dataset
from tokenizers import Tokenizer
import time
import os
import json
from model import Phase2Config, BabyGPT 

# --- ⚙️ CONFIGURATION ---
SELECTED_FFN_TYPE = "gated_deep_mlp" 

# PRO SETTINGS
MAX_ITERS = 150000       
BATCH_SIZE = 16         # Physical Batch (Keeps VRAM safe)
GRAD_ACCUM_STEPS = 4    # <--- NEW! Simulates Batch Size 64 (16 * 4)
BLOCK_SIZE = 512        
LEARNING_RATE = 3e-4    
WARMUP_STEPS = 1000     
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_INTERVAL = 500  
RESUME_FILE = "checkpoint_latest.pth"

print(f"[INFO] Phase 2 Extended Run (Accumulation Active) 🚀")
print(f"[INFO] Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")

# 1. SETUP TOKENIZER
if not os.path.exists("tokenizer_phase1.json"):
    print("[ERROR] Tokenizer not found.")
    exit()
tokenizer = Tokenizer.from_file("tokenizer_phase1.json")
VOCAB_SIZE = tokenizer.get_vocab_size()

# 2. DATASET
print("[INFO] Streaming FineWeb-Edu...")
ds_stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
ds_stream = ds_stream.shuffle(seed=42, buffer_size=10000)

def data_generator():
    buffer = []
    iterator = iter(ds_stream)
    while True:
        try:
            text = next(iterator)["text"]
        except StopIteration:
            iterator = iter(ds_stream)
            text = next(iterator)["text"]
        encoded = tokenizer.encode(text).ids
        buffer.extend(encoded)
        chunk_size = (BLOCK_SIZE + 1) * BATCH_SIZE
        while len(buffer) >= chunk_size:
            chunk = buffer[:chunk_size]
            buffer = buffer[chunk_size:]
            data = torch.tensor(chunk, dtype=torch.long).view(BATCH_SIZE, BLOCK_SIZE + 1)
            yield data[:, :-1].to(DEVICE), data[:, 1:].to(DEVICE)

train_gen = data_generator()

# 3. INITIALIZE MODEL
config = Phase2Config(ffn_type=SELECTED_FFN_TYPE)
config.vocab_size = VOCAB_SIZE 
model = BabyGPT(config).to(DEVICE)

# 4. OPTIMIZER
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=0.1, 
    betas=(0.9, 0.95), 
    eps=1e-8
)

# 5. AUTO-RESUME
start_step = 0
history = {"loss": [], "accuracy": [], "perplexity": []}

if os.path.exists(RESUME_FILE):
    print(f"[RESUME] Found checkpoint! Loading...")
    checkpoint = torch.load(RESUME_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step'] + 1
    if 'history' in checkpoint: history = checkpoint['history']
    if "perplexity" not in history: history["perplexity"] = []
    print(f"[RESUME] Starting from Step {start_step}")
else:
    print("[INIT] Starting fresh.")

# Scheduler
def get_lr(step):
    # Adjust schedule to account for slower effective steps? 
    # For simplicity, we keep step-based scheduling, just smoother updates.
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / (WARMUP_STEPS + 1)
    if step > MAX_ITERS:
        return LEARNING_RATE * 0.1
    decay_ratio = (step - WARMUP_STEPS) / (MAX_ITERS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LEARNING_RATE * 0.1 + coeff * (LEARNING_RATE * 0.9)

# 6. TRAINING LOOP (WITH ACCUMULATION)
model.train()
start_time = time.time()
optimizer.zero_grad(set_to_none=True) # Initialize gradient

for step in range(start_step, MAX_ITERS):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Get micro-batch
    x, y = next(train_gen)
    
    # Forward Pass
    logits, loss = model(x, y)
    
    # Scale Loss (Critical for Accumulation Math)
    loss = loss / GRAD_ACCUM_STEPS 
    
    # Backward Pass (Accumulate Gradients)
    loss.backward()

    # --- THE ACCUMULATION LOGIC ---
    if (step + 1) % GRAD_ACCUM_STEPS == 0:
        # Update weights ONLY every 4 steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    # -------------------------------

    # Metrics (Unscale loss for logging so it looks normal)
    raw_loss = loss.item() * GRAD_ACCUM_STEPS
    ppl = math.exp(raw_loss) if raw_loss < 20 else 1e9

    preds = torch.argmax(logits, dim=-1)
    y_flat = y.view(-1)
    correct = (preds == y_flat).float().sum()
    accuracy = correct / y_flat.numel()

    history["loss"].append(raw_loss)
    history["accuracy"].append(accuracy.item())
    history["perplexity"].append(ppl)
    
    # Logging
    if step % 100 == 0:
        avg_loss = sum(history["loss"][-100:]) / 100 if step > 0 else history["loss"][0]
        avg_ppl = sum(history["perplexity"][-100:]) / 100 if step > 0 else history["perplexity"][0]
        dt = time.time() - start_time
        dt = max(dt, 0.001) 
        tok_sec = (step - start_step + 1) * BATCH_SIZE * BLOCK_SIZE / dt
        
        print(f"Step {step:05d} | Loss: {avg_loss:.4f} | PPL: {avg_ppl:.1f} | Acc: {accuracy:.2%} | LR: {lr:.2e}")

    # SAVE
    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }
        torch.save(checkpoint, RESUME_FILE) 
        print(f"   💾 Saved (Step {step})")

# Final Save
print(f"\n[DONE] Training Complete.")
torch.save(model.state_dict(), f"phase2_accum_{SELECTED_FFN_TYPE}.pth")
with open(f"history_phase2_accum.json", "w") as f:
    json.dump(history, f)