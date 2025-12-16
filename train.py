# FILE: train_phase3.py
import torch
import math
import os
import time
from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer
from model import Phase3Config, StudentGPT

# --- ⚙️ CONFIGURATION ---
MAX_STEPS = 50000        
BATCH_SIZE = 32          # 124M fits easily on B200 with batch 32
GRAD_ACCUM = 4           
BLOCK_SIZE = 1024        
LEARNING_RATE = 4e-4     
WARMUP_STEPS = 2000      
CHECKPOINT_DIR = "checkpoints_phase3"
RESUME_FILE = f"{CHECKPOINT_DIR}/latest.pth"
DEVICE = "cuda"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"[INFO] Phase 3: The 124M Student - Dynamic YaRN Edition 🚀")

# 1. LOAD TOKENIZER
# Note: Using your existing Phase 1 tokenizer (4096)
if os.path.exists("tokenizer_phase3_new.json"):
    print("[INFO] Loading Phase 3 tokenizer...")
    tokenizer_path = "tokenizer_phase3_new.json"
elif os.path.exists("tokenizer_phase1.json"):
    print("[INFO] Loading Phase 1 tokenizer (4096)...")
    tokenizer_path = "tokenizer_phase1.json"
else:
    print("❌ Error: No tokenizer found. Upload 'tokenizer_phase1.json'!")
    exit()

tokenizer = Tokenizer.from_file(tokenizer_path)
VOCAB_SIZE = tokenizer.get_vocab_size()

# 2. THE DATA MIX (Text + Code + Math)
def get_data_stream(start_step=0):
    print("[DATA] Connecting to Open SOTA Streams...")
    ds_text = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    ds_code = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)
    ds_math = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True)

    dataset = interleave_datasets(
        [ds_text, ds_code, ds_math], 
        probabilities=[0.45, 0.35, 0.20], 
        seed=42
    )
    
    skip_n = start_step * BATCH_SIZE * GRAD_ACCUM
    if skip_n > 0:
        print(f"[RESUME] ⏩ Skipping {skip_n} samples...")
        dataset = dataset.skip(skip_n)
        
    return iter(dataset)

# 3. MODEL INIT
config = Phase3Config(vocab_size=VOCAB_SIZE)
model = StudentGPT(config).to(DEVICE)
print(f"[INFO] Model Parameters: {model.get_num_params()/1e6:.2f}M")

print(f"[INFO] Compiling model...")
try:
    model = torch.compile(model) 
except Exception as e:
    print(f"[WARN] Compilation skipped: {e}")

# 4. OPTIMIZER
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
scaler = torch.amp.GradScaler('cuda')

# 5. RESUME
start_step = 0
history = {"loss": [], "ppl": []}

if os.path.exists(RESUME_FILE):
    print(f"[RESUME] Loading checkpoint...")
    ckpt = torch.load(RESUME_FILE)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_step = ckpt['step']
    history = ckpt['history']
    print(f"[RESUME] Starting at Step {start_step}")

train_gen = get_data_stream(start_step)

def get_lr(step):
    if step < WARMUP_STEPS: return LEARNING_RATE * (step+1)/(WARMUP_STEPS+1)
    if step > MAX_STEPS: return LEARNING_RATE * 0.1
    decay_ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LEARNING_RATE * 0.1 + coeff * (LEARNING_RATE * 0.9)

# 6. TRAINING LOOP
model.train()
t0 = time.time()

print("🚀 Training Started...")
for step in range(start_step, MAX_STEPS):
    lr = get_lr(step)
    for param_group in optimizer.param_groups: param_group['lr'] = lr
    
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    step_executed = False  
    
    for _ in range(GRAD_ACCUM):
        try:
            batch_texts = []
            for _ in range(BATCH_SIZE):
                sample = next(train_gen)
                txt = (sample.get('text') or sample.get('content') or sample.get('code') or sample.get('question') or sample.get('source'))
                if not txt: continue 
                batch_texts.extend(tokenizer.encode(txt).ids)
            
            # Check if batch is valid/full
            if len(batch_texts) < (BATCH_SIZE * (BLOCK_SIZE + 1)):
                continue 
                
            data = torch.tensor(batch_texts[:(BATCH_SIZE*(BLOCK_SIZE+1))], dtype=torch.long)
            data = data.view(BATCH_SIZE, BLOCK_SIZE+1).to(DEVICE)
            x, y = data[:, :-1], data[:, 1:]
            
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=DEVICE, dtype=dtype):
                logits, loss = model(x, y)
                loss = loss / GRAD_ACCUM
            
            scaler.scale(loss).backward()
            loss_accum += loss.item()
            step_executed = True
            
        except StopIteration:
            print("[STOP] Dataset exhausted.")
            break

    if step_executed:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        final_loss = loss_accum * GRAD_ACCUM
        ppl = math.exp(final_loss) if final_loss < 20 else 1e9
        history['loss'].append(final_loss)
        history['ppl'].append(ppl)
        
        if step % 50 == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"Step {step:05d} | Loss: {final_loss:.4f} | PPL: {ppl:.1f} | LR: {lr:.2e}")
        
        if step > 0 and step % 1000 == 0:
            # 🛡️ AUTO-SAVE TOKENIZER (Fixes the Github issue)
            tokenizer.save(f"{CHECKPOINT_DIR}/tokenizer.json")
            
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history
            }, RESUME_FILE)
            print(f"💾 Checkpoint & Tokenizer Saved")

print("✅ Training Complete.")