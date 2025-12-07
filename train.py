# FILE: train.py
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
import time
import os
import json
from model import BabyGPT, ModelConfig

# ==========================================
# EXPERIMENT CONTROL
# Change this to "swiglu" or "deep_res_mlp"
SELECTED_FFN_TYPE = "gated_deep_mlp"
# ==========================================

# HYPERPARAMETERS
BATCH_SIZE = 32
BLOCK_SIZE = 256
MAX_ITERS = 5000 
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"[INFO] Running on {DEVICE} | Mode: {SELECTED_FFN_TYPE.upper()}")

# 1. LOAD TOKENIZER
if not os.path.exists("tokenizer_phase1.json"):
    print("[ERROR] tokenizer_phase1.json not found. Run train_tokenizer.py first.")
    exit()

tokenizer = Tokenizer.from_file("tokenizer_phase1.json")
VOCAB_SIZE = tokenizer.get_vocab_size()
print(f"[INFO] Vocab Size: {VOCAB_SIZE}")

# 2. STREAM DATASET
print("[INFO] Streaming Dataset...")
ds_stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

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
config = ModelConfig(ffn_type=SELECTED_FFN_TYPE)
config.vocab_size = VOCAB_SIZE
config.block_size = BLOCK_SIZE

model = BabyGPT(config).to(DEVICE)
params = model.get_num_params() / 1e6
print(f"[INFO] Model Size: {params:.2f} Million Parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 4. TRAINING LOOP
print("\n[INFO] Starting Training Loop...")
start_time = time.time()
history = {"loss": [], "accuracy": []} 

model.train()
for step in range(MAX_ITERS):
    x, y = next(train_gen)
    
    logits, loss = model(x, y)
    
    # Calculate Accuracy (Flatten logic applied)
    preds = torch.argmax(logits, dim=-1)
    y_flat = y.view(-1)
    
    correct = (preds == y_flat).float().sum()
    accuracy = correct / y_flat.numel()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    history["loss"].append(loss.item())
    history["accuracy"].append(accuracy.item())
    
    if step % 100 == 0:
        avg_loss = sum(history["loss"][-100:]) / 100 if step > 0 else history["loss"][0]
        avg_acc = sum(history["accuracy"][-100:]) / 100 if step > 0 else history["accuracy"][0]
        dt = time.time() - start_time
        tok_sec = (step + 1) * BATCH_SIZE * BLOCK_SIZE / dt
        print(f"Step {step:04d} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2%} | Speed: {tok_sec:.0f} tok/s")

# 5. SAVE & RESULT
total_time = time.time() - start_time
final_loss = sum(history["loss"][-50:]) / 50
final_acc = sum(history["accuracy"][-50:]) / 50

print(f"\n[RESULT] Mode: {SELECTED_FFN_TYPE.upper()}")
print(f"Total Time: {total_time:.2f}s")
print(f"Final Loss: {final_loss:.4f}")
print(f"Final Acc:  {final_acc:.2%}")

# Save Stats
json_filename = f"history_{SELECTED_FFN_TYPE}.json"
with open(json_filename, "w") as f:
    json.dump(history, f)
print(f"[SUCCESS] Stats saved to {json_filename}")

save_path = f"babygpt_{SELECTED_FFN_TYPE}.pth"
torch.save(model.state_dict(), save_path)
print(f"[SUCCESS] Model saved to {save_path}")