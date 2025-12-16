# FILE: generate_phase3.py
import torch
import torch.nn.functional as F
from model_phase3 import Phase3Config, StudentGPT
from tokenizers import Tokenizer
import os

MODEL_PATH = "checkpoints_phase3/latest.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the 4096 tokenizer
tokenizer = Tokenizer.from_file("tokenizer_phase1.json")
config = Phase3Config(vocab_size=tokenizer.get_vocab_size())
model = StudentGPT(config).to(DEVICE)

print(f"Loading {MODEL_PATH}...")
try:
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
except Exception as e:
    print(f"⚠️ Could not load checkpoint: {e}")
    print("Starting with random weights (for testing only).")

model.eval()

def generate(prompt, max_new_tokens=100):
    encoded = tokenizer.encode(prompt).ids
    idx = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
    
    for _ in range(max_new_tokens):
        # Crop to block_size if needed
        idx_cond = idx[:, -config.block_size:]
        
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        logits = logits[:, -1, :] / 0.8 # Temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
    return tokenizer.decode(idx[0].tolist())

while True:
    p = input("\nPrompt: ")
    print(generate(p))