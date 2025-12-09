# FILE: generate.py
import torch
import torch.nn.functional as F
from model import BabyGPT, Phase2Config
from tokenizers import Tokenizer
import os

# --- ⚙️ CONFIGURATION ---
MODEL_PATH = "checkpoint_latest.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. LOAD TOKENIZER
if not os.path.exists("tokenizer_phase1.json"):
    print("❌ Error: tokenizer_phase1.json not found.")
    exit()
tokenizer = Tokenizer.from_file("tokenizer_phase1.json")

# 2. INITIALIZE MODEL (50M TANK CONFIG)
print(f"[INFO] Initializing Phase 2 Model...")
config = Phase2Config(ffn_type="gated_deep_mlp")
config.vocab_size = tokenizer.get_vocab_size()

model = BabyGPT(config).to(DEVICE)

# 3. LOAD WEIGHTS
print(f"[INFO] Loading weights from {MODEL_PATH}...")
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('_orig_mod.', '').replace('module.', '')
        clean_state_dict[new_k] = v
        
    model.load_state_dict(clean_state_dict)
    print("✅ Weights loaded.")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

model.eval()

# 4. GENERATION FUNCTION (FINAL BUG FIX 🐛)
def generate_text(prompt, max_tokens=150, temperature=0.8, top_k=50, repetition_penalty=1.2):
    
    encoded = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
    
    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)
    
    for _ in range(max_tokens):
        idx_cond = input_tensor[:, -config.block_size:]
        
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        logits = logits[:, -1, :] 
        
        # --- 🛡️ FINAL CORRECTED REPETITION PENALTY ---
        # Apply penalty ONLY to previously used tokens
        # Handle positive/negative scores correctly
        for token_id in set(input_tensor[0].tolist()):
            score = logits[0, token_id]
            if score < 0:
                logits[0, token_id] = score * repetition_penalty
            else:
                logits[0, token_id] = score / repetition_penalty
        # ---------------------------------------------

        logits = logits / temperature
        
        # Top-K Sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        input_tensor = torch.cat((input_tensor, idx_next), dim=1)

    return tokenizer.decode(input_tensor[0].tolist())

# 5. UI
print("\n--- 🤖 BABY GPT (PHASE 2) ---")
print("Settings: Temp=0.8, RepetitionPenalty=1.2")
while True:
    try:
        user_input = input("\nType a prompt (or 'q'): ")
        if user_input.lower() in ['q', 'exit']: break
        
        # Use the default settings defined in the function header
        generated = generate_text(user_input) 
        print("\n\n--- RESULT ---")
        print(generated)
        print("----------------")
        
    except KeyboardInterrupt:
        break