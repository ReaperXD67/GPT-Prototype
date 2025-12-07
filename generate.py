import torch
import torch.nn.functional as F
from model import BabyGPT, ModelConfig
from tokenizers import Tokenizer
import os

# --- CONFIGURATION ---
# Which model do you want to talk to?
MODEL_PATH = "babygpt_gated_deep_mlp.pth" 
# MODEL_PATH = "babygpt_gated_deep_mlp.pth" 

# Must match exactly what you used in train.py (Phase 1)
TYPE = "swiglu" if "swiglu" in MODEL_PATH else "gated_deep_mlp"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Load Tokenizer
if not os.path.exists("tokenizer_phase1.json"):
    print("Error: Tokenizer not found.")
    exit()
tokenizer = Tokenizer.from_file("tokenizer_phase1.json")

# 2. Initialize Model (Empty Brain)
print(f"Loading {TYPE} model from {MODEL_PATH}...")
config = ModelConfig(ffn_type=TYPE)

# --- CRITICAL: MATCH PHASE 1 PARAMS ---
config.vocab_size = 4096
config.d_model = 128
config.n_layer = 4
config.n_head = 4
config.block_size = 256
# --------------------------------------

model = BabyGPT(config).to(DEVICE)

# 3. Load Trained Weights (The Knowledge)
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("Weights loaded successfully!")
except FileNotFoundError:
    print(f"Error: Could not find {MODEL_PATH}. Did you train it?")
    exit()
except Exception as e:
    print(f"Error loading weights: {e}")
    exit()

model.eval() # Switch to "Testing Mode"

# 4. Generation Loop
def generate_text(prompt, max_tokens=100, temperature=1.0):
    # Encode prompt
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)
    
    # Loop to generate tokens one by one
    for _ in range(max_tokens):
        # Crop context if it gets too long
        idx_cond = input_tensor[:, -config.block_size:]
        
        # Get predictions
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        # Focus on the last step
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        
        # Sample (Pick the next word)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        input_tensor = torch.cat((input_tensor, idx_next), dim=1)
        
        # Live Stream Print (Optional)
        # new_word = tokenizer.decode([idx_next.item()])
        # print(new_word, end="", flush=True)

    # Decode final result
    output_text = tokenizer.decode(input_tensor[0].tolist())
    return output_text

# 5. Run it
while True:
    user_input = input("\n\nType a prompt (or 'q' to quit): ")
    if user_input.lower() == 'q': break
    
    generated = generate_text(user_input, max_tokens=100, temperature=0.8)
    print("\n--- RESULT ---")
    print(generated)