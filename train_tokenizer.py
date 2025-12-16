# FILE: train_tokenizer_phase3.py
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from datasets import load_dataset, interleave_datasets
import os

# CONFIGURATION
VOCAB_SIZE = 4096  # Keeping your requested size

print(f"⏳ Initializing Tokenizer Training (Target: {VOCAB_SIZE})...")

# 1. LOAD DATA STREAMS (Fixed!)
print("   - Connecting to data streams...")

# A. English (FineWeb-Edu)
ds_text = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

# B. Code (The Stack Smol - requires login, or use CodeParrot if that fails)
# Note: Ensure you are logged in via 'huggingface-cli login' if using The Stack
try:
    ds_code = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
except Exception:
    print("   ⚠️ Auth error on Stack-Smol. Fallback to CodeParrot (Public)...")
    ds_code = load_dataset("codeparrot/github-code", split="train", streaming=True)

# C. Math (Microsoft Orca - REPLACES BROKEN PROOF-PILE)
# This uses standard Parquet files, so it won't crash.
ds_math = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True)

# 2. CONFIGURE TOKENIZER
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# 3. TRAINER
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<|endoftext|>", "<|padding|>"],
    min_frequency=2
)

# 4. ITERATOR (The Mix)
def get_training_corpus():
    # Mix: 50% Text, 30% Code, 20% Math
    dataset = interleave_datasets([ds_text, ds_code, ds_math], probabilities=[0.5, 0.3, 0.2])
    
    # Train on 100k samples
    for i, sample in enumerate(dataset):
        if i >= 100_000: break
        
        # Handle different column names
        txt = sample.get("text") or sample.get("content") or sample.get("question") or ""
        yield txt

# 5. RUN
print("🚀 Training Tokenizer... (This takes ~5 mins)")
tokenizer.train_from_iterator(get_training_corpus(), trainer)

# 6. SAVE
save_path = "tokenizer_phase3_new.json"
tokenizer.save(save_path)
print(f"✅ Done! Saved to: {save_path}")