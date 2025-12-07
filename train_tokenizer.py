# FILE: train_tokenizer.py
from datasets import load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers, Tokenizer
import os

# --- CONFIGURATION ---
VOCAB_SIZE = 4096
BATCH_SIZE = 1000
dataset_name = "HuggingFaceFW/fineweb-edu"
subset = "sample-10BT"

print(f"[INFO] Loading {dataset_name} (Streaming Mode)...")
# Stream the dataset so we don't download 1TB
dataset = load_dataset(dataset_name, name=subset, split="train", streaming=True)

def batch_iterator(batch_size=1000):
    batch = []
    counter = 0
    limit = 50000  # Scan 50k documents to learn the vocab
    
    for item in dataset:
        batch.append(item["text"])
        if len(batch) == batch_size:
            yield batch
            batch = []
        counter += 1
        if counter >= limit:
            break
    if batch:
        yield batch

print("[INFO] Training Tokenizer...")
# Initialize BPE Tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Trainer
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<|endoftext|>", "<|padding|>"],
    show_progress=True
)

# Train
tokenizer.train_from_iterator(batch_iterator(BATCH_SIZE), trainer=trainer)

# Configure decoder for correct playback
tokenizer.decoder = decoders.ByteLevel()

# Save
save_path = "tokenizer_phase1.json"
tokenizer.save(save_path)
print(f"[SUCCESS] Tokenizer saved to: {save_path} (Vocab: {VOCAB_SIZE})")