import os
import numpy as np
from datasets import load_dataset, interleave_datasets
from tokenizers import Tokenizer 

# --- CONFIGURATION ---
TOTAL_SAMPLES = 50000    # How many tokens to pack?
PROBABILITIES = [0.5, 0.3, 0.2]  # 50% Text, 30% Code, 20% Math

def prepare_mixture():
    print("🚀 Starting Data Preparation...")

    # 1. LOAD TOKENIZER
    try:
        enc = Tokenizer.from_file("tokenizer_phase3_new.json")
        print("✅ Custom Tokenizer loaded.")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return
        
    eos_id = enc.token_to_id("<|endoftext|>") 
    if eos_id is None: eos_id = 0

    # 2. CONNECT TO DATA STREAMS (Switched to Modern Parquet Datasets)
    print("🌊 Connecting to data streams...")
    
    # A. English (FineWeb-Edu) - Works perfectly
    print("   - Loading Text (FineWeb)...")
    ds_text = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    # B. Code (The Stack Smol) - REPLACES BROKEN CODEPARROT
    # 'data/python' targets just python code. This dataset is modern and won't crash.
    print("   - Loading Code (The Stack Smol)...")
    ds_code = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)

    # C. Math (Orca Math) - Works perfectly
    print("   - Loading Math (Orca)...")
    ds_math = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True)

    # 3. MIX THEM TOGETHER
    print(f"🌪️ Mixing Data: {PROBABILITIES[0]*100}% Web, {PROBABILITIES[1]*100}% Code, {PROBABILITIES[2]*100}% Math")
    mixed_dataset = interleave_datasets(
        [ds_text, ds_code, ds_math],
        probabilities=PROBABILITIES,
        seed=42
    )

    all_ids = []
    count = 0

    # 4. PROCESS THE STREAM
    print("⏳ Packing tokens (this might take a minute)...")
    for sample in mixed_dataset:
        text = ""
        
        # --- ROBUST COLUMN MAPPING ---
        # 1. FineWeb uses 'text'
        if 'text' in sample and sample['text']:
            text = sample['text']
        
        # 2. The Stack (Code) uses 'content'
        elif 'content' in sample and sample['content']:
            text = sample['content']
            
        # 3. Orca (Math) uses 'question' + 'answer'
        elif 'question' in sample and 'answer' in sample:
            text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
        
        # Skip if we couldn't find text
        if not text: continue

        # Encode and Pack
        ids = enc.encode(text).ids
        all_ids.extend(ids)
        all_ids.append(eos_id) # Separator
        
        count += 1
        if count % 1000 == 0:
            print(f"   Packed {count}/{TOTAL_SAMPLES} documents...", end='\r')
            
        if count >= TOTAL_SAMPLES:
            break

    # 5. SAVE TO BINARY
    print(f"\n📦 Saving mixed recipe to train.bin...")
    if len(all_ids) > 0:
        # Use uint16 if vocab is small (<65k), otherwise uint32
        dtype = np.uint16 if max(all_ids) < 65535 else np.uint32
        ids_array = np.array(all_ids, dtype=dtype)
        ids_array.tofile("train.bin")
        print(f"✅ Success! Saved 'train.bin' with {len(all_ids)} tokens.")
        print("---------------------------------------------------")
        print("🎉 NEXT STEP: Run 'python train.py' to start training!")
        print("---------------------------------------------------")
    else:
        print("⚠️ Something went wrong. No data found.")

if __name__ == "__main__":
    prepare_mixture()