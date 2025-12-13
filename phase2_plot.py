import torch
import matplotlib.pyplot as plt
import os
import numpy as np

# CONFIG
CHECKPOINT_FILE = "checkpoint_latest.pth"
JSON_FILE = "history_phase2_sequential.json"

def load_history():
    # STRATEGY 1: Try loading from the live checkpoint (Most up-to-date)
    if os.path.exists(CHECKPOINT_FILE):
        try:
            print(f"🔍 Found checkpoint: {CHECKPOINT_FILE}")
            # Map location ensures we can plot even on a CPU-only laptop
            checkpoint = torch.load(CHECKPOINT_FILE, map_location='cpu')
            if 'history' in checkpoint:
                print(f"✅ Loaded live history from Step {checkpoint['step']}")
                return checkpoint['history']
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")

    # STRATEGY 2: Fallback to the JSON file (If training finished)
    if os.path.exists(JSON_FILE):
        import json
        print(f"📂 Found JSON log: {JSON_FILE}")
        with open(JSON_FILE, "r") as f:
            return json.load(f)
            
    return None

def smooth(data, window=50): 
    if not data or len(data) < window: return data
    # Simple moving average
    return np.convolve(data, np.ones(window)/window, mode='valid')

# 1. Load Data
history = load_history()

if history:
    steps = range(len(history['loss']))
    
    # 2. Setup Canvas (3 Rows now: Loss, Accuracy, Perplexity)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"🚜 The Tank (Sequential): {len(steps)} Steps", fontsize=16)

    # --- PLOT 1: LOSS ---
    ax1.set_title("Training Loss (Lower is Better)")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.plot(history["loss"], color="red", alpha=0.2, label="Raw")
    ax1.plot(smooth(history["loss"]), color="darkred", linewidth=2, label="Smooth")
    ax1.legend()

    # --- PLOT 2: PERPLEXITY (New!) ---
    # Only plot if available
    if "perplexity" in history and len(history["perplexity"]) > 0:
        ax2.set_title("Perplexity (Confusion Level)")
        ax2.set_ylabel("PPL")
        ax2.grid(True, alpha=0.3)
        # Clip huge values for graph readability
        ppl_clean = [min(x, 100) for x in history["perplexity"]]
        ax2.plot(ppl_clean, color="orange", alpha=0.2)
        ax2.plot(smooth(ppl_clean), color="darkorange", linewidth=2)
        # Add a target line
        ax2.axhline(y=25, color='green', linestyle='--', label="Target (25.0)")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Perplexity Data Not Found", ha='center')

    # --- PLOT 3: ACCURACY ---
    ax3.set_title("Accuracy (Higher is Better)")
    ax3.set_xlabel("Training Steps")
    ax3.set_ylabel("Accuracy")
    ax3.grid(True, alpha=0.3)
    ax3.plot(history["accuracy"], color="blue", alpha=0.2)
    ax3.plot(smooth(history["accuracy"]), color="darkblue", linewidth=2)

    plt.tight_layout()
    output_file = "phase2_live_status.png"
    plt.savefig(output_file)
    print(f"✅ Graph saved to {output_file}")
    plt.show()
else:
    print("❌ No history data found. Make sure 'checkpoint_latest.pth' exists.")