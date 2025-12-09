import json
import matplotlib.pyplot as plt
import os

# CONFIG
FILE_PHASE2 = "history_phase2.json"

def load_data(filename):
    if not os.path.exists(filename):
        print(f"⚠️ Warning: {filename} not found.")
        return None
    with open(filename, "r") as f:
        return json.load(f)

# 1. Load Data
data = load_data(FILE_PHASE2)

def smooth(data, window=100): # Higher smoothing for long runs
    if not data or len(data) < window: return data
    return [sum(data[i:i+window])/window for i in range(len(data)-window)]

if data:
    # 2. Setup Canvas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"🚀 Phase 2: The 50M 'Tank' Run ({len(data['loss'])} Steps)", fontsize=16)

    # --- PLOT 1: LOSS ---
    ax1.set_title("Training Loss (Intelligence)")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot raw faint line + smooth strong line
    ax1.plot(data["loss"], color="lightgreen", alpha=0.3)
    ax1.plot(smooth(data["loss"]), label="GatedDeepMLP (50M)", color="darkgreen", linewidth=2)
    ax1.legend()

    # --- PLOT 2: ACCURACY ---
    ax2.set_title("Training Accuracy (Precision)")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax2.plot(data["accuracy"], color="lightgreen", alpha=0.3)
    ax2.plot(smooth(data["accuracy"]), label="Accuracy", color="darkgreen", linewidth=2)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("phase2_results.png")
    print("✅ Graph saved to phase2_results.png")
    plt.show()
else:
    print("❌ No data found. Did Phase 2 finish saving?")