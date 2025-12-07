# FILE: plot.py
import json
import matplotlib.pyplot as plt
import os

# CONFIG: Filenames must match those saved by train.py
FILE_SWIGLU = "history_swiglu.json"
FILE_DEEP   = "history_deep_res_mlp.json"

def load_data(filename):
    if not os.path.exists(filename):
        print(f"[WARN] {filename} not found.")
        return None
    with open(filename, "r") as f:
        return json.load(f)

# 1. Load Data
data_swiglu = load_data(FILE_SWIGLU)
data_deep   = load_data(FILE_DEEP)

def smooth(data, window=50):
    if not data or len(data) < window: return data
    return [sum(data[i:i+window])/window for i in range(len(data)-window)]

# 2. Setup Canvas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Phase 1: SwiGLU vs DeepResMLP Comparison", fontsize=16)

# --- PLOT 1: LOSS ---
ax1.set_title("Training Loss (Lower is Better)")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle='--', alpha=0.5)

if data_swiglu:
    ax1.plot(smooth(data_swiglu["loss"]), label="SwiGLU", color="blue", linewidth=2)
if data_deep:
    ax1.plot(smooth(data_deep["loss"]), label="DeepResMLP", color="red", linewidth=2)
ax1.legend()

# --- PLOT 2: ACCURACY ---
ax2.set_title("Training Accuracy (Higher is Better)")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Accuracy")
ax2.grid(True, linestyle='--', alpha=0.5)

if data_swiglu:
    ax2.plot(smooth(data_swiglu["accuracy"]), label="SwiGLU", color="blue", linewidth=2)
if data_deep:
    ax2.plot(smooth(data_deep["accuracy"]), label="DeepResMLP", color="red", linewidth=2)
ax2.legend()

plt.tight_layout()
plt.savefig("comparison_v2.png")
print("[SUCCESS] Graph saved as comparison_v2.png")
plt.show()