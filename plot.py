import json
import matplotlib.pyplot as plt
import os

# CONFIG: The 3 Fighters
FILE_SWIGLU = "history_swiglu.json"         # The Champion (1.6M)
FILE_DEEP   = "history_deep_res_mlp.json"   # The Failed Challenger (2.4M)
FILE_GATED  = "history_gated_deep_mlp.json" # The Heavyweight Tank (5.55M)

def load_data(filename):
    if not os.path.exists(filename):
        print(f"Missing: {filename}")
        return None
    with open(filename, "r") as f:
        return json.load(f)

# 1. Load Data
data_swiglu = load_data(FILE_SWIGLU)
data_deep   = load_data(FILE_DEEP)
data_gated  = load_data(FILE_GATED)

def smooth(data, window=50):
    if not data or len(data) < window: return data
    return [sum(data[i:i+window])/window for i in range(len(data)-window)]

# 2. Setup Canvas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("The Final Showdown: Small vs Deep vs Gated", fontsize=16)

# --- PLOT 1: LOSS (Intelligence) ---
ax1.set_title("Loss (Lower is Better)")
ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle='--', alpha=0.5)

if data_swiglu:
    ax1.plot(smooth(data_swiglu["loss"]), label="SwiGLU (1.6M)", color="blue", linewidth=2)
if data_deep:
    ax1.plot(smooth(data_deep["loss"]), label="DeepRes (2.4M)", color="red", linewidth=1, alpha=0.7)
if data_gated:
    ax1.plot(smooth(data_gated["loss"]), label="GatedDeep (5.5M)", color="green", linewidth=2.5) # Thicker line
ax1.legend()

# --- PLOT 2: ACCURACY (Precision) ---
ax2.set_title("Accuracy (Higher is Better)")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Accuracy")
ax2.grid(True, linestyle='--', alpha=0.5)

if data_swiglu:
    ax2.plot(smooth(data_swiglu["accuracy"]), label="SwiGLU", color="blue", linewidth=2)
if data_deep:
    ax2.plot(smooth(data_deep["accuracy"]), label="DeepRes", color="red", linewidth=1, alpha=0.7)
if data_gated:
    ax2.plot(smooth(data_gated["accuracy"]), label="GatedDeep", color="green", linewidth=2.5)
ax2.legend()

plt.tight_layout()
plt.savefig("comparison_final.png")
print("[SUCCESS] Comparison graph saved to comparison_final.png")
plt.show()