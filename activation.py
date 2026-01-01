import numpy as np
import torch
import torch.nn as nn

# --- 1. DEFINE INPUT DATA ---
# We use a range from negative to positive to see the effects
data = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])

print(f"Raw Input: {data.numpy()}")

# --- 2. SIGMOID (0 to 1) ---
sigmoid = nn.Sigmoid()
print(f"\nSigmoid:   {sigmoid(data).numpy().round(2)}")
# Note: -10 becomes 0.0, 10 becomes 1.0. It squashes everything.

# --- 3. TANH (-1 to 1) ---
tanh = nn.Tanh()
print(f"Tanh:      {tanh(data).numpy().round(2)}")
# Note: -10 becomes -1.0. It allows negative outputs.

# --- 4. RELU (0 to Infinity) ---
relu = nn.ReLU()
print(f"ReLU:      {relu(data).numpy().round(2)}")
# Note: Negatives become 0. Positives stay exactly the same.

# --- 5. SOFTMAX (Probabilities) ---
# Let's use a different input representing "scores" for 3 classes
logits = torch.tensor([2.0, 1.0, 0.1])
softmax = nn.Softmax(dim=0)
probs = softmax(logits)

print(f"\n--- Softmax Example ---")
print(f"Raw Scores:   {logits.numpy()}")
print(f"Probabilities:{probs.numpy().round(2)}")
print(f"Sum of Probs: {probs.sum().item()}") # Always adds up to 1.0