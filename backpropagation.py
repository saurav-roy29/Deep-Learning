import numpy as np

# --- 1. SETUP ---
# Input (x) and Target (y)
x = 3.0 
y_target = 6.0 

# Initial "dumb" weight (Random guess)
w = 0.5  
learning_rate = 0.1

print(f"Start: Input={x}, Target={y_target}, Initial Weight={w:.2f}")

# --- 2. TRAINING LOOP ---
for epoch in range(10):
    # A. FORWARD PASS (The Guess)
    # Formula: prediction = x * weight
    y_pred = x * w
    
    # Calculate Loss (Error)
    # Formula: (prediction - target)^2
    loss = (y_pred - y_target) ** 2
    
    # B. BACKPROPAGATION (The Math/GPS)
    # We need to find the gradient: "How much does Loss change if w changes?"
    # Chain Rule derivation: 
    # d(Loss)/d(w) = 2 * (y_pred - y_target) * x
    gradient = 2 * (y_pred - y_target) * x
    
    # C. GRADIENT DESCENT (The Step)
    # Update rule: w_new = w_old - (learning_rate * gradient)
    w = w - (learning_rate * gradient)
    
    print(f"Epoch {epoch+1}: Pred={y_pred:.2f} | Loss={loss:.2f} | Gradient={gradient:.2f} | New Weight={w:.2f}")

print(f"\nFinal learned weight: {w:.2f} (Target was 2.0)")