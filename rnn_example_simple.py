import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. PREPARE THE DATA ---
# We want the model to learn: "Add 10 to the sequence"
# Input: A sequence of 4 numbers
X = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]]) # Shape: (1, 4, 1)

# Target: The next number in the sequence (50)
y = torch.tensor([[50.0]])                           # Shape: (1, 1)

print(f"Input Shape: {X.shape}  (Batch: 1, Length: 4, Features: 1)")
print(f"Target Shape: {y.shape}")

# --- 2. DEFINE THE MODEL ---
class SuperSimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input Size = 1 (We feed 1 number at a time)
        # Hidden Size = 10 (The "brain" has 10 neurons to remember context)
        # batch_first=True (Standard format: Batch, Seq, Feature)
        self.rnn = nn.RNN(input_size=1, hidden_size=10, batch_first=True)
        
        # Output Layer: Take the memory (10 numbers) and predict 1 final number
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        # 1. Pass data to RNN
        # out: The output at every single time step
        # hidden: The final memory state after the last step
        out, hidden = self.rnn(x)
        
        # 2. We only care about the LAST step's output (at the end of the sequence)
        # out shape is (Batch, Seq, Hidden) -> (1, 4, 10)
        # We grab the last time step: index -1
        last_step_output = out[:, -1, :] 
        
        # 3. Pass that last memory into the Linear layer to get prediction
        prediction = self.fc(last_step_output)
        return prediction

model = SuperSimpleRNN()

# --- 3. TRAIN ---
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss() # Mean Squared Error (Standard for number predictions)

print("\n--- Training ---")
for epoch in range(500):
    optimizer.zero_grad()
    
    # Forward pass
    output = model(X)
    
    # Calculate error (How far is output from 50.0?)
    loss = criterion(output, y)
    
    # Backprop
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Prediction = {output.item():.2f} | Loss = {loss.item():.4f}")

# --- 4. TEST WITH NEW DATA ---
print("\n--- Testing ---")
# Let's see if it learned the pattern "Add 10"
# Input: [100, 110, 120, 130] -> Expected: 140
test_input = torch.tensor([[[100.0], [110.0], [120.0], [130.0]]])

with torch.no_grad():
    prediction = model(test_input)
    print(f"Input Sequence: 100, 110, 120, 130")
    print(f"Model Predicts: {prediction.item():.2f}")
    print(f"Expected:       140.00")