import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. DATA PREP (Same as before) ---
text = "hello"
chars = ['h', 'e', 'l', 'o']
char_to_ix = {char: i for i, char in enumerate(chars)}
ix_to_char = {i: char for i, char in enumerate(chars)}

input_seq = [char_to_ix[c] for c in text[:-1]] 
target_seq = [char_to_ix[c] for c in text[1:]]

def one_hot_encode(sequence, vocab_size=len(chars)):
    features = np.zeros((len(sequence), 1, vocab_size), dtype=np.float32)
    for i, idx in enumerate(sequence):
        features[i, 0, idx] = 1.0
    return torch.from_numpy(features)

inputs = one_hot_encode(input_seq)     
targets = torch.tensor(target_seq)     

# --- 2. THE LSTM MODEL ---
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size

        # ### CHANGE 1: Use LSTM instead of RNN
        # LSTM needs to know input size and hidden memory size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=False)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # ### CHANGE 2: LSTM Inputs/Outputs
        # In simple RNN, 'hidden' was just one tensor (h).
        # In LSTM, 'hidden' is a TUPLE of two tensors: (h, c)
        # h = hidden state (short term), c = cell state (long term)
        
        out, hidden = self.lstm(x, hidden)
        
        out = out.view(-1, self.hidden_size)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self):
        # ### CHANGE 3: Initialize TWO states
        # We return a tuple: (h_0, c_0)
        # Shape: (Num_Layers, Batch_Size, Hidden_Size)
        h0 = torch.zeros(1, 1, self.hidden_size)
        c0 = torch.zeros(1, 1, self.hidden_size)
        return (h0, c0)

# Configuration
input_size = 4
hidden_size = 10 
output_size = 4 
model = SimpleLSTM(input_size, hidden_size, output_size)

# --- 3. LOSS & OPTIMIZER ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- 4. TRAINING LOOP ---
print("--- Training LSTM ---")
for epoch in range(100):
    model.zero_grad()
    
    # Initialize hidden AND cell states
    hidden = model.init_hidden()
    
    # Forward Pass
    # Note: 'hidden' variable here actually holds the tuple (h, c)
    output, hidden = model(inputs, hidden)
    
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}: Loss: {loss.item():.4f}')

# --- 5. TESTING ---
print("\n--- Testing LSTM ---")
with torch.no_grad():
    hidden = model.init_hidden()
    prediction, _ = model(inputs, hidden)
    
    _, predicted_indices = torch.max(prediction, 1)
    
    result_word = ""
    for idx in predicted_indices:
        result_word += ix_to_char[idx.item()]
        
    print(f"Input:  {text[:-1]}")
    print(f"Output: {result_word}")