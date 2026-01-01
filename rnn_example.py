import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. DATA PREPARATION ---
# The string we want to learn
text = "hello"
# Define the "vocabulary" (unique characters)
chars = ['h', 'e', 'l', 'o']

# Helper: Convert char to index (h -> 0) and index to char (0 -> h)
char_to_ix = {char: i for i, char in enumerate(chars)}
ix_to_char = {i: char for i, char in enumerate(chars)}

# Prepare Input (x) and Target (y)
# Input: "h", "e", "l", "l"
# Target: "e", "l", "l", "o"
input_seq = [char_to_ix[c] for c in text[:-1]] 
target_seq = [char_to_ix[c] for c in text[1:]]

# Convert to One-Hot Encoding (RNNs need numbers, not strings)
# Shape: [Sequence_Length, Batch_Size, Input_Size]
# We have 1 sequence, length 4, and 4 possible characters (Input Size)
def one_hot_encode(sequence, vocab_size=len(chars)):
    features = np.zeros((len(sequence), 1, vocab_size), dtype=np.float32)
    for i, idx in enumerate(sequence):
        features[i, 0, idx] = 1.0
    return torch.from_numpy(features)

inputs = one_hot_encode(input_seq)     # Shape: [4, 1, 4]
targets = torch.tensor(target_seq)     # Shape: [4]

# --- 2. THE RNN MODEL ---
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # The RNN Layer
        # It takes input and updates its internal "hidden state"
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=False)
        
        # The Output Layer
        # Converts the "hidden state" into a prediction (probability of next char)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x shape: [Seq_Len, Batch, Input_Size]
        # hidden shape: [1, Batch, Hidden_Size]
        
        # Pass data through RNN
        # out: contains output features for every time step
        # hidden: contains the final memory state
        out, hidden = self.rnn(x, hidden)
        
        # Reshape output to pass to Linear Layer
        # We flatten the time steps to treat them as a big batch
        out = out.view(-1, self.hidden_size)
        
        # Get final predictions
        out = self.fc(out)
        return out, hidden

    def init_hidden(self):
        # Initialize the memory (Hidden State) to zeros
        return torch.zeros(1, 1, self.hidden_size)

# Configuration
input_size = 4   # 4 unique chars ('h', 'e', 'l', 'o')
hidden_size = 10 # Size of "brain memory"
output_size = 4  # 4 possible predictions
model = SimpleRNN(input_size, hidden_size, output_size)

# --- 3. LOSS & OPTIMIZER ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- 4. TRAINING LOOP ---
print("--- Training ---")
for epoch in range(100):
    model.zero_grad()
    
    # Initialize hidden state (Short-term memory is clear at start of sentence)
    hidden = model.init_hidden()
    
    # Forward Pass
    output, hidden = model(inputs, hidden)
    
    # Calculate Loss
    loss = criterion(output, targets)
    
    # Backprop
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch}: Loss: {loss.item():.4f}')

# --- 5. TEST THE MODEL ---
print("\n--- Testing ---")
with torch.no_grad():
    hidden = model.init_hidden()
    # Let's feed it the sequence "hell" and see if it outputs "ello"
    prediction, _ = model(inputs, hidden)
    
    # Convert probability scores to character indices
    _, predicted_indices = torch.max(prediction, 1)
    
    # Decode result
    result_word = ""
    for idx in predicted_indices:
        result_word += ix_to_char[idx.item()]
        
    print(f"Input:  {text[:-1]}")
    print(f"Output: {result_word}")