import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # We need 3 Linear layers for Query, Key, and Value
        # In practice, we often fuse them, but let's keep them separate for clarity
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # Final unification layer
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # N = Batch Size
        N = query.shape[0]
        
        # Length of the sequence (e.g., sentence length)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        # Example: Embed_size=256, Heads=8 -> Each head gets 32 dimensions
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        # Get the Q, K, V vectors
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # --- THE MATH PART (Attention Formula) ---
        
        # 1. MatMul Q and K (Transpose K to match dimensions)
        # outcome: "energy" scores (How much Q matches K)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # n: batch, h: heads, q: query_len, k: key_len, d: head_dim

        # 2. Scale (Divide by square root of dimension) to keep gradients stable
        # This is the "Scaled" in "Scaled Dot-Product Attention"
        # If we don't do this, the softmax gets stuck.
        energy = energy / (self.embed_size ** (1 / 2))

        # 3. Softmax (Normalize scores to 0-1)
        # dim=3 means we normalize across the Key dimension
        attention = torch.softmax(energy, dim=3)

        # 4. MatMul with V (Apply attention scores to Values)
        out = torch.einsum("nhqk,nvhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # Final linear pass
        out = self.fc_out(out)
        return out

# --- TESTING IT ---
# Fake data: Batch=1, Sequence Length=5 words, Embedding Size=256
x = torch.randn(1, 5, 256) 

# Initialize Attention with 8 "heads" (8 parallel attention perspectives)
attention_layer = SelfAttention(embed_size=256, heads=8)

# In Self-Attention, Q, K, and V all come from the same source (x)
output = attention_layer(x, x, x, mask=None)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}") 
# Output shape matches input shape, but now contains "context-aware" data.