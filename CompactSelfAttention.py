import torch.nn as nn 
import torch
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89], # Your
        [0.55, 0.87, 0.66], # Journey
        [0.57, 0.85, 0.64], # Starts
        [0.22, 0.58, 0.33], # with
        [0.77, 0.25, 0.10], # one
        [0.05, 0.80, 0.55]  # step
    ]

)
class SelfAttention_v1(nn.Module): 
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out # Dimension of the output.
        self.W_query = nn.Parameter(torch.rand(d_in, d_out)) # Weight matrix for the query.
        self.W_key = nn.Parameter(torch.rand(d_in, d_out)) # Weight matrix for the key.
        self.W_value = nn.Parameter(torch.rand(d_in, d_out)) # Weight matrix for the value.

    def forward(self, x):
        keys = x @ self.W_key # Keys.
        values = x @ self.W_value # Values.
        queries = x @ self.W_query # Queries.
        attn_scores = queries @ keys.T # Attention scores.
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # Attention weights.
        context_vector = attn_weights @ values
        return context_vector


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in=3, d_out=2)
print(sa_v1(inputs))
    
