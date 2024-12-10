import torch.nn as nn 
import torch
from CompactSelfAttentionWithLinear import inputs

class SelfAttention_v1(nn.Module): 
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out # Dimension of the output.
        self.W_query = nn.Parameter(torch.rand(d_in, d_out)) # Weight matrix for the query.
        self.W_key = nn.Parameter(torch.rand(d_in, d_out)) # Weight matrix for the key.
        self.W_value = nn.Parameter(torch.rand(d_in, d_out)) # Weight matrix for the value.
        # print(self.W_query.weight)
        # print(self.W_key.weight)
        # print(self.W_value.weight)

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
#print(sa_v1(inputs))


class SelfAttention_v2(nn.Module): 
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out 
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # print(self.W_key.weight)
        # print(self.W_query.weight)
        # print(self.W_value.weight)

    def forward(self, x): 
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vector = attn_weights @ values 
        return context_vector.T


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in=3, d_out=2)

sa_v1.W_query.data = sa_v2.W_query.weight.data.T
sa_v1.W_key.data = sa_v2.W_key.weight.data.T
sa_v1.W_value.data = sa_v2.W_value.weight.data.T

#print(sa_v1(inputs))
#print(sa_v2(inputs).T)






