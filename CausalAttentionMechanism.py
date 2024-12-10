from CompactSelfAttentionWithLinear import SelfAttention_v2
from CompactSelfAttentionWithLinear import inputs
import torch

# Attention Scores
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in=3, d_out=2)
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
#Attention weights 
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
#Masked attention scores 
masked_simple = attn_weights * mask_simple
#print(masked_simple)

row_sums = masked_simple.sum(dim=1, keepdim=True)
masked_simple_normalized = masked_simple / row_sums 
print(masked_simple_normalized)


mask = torch.triu(torch.ones(context_length, context_length))
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)