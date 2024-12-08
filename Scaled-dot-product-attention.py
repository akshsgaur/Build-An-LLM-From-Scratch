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

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2



torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # <- This is the weight matrix for the query.
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) # <- This is the weight matrix for the key.
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query # <- This is the query for the second input.
key_2 = x_2 @ W_key # <- This is the key for the second input.
value_2 = x_2 @ W_value # <- This is the value for the second input.

print("query_2: ", query_2)

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape: ", keys.shape)
print("values.shape: ", values.shape)

keys_2 = keys[1]
attn_scores_22 = query_2.dot(keys_2)
print("attn_scores_22: ", attn_scores_22)

attn_scores_2 = query_2 @ keys.T #All attention scores for the second input.
print("attn_scores_2: ", attn_scores_2)

d_k = keys.shape[-1] # Dimension of the key.
attn_weights_2 = torch.softmax(attn_scores_2/d_k**0.5, dim=-1)
print("attn_weights_2: ", attn_weights_2)

context_vec_2 = attn_weights_2 @ values # Context vector for the second input.
print("context_vec_2: ", context_vec_2)



