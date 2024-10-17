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

query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])
print("attention scores empty: ",attn_scores_2)

# The dot product of each element to determine the extent which elements in a sequence attend to each other.
for i, x_i in enumerate(inputs):
    print("i, x_i: ", i, x_i)
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)
# Higher the dot score, the higher the similarity and attention score between the two.

attn_weights_2_tmp = attn_scores_2/attn_scores_2.sum()
print("Attention weights: ", attn_weights_2_tmp)
print("Sum: ", attn_weights_2_tmp.sum())

# Softmax naive implementation

def softmax_naive(attn_scores):
    return torch.exp(attn_scores) / torch.exp(attn_scores).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attenion weights: ", attn_weights_2_naive)
print("Sum: ", attn_weights_2_naive.sum())

# Softmax torch implementation

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights: ", attn_scores_2)
print("Sum: ", attn_weights_2.sum())

context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)

#Attention scores using for-loops
attn_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i,j] = torch.dot(x_i, x_j)
print(attn_scores)

#Attention scores using matrix multiplication
attn_scores = inputs @ inputs.T
print(attn_scores)

attn_weights =  torch.softmax(attn_scores,dim=0)
print(attn_weights)

all_context_vecs = attn_weights @ inputs
print(all_context_vecs)



