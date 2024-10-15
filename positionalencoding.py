import torch
from sliding_window import create_dataloader_v1, raw_text
output_dim = 256
vocab_size = 50257
# torch.nn.Embedding take vocab_size and output_dim
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#print(token_embedding_layer)

max_length = 4
print("raw_text: ", raw_text)
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride= max_length)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
#print("Token IDs: \n", inputs)
#print("\n Inputs shape: \n", inputs.shape)



'''
tensor([[   12, 12239,    13,   198],
        [  340,   546, 12622, 41379],
        [ 1675,   262,  6846,    11],
        [  276,  5118,    11,   550],          <--- 8 text sample with 4 tokens each. 
        [  616, 15185,  2900,   656],
        [  338, 19992, 31564,   286],
        [ 1022,   514,  2474,   198],
        [   13,   198,   198,     1]])

'''

# Using the embedding layer to embed these Tokens into 256 dimensional vectors:

token_embedding = token_embedding_layer(inputs)
#print(token_embedding.shape)

#GPT's model absolute embedding approach: we nedd to create another embedding layer that has the same dimension as the token_embedding_layer:

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embedding = pos_embedding_layer(torch.arange(context_length)) #A placeholder vector.
#print(pos_embedding.shape)

input_embedding = token_embedding + pos_embedding
#print(torch.Size([8,4,256]))



