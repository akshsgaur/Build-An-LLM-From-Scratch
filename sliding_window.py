
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt", "r", encoding='utf-8') as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]
print(enc_sample)

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:       {y}")

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)


for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,stride=128, shuffle=True, drop_last=True):
    tokenizor = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt,tokenizor,max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,drop_last=drop_last)
    return dataloader

dataloader = create_dataloader_v1(raw_text, batch_size = 8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)

second_batch = next(data_iter)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


