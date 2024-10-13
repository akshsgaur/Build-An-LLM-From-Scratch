import re
from SimpleTokenizerV1 import SimpleTokenizerV1, SimpleTokenizerV2
import tiktoken
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    print("Total number of character", len(raw_text))
    print(raw_text[:99])
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if len(item) != 0]
    print(len(preprocessed))
    print(preprocessed[:30])
    all_words = sorted(list(set(preprocessed)))
    vocab_size = len(all_words)
    print(vocab_size)
    vocab = {token:integer for integer,token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i > 50:
            break

    tokenizer = SimpleTokenizerV1(vocab)
    text = """It's the last he painted, you know," Mrs. Gisburn said"""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))

    # text_2 = "Hello, do you like tea?"
    # tokenizer.encode(text_2)

    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>","<|unk|>"])
    vocab = {token:integer for integer, token in enumerate(all_tokens)}

    print(len(vocab.items()))

    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)


    text3 = "Hello do you like tea?"
    text4 = "In the sunlit terraces of the palace."
    text5 = "<|endoftext|> ".join((text3, text4))
    print(text5)

    tokenizer = SimpleTokenizerV2(vocab)
    print(tokenizer.encode(text5))

    print(tokenizer.decode(tokenizer.encode(text5)))
    tokenizer = tiktoken.get_encoding("gpt2")

    integers = tokenizer.encode(text5, allowed_special={'<|endoftext|>'})
    print(integers)
    strings = tokenizer.decode(integers)
    print(strings)
