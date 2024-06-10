import numpy as np
import re
with open("the_verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()

# #converting raw_tex to tokens

# split_text = re.split(r'([.,?_\'()"!]|--|\s)',raw_text)
# split_text = [item.strip() for item in split_text if item.strip()]

# # converting tokens into token-ID

# words = sorted(list(set(split_text)),reverse=False)
# # creating a vocab (dtype = dict)
# vocab = {token:i for i,token in enumerate(words)}

# #initializing tokenizer class

# class Tokenizer_v1:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i:s for s,i in vocab.items()} 
#     def encode(self,text):
#         preprocessed = re.split(r'([.,?!"\'()_]|--|\s)',text)
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
#         ids = [self.str_to_int[word] for word in preprocessed]
#         return ids
#     def decode(self,ids):
#         text = " ".join([self.int_to_str[id] for id in ids])
#         output = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
#         return output

# words.extend(["<|unk|>","<|endoftext|>"])
# vocab = {token:i for i,token in enumerate(words)}

# class Tokenizer_v2:
#     def __init__(self,vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {i:s for s,i in (vocab.items())}
#     def encode(self,text):
#         preprocessed = re.split(r'([.,?!"\'()_]|--|\s)',text)
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
#         preprocessed = [token if token in self.str_to_int
#                         else "<|unk|>" for token in preprocessed]
#         ids = [self.str_to_int[word] for word in preprocessed]
#         return ids
#     def decode(self,ids):
#         text = " ".join([self.int_to_str[id] for id in ids])
#         output = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
#         return output


# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))
# # tokenizer = Tokenizer_v2(vocab)
# # ids = tokenizer.encode(text)
# # text_n = tokenizer.decode(ids)
# # print(ids,text_n)

import tiktoken
from importlib.metadata import version

tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(text=raw_text)
print(len(enc_text))

enc_smaple = enc_text[50:]

context_len = 4

x = enc_smaple[:context_len]
y = enc_smaple[1:context_len+1]

for i in range(1,context_len+1):
    inp = tokenizer.decode(x[:i])
    guess =  tokenizer.decode([y[i-1]])
    print(inp,"----->",guess)
