
# # basic self-attention mechanism without masking and only for batch size of 1

# class Self_attention_v1(nn.Module):
#     def __init__(self,inp_dim,out_dim):
#           super().__init__()
#           self.out_dim = out_dim
#           self.weight_q = nn.Parameter(torch.rand([inp_dim,out_dim]))
#           self.weight_k = nn.Parameter(torch.rand([inp_dim,out_dim]))
#           self.weight_v = nn.Parameter(torch.rand([inp_dim,out_dim]))
#     def forward(self,embed_vectors):
#          query = embed_vectors @ self.weight_q
#          keys = embed_vectors @ self.weight_k
#          values = embed_vectors @ self.weight_v
#          attention_scores = query @ keys.T
#          attention_weights = torch.softmax(attention_scores/(keys.shape[-1]**0.5),dim=-1)
#          context_vectors = attention_weights @ values
#          return context_vectors

# # including causal and drop-out masking and scaling to all batches

# class Self_attention_v2(nn.Module):
#     def __init__(self,inp_dim,out_dim,dropout,context_len,bias):
#           super().__init__()
#           self.out_dim = out_dim
#           self.drop_out = nn.Dropout(dropout)
#           self.w_q = nn.Linear(inp_dim,out_dim,bias)
#           self.w_k = nn.Linear(inp_dim,out_dim,bias)
#           self.w_v = nn.Linear(inp_dim,out_dim,bias)
#           self.register_buffer('mask',torch.triu(torch.ones(context_len,context_len),diagonal=1))
#     def forward(self,embed_vectors):
#          b, context_len, inp_dim = embed_vectors.shape
#          query = self.w_q(embed_vectors)
#          keys = self.w_k(embed_vectors)
#          values = self.w_v(embed_vectors)
#          attention_scores = query @ keys.transpose(1,2)
#          attention_scores.masked_fill_(self.mask.bool(),-torch.inf)
#          attention_weights = torch.softmax(attention_scores/(keys.shape[-1]**0.5),dim=-1)
#          attention_weights = self.drop_out(attention_weights)
#          context_vectors = attention_weights @ values
#          return context_vectors
    
# class Multi_head_attention(nn.Module):
#      def __init__(self,num_heads,inp_dim,out_dim,context_len,drop_out,bias=False):
#           super().__init__()
#           self.heads = nn.ModuleList([Self_attention_v2(inp_dim,out_dim,drop_out,context_len,bias) for _ in range(num_heads)])
#      def forward(self,embed_vectors):
#           context_vec = torch.cat([ca(embed_vectors) for ca in self.heads],dim=-1)
#           return context_vec

# including multi-head attention along with self-attention by reshaping

# inputs = torch.tensor(
#   [[0.43, 0.15, 0.89], # Your     (x^1)
#    [0.55, 0.87, 0.66], # journey  (x^2)
#    [0.57, 0.85, 0.64], # starts
#    [0.22, 0.58, 0.33], # with 
#    [0.77, 0.25, 0.10], # one
#    [0.05, 0.80, 0.55]] # step
# )

# inp_dim = 3
# out_dim = 6
# num_heads = 3
# batch = torch.stack((inputs,inputs),dim=0)
# mha = Multi_head_attention_v2(inp_dim,out_dim,num_heads,context_len=batch.shape[1],drop_out=0.0)
# print(mha(batch).shape)


# class Example_deep_nn(nn.Module):
#     def __init__(self,inp_dim,layer_shape,use_shortcut) -> None:
#         super().__init__()
#         self.use_shortcut = use_shortcut
#         self.layers = nn.ModuleList([
#             nn.Sequential(nn.Linear(inp_dim,layer_shape[0],bias=True),GeLU()),
#             nn.Sequential(nn.Linear(layer_shape[0],layer_shape[1],bias=True),GeLU()),
#             nn.Sequential(nn.Linear(layer_shape[1],layer_shape[2],bias=True),GeLU()),
#             nn.Sequential(nn.Linear(layer_shape[2],layer_shape[3],bias=True),GeLU()),
#             nn.Sequential(nn.Linear(layer_shape[3],layer_shape[4],bias=True),GeLU()),
#         ])
#     def forward(self,x):
#         for layer in self.layers:
#             out = layer(x)
#             if self.use_shortcut  and out.shape == x.shape:
#                 x = x+out
#             else:
#                 x = out
#         return x

# def print_grad(model,x,target):
#     loss = nn.MSELoss()
#     loss = loss(model(x),target)
#     loss.backward()
#     for name,param in model.named_parameters():
#         if 'weight' in name:
#             print(f"{name} has gradient mean of {param.grad.abs().mean()}")



## testing activation functions
# x = torch.linspace(-3,3,100)
# y_gelu = nn.functional.gelu(x)
# y_relu = nn.functional.relu(x)

# for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
#     plt.subplot(1, 2, i)
#     plt.plot(x, y)
#     plt.title(f"{label} activation function")
#     plt.xlabel("x")
#     plt.ylabel(f"{label}(x)")
#     plt.grid(True)
# plt.tight_layout()
# plt.show()

## testing shortcuts
# inp_dim = 3
# layer_shape = [3,3,3,3,1]
# model_without_shortcut = Example_deep_nn(inp_dim,layer_shape,use_shortcut=False)
# target = torch.tensor([[0.]])
# torch.manual_seed(123)
# print_grad(model_without_shortcut,torch.tensor([[-1.,0.,1.]]),target)
# model_with_shortcut = Example_deep_nn(inp_dim,layer_shape,use_shortcut=True)
# torch.manual_seed(123)
# print_grad(model_with_shortcut,torch.tensor([[1.,0.,-1.]]),target)

import tiktoken
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader,Dataset


# GPT2_config = {
#      "vocab_size": 50257,
#      "num_heads" : 12,
#      "num_layers" : 12,
#      "qkv_bias" : False,
#      "context_len" : 1024,
#      "embed_dim":768,
#      "drop_rate" : 0.1
# }

class GPT_dataset_v1(Dataset):
    def __init__(self,txt,tokenizer,max_len,stride):
        self.tokenizer = tokenizer
        self.input_id = []
        self.target_id = []
        tokens = tokenizer.encode(txt)
        for i in range(0,len(tokens)-max_len,stride):
            inp = tokens[i:i+max_len]
            target = tokens[i+1:i+max_len+1]
            self.input_id.append(torch.tensor(inp))
            self.target_id.append(torch.tensor(target))
    def __len__(self):
        return len(self.input_id)
    def __getitem__(self,id):
        return self.input_id[id],self.target_id[id]

# class Multi_head_attention_v2(nn.Module):
#      def __init__(self,inp_dim,out_dim,num_heads,drop_out,context_len,bias=False):
#           super().__init__()
#           assert out_dim % num_heads == 0," out_dim must be divisible by num_heads"
#           self.num_heads = num_heads
#           self.head_dim = out_dim // num_heads 
#           self.out_dim = out_dim
#           self.w_q = nn.Linear(inp_dim,out_dim,bias=bias)
#           self.w_k = nn.Linear(inp_dim,out_dim,bias=bias)
#           self.w_v = nn.Linear(inp_dim,out_dim,bias=bias)
#           self.dropout = nn.Dropout(drop_out)
#           self.register_buffer('mask',torch.triu(torch.ones(context_len,context_len),diagonal=1))

#      def forward(self,embed_vec):
#           b,num_tokens, inp_dim = embed_vec.shape
#           query = self.w_q(embed_vec)
#           keys = self.w_k(embed_vec)
#           values = self.w_v(embed_vec)

#           query = query.view([b,num_tokens,self.num_heads,self.head_dim])
#           keys = keys.view([b,num_tokens,self.num_heads,self.head_dim])
#           values = values.view([b,num_tokens,self.num_heads,self.head_dim])

#           query = query.transpose(1,2)
#           keys = keys.transpose(1,2)
#           values = values.transpose(1,2)

#           attention_scores = query @ keys.transpose(2,3)
#           attention_scores.masked_fill_(self.mask.bool(),-torch.inf)
#           attention_scores = torch.softmax(attention_scores/(keys.shape[-1]**0.5),dim=-1)
#           attention_weights = self.dropout(attention_scores)

#           context_vec = (attention_weights @ values).transpose(1,2)
#           context_vec = context_vec.contiguous().view([b,num_tokens,self.out_dim])

#           return context_vec
         
def GPT_dataloader_v1(txt,batchsize=4,stride=128,max_len=256,shuffle=False,drop_last=True):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPT_dataset_v1(txt,tokenizer,max_len,stride)
        dataloader = DataLoader(
             dataset=dataset, batch_size= batchsize,shuffle=shuffle,drop_last=drop_last
        )
        return dataloader

# with open("the_verdict.txt","r",encoding="utf-8") as f:
#     raw_text = f.read()

# dataloader = GPT_dataloader_v1(raw_text,batchsize=8,max_len=4,stride=2)
# data_iter = iter(dataloader)
# inp,target = next(data_iter)
# print(inp)
# #converting token-ids into embedding vectors

# vocab_size = 50257
# output_dim = 256
# torch.manual_seed(123)
# token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
# embedded_tokens = token_embedding_layer(inp)
# print(embedded_tokens.shape)
# # including positional embeddings (relative)

# context_len = 4
# pos_embedding_layer = torch.nn.Embedding(context_len,output_dim)
# pos_embedding = pos_embedding_layer(torch.arange(context_len))
# # print(pos_embedding.shape)

# embedded_tokens = embedded_tokens + pos_embedding
# print(embedded_tokens.shape)

