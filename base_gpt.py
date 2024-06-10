import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import DataLoader,Dataset

GPT2_config = {
     "vocab_size": 50257,
     "n_heads" : 12,
     "n_layers" : 12,
     "qkv_bias" : False,
     "context_len" : 1024,
     "emb_dim":768,
     "drop_rate" : 0.1
}
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

def GPT_dataloader_v1(txt,batchsize=4,stride=128,max_len=256,shuffle=False,drop_last=True):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPT_dataset_v1(txt,tokenizer,max_len,stride)
        dataloader = DataLoader(
             dataset=dataset, batch_size= batchsize,shuffle=shuffle,drop_last=drop_last
        )
        return dataloader

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class Feed_forward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            GeLU(),
            nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
        )
    def forward(self,x):
        return self.layers(x)

class Multi_head_attention(nn.Module):
     def __init__(self,inp_dim,out_dim,num_heads,drop_out,context_len,bias=False):
          super().__init__()
          assert out_dim % num_heads == 0," out_dim must be divisible by num_heads"
          self.num_heads = num_heads
          self.head_dim = out_dim // num_heads 
          self.out_dim = out_dim
          self.w_q = nn.Linear(inp_dim,out_dim,bias=bias)
          self.w_k = nn.Linear(inp_dim,out_dim,bias=bias)
          self.w_v = nn.Linear(inp_dim,out_dim,bias=bias)
          self.dropout = nn.Dropout(drop_out)
          self.out_proj = nn.Linear(out_dim,out_dim)
          self.register_buffer('mask',torch.triu(torch.ones(context_len,context_len),diagonal=1))

     def forward(self,embed_vec):
          b,num_tokens, inp_dim = embed_vec.shape
          query = self.w_q(embed_vec)
          keys = self.w_k(embed_vec)
          values = self.w_v(embed_vec)

          query = query.view([b,num_tokens,self.num_heads,self.head_dim])
          keys = keys.view([b,num_tokens,self.num_heads,self.head_dim])
          values = values.view([b,num_tokens,self.num_heads,self.head_dim])

          query = query.transpose(1,2)
          keys = keys.transpose(1,2)
          values = values.transpose(1,2)

          attention_scores = query @ keys.transpose(2,3)
          attention_scores.masked_fill_(self.mask.bool()[:num_tokens,:num_tokens],-torch.inf)
          attention_scores = torch.softmax(attention_scores/(keys.shape[-1]**0.5),dim=-1)
          attention_weights = self.dropout(attention_scores)

          context_vec = (attention_weights @ values).transpose(1,2)
          context_vec = context_vec.contiguous().view([b,num_tokens,self.out_dim])

          context_vec = self.out_proj(context_vec)
          return context_vec
    
class Layer_norm(nn.Module):
    def __init__(self,normshape,eps=1e-15) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normshape))
        self.shift = nn.Parameter(torch.zeros(normshape))
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True,unbiased=False)
        out = (x-mean)/torch.sqrt(var + self.eps)
        out = self.scale * out + self.shift
        return out

class Transformer_blk(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.MHA = Multi_head_attention(
            inp_dim=cfg["emb_dim"],
            out_dim=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            drop_out=cfg["drop_rate"],
            context_len=cfg["context_len"],
            bias=cfg["qkv_bias"]
        )
        self.ff = Feed_forward(cfg)
        self.norm1 = Layer_norm(normshape=cfg["emb_dim"])
        self.norm2 = Layer_norm(normshape= cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
    def forward(self,x):
        shortcut = x
        x = self.norm1(x)
        x = self.MHA(x)
        x = self.drop(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)
        x = x+ shortcut
        return x

# only for a single batch
class GPT_model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.tok_embed = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_embed = nn.Embedding(cfg["context_len"],cfg["emb_dim"])
        self.trf_blks = nn.Sequential(*[Transformer_blk(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = Layer_norm(cfg["emb_dim"]) 
        self.out_head = nn.Linear(cfg["emb_dim"],cfg["vocab_size"],bias=False)
    
    def forward(self,x):
        batch_size,context_len = x.shape
        tok_emb = self.tok_embed(x)
        pos_emb = self.pos_embed(torch.arange(context_len)) ##
        emb = tok_emb + pos_emb
        emb = self.dropout(emb)
        proc_emb = self.final_norm(self.trf_blks(emb))
        logits = self.out_head(proc_emb)
        return logits

# including temperature and top_k sampling

def generate_text(model,inp_tokens,new_tokens,context_len,temp=1,top_k = None):
    for _ in range(new_tokens):
        inp = inp_tokens[:,-context_len:]   # including features upto a max of context_len from back
        # print(inp)
        with torch.no_grad():
            out = model(inp)
        pred_string = out[:,-1,:]           # last_row --> prediction
        if top_k is not None:
            top_logits,_ = torch.topk(pred_string,top_k)
            min_val = top_logits[:,-1]
            pred_string = torch.where(
                pred_string < min_val,
                torch.tensor(float('-inf')).to(pred_string.device),
                pred_string
            ) 
        if temp > 0.0 :
            pred_string = pred_string/temp
            probs = torch.softmax(pred_string,dim=-1)
            pred_token = torch.multinomial(probs,num_samples=1)
        else:
            pred_token = torch.argmax(probs,dim=-1,keepdim=True)
        inp_tokens = torch.cat((inp_tokens,pred_token),dim=1)
    return inp_tokens

# print("Revised")
# #testing GPT_model
# tokenizer = tiktoken.get_encoding("gpt2")
# batch = []
# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"
# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim=0)

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

# mha = Multi_head_attention(inp_dim,out_dim,num_heads,drop_out=0.2,context_len=7)

# print(mha(batch).shape)

# torch.manual_seed(123)

# model = GPT_model(GPT2_config)
# print(f"input : {batch.shape}\n ",batch)
# print(f"output : {model(batch).shape} \n", model(batch))

# total_params = sum(p.numel() for p  in model.parameters())
# print("Total parameters : ", total_params)
# total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
# print("Totla parametes considering weight tying:",total_params_gpt2)

# txt = "Hello, I am"
# tokens = torch.tensor(tokenizer.encode(txt))
# print("input tokens : ",tokens)
# batch = tokens.unsqueeze(0)
# model.eval()
# pred = generate_text(model,batch,new_tokens=4,context_len=GPT2_config["context_len"])
# pred_tokens = pred.squeeze(0).tolist()
# out_txt = tokenizer.decode(pred_tokens)
# print(out_txt)