import torch
import torch.nn as nn
import torch.nn.functional as F
from config import parse_option



opt = parse_option()

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
WINDOW_SIZE = opt.window_size
HEAD_SIZE = opt.head_size
N_EMBED = opt.n_embed
N_LAYERS = opt.n_layer
DEVICE = opt.device
N_FEATURES = opt.n_features
N_HEADS = N_EMBED//HEAD_SIZE



class DecoderHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.key = nn.Linear(N_EMBED, HEAD_SIZE)
        self.query = nn.Linear(N_EMBED, HEAD_SIZE)
        self.value = nn.Linear(N_EMBED, HEAD_SIZE)
        self.tril =  torch.tril(torch.ones((WINDOW_SIZE,WINDOW_SIZE), device=DEVICE)).to(DEVICE)

    def forward(self, x):
        x = x                                                                       # (B,T,C) C = 27
        k = self.key(x)                                                             # (B,T,C) C = 16
        q = self.query(x)                                                           # (B,T,C) C = 16 
        v = self.value(x)                                                           # (B,T,C) C = 16 
        # weights = q @ k.transpose(1,2) * HEAD_SIZE**-0.5                                             # (B,T,C) @ (B,C,T) -->  (B,T,T)            T=8, C=16         
        # weights = weights.masked_fill(self.tril==0, float('-inf'))
        # weights = F.softmax(weights,dim=1)
        # x =  weights @ v 
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)                                                           # (B,T,T) @ (B,T,C) -->  (B,T,C)            T=8, C=16
        return x


class MultiHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleList([DecoderHead() for _ in range(N_HEADS)])
    
    def forward(self, x):
        x = torch.cat([i(x) for i in self.heads], dim=-1)
        return x


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = MultiHead()
        self.ffn = nn.Sequential(
            nn.Linear(N_EMBED,N_EMBED*4),
            nn.ReLU(),
            nn.Linear(4*N_EMBED,N_EMBED)
        )
        self.ln1 = nn.LayerNorm(N_EMBED)
        self.ln2 = nn.LayerNorm(N_EMBED)

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.decoder(x)
        x = self.ln2(x)
        x = x + self.ffn(x)
        return x


class Bitcoin(nn.Module):
    def __init__(self, input_size = N_FEATURES, embed_dim = N_EMBED, num_layers = N_LAYERS, window_size=WINDOW_SIZE, device=DEVICE):
        super(Bitcoin, self).__init__()
        self.device = device
        self.window_size = window_size
        self.linear =  nn.Linear(input_size, embed_dim)
        self.positional_embedding = nn.Embedding(WINDOW_SIZE, N_EMBED)
        self.blocks = nn.ModuleList([Block() for _ in range(num_layers)])
        self.ln = nn.LayerNorm(N_EMBED)
        self.linear_out = nn.Linear(embed_dim, 1)


        
    def forward(self, x):
        x = x.permute(0,2,1)
        token_embed = self.linear(x)
        pos_embed = self.positional_embedding(torch.arange(self.window_size,device=self.device))
        x = token_embed + pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        x = self.linear_out(x)        
        return x
    



   

