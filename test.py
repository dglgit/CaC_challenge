import torch
import torch.nn as nn
from data_extraction import *
device=torch.device('cpu')



class Thiccatten(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.qw = nn.Linear(k, k * heads)
        self.kw = nn.Linear(k, k * heads)
        self.vw = nn.Linear(k, k * heads)
        self.fc = nn.Linear(k * heads, k)
        self.heads = heads

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        q = self.qw(x).view(b, t, h, k)
        key = self.kw(x).view(b, t, h, k)
        v = self.vw(x).view(b, t, h, k)
        keys = key.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = q.transpose(1, 2).contiguous().view(b * h, t, k)
        values = v.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys / (k ** 0.25)
        queries = queries / (k ** 0.25)
        dot = torch.bmm(keys, queries.transpose(1, 2))
        scaled_dot = torch.softmax(dot, dim=2)
        out = torch.bmm(scaled_dot, values).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, k * h)
        return self.fc(out)


class tblock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = Thiccatten(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)


class c_transformer(nn.Module):
    def __init__(self, heads=8, depth=7, word_embed=20, max_seq=6000):
        super().__init__()
        self.transformers = nn.Sequential(*[tblock(word_embed, heads) for i in range(depth)])
        self.w_embed = nn.EmbeddingBag(len(vocab) + 1, word_embed)
        self.pos_embed = nn.Embedding(max_seq + 1, word_embed)
        self.fc = nn.Linear(word_embed, 1)

    def forward(self, x):
        w = torch.stack([self.w_embed(word2tensor(i).unsqueeze(0),None) for i in remove_stops(x.split(' '))]).transpose(0,1).to(device)
        b, t, k = w.size()
        pos_embeddings = self.pos_embed(torch.arange(t)).expand(b, t, k)
        attended = self.transformers(pos_embeddings + w)
        classes = self.fc(attended).mean(dim=1)
        return torch.sigmoid(classes.reshape(-1))

model_75=torch.load('./Data_and_models/model_75v2')
print(model_75('fdhfha'))

best_model=torch.load('./model80')
correct=0
best_model.eval()
with torch.no_grad():
    for i,j in all_text:
        if torch.round(best_model(i))==j:
            correct+=1
print(correct/len(all_text))
