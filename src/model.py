import torch
import torch.nn as nn

from config import EMB_DIM

class BPR(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=EMB_DIM):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i, j):
        pu = self.user_emb(u)
        qi = self.item_emb(i)
        qj = self.item_emb(j)
        xi = (pu * qi).sum(1)
        xj = (pu * qj).sum(1)
        return xi - xj