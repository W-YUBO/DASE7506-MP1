"""C4 候选：256/10/8（深度优先）+ Dropout 0.1。

基于官方 model.py 的 GPT 结构，仅在两个残差分支的输出后加 dropout：
1. 注意力分支 proj 之后
2. MLP 分支之后
dropout 只在训练时生效（model.eval() 时自动关闭），评测保持确定性。
"""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width=128, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.drop1 = nn.Dropout(dropout)   # 注意力残差分支后
        self.drop2 = nn.Dropout(dropout)   # MLP 残差分支后

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop1(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.drop2(self.mlp(self.norm2(x)))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([Block(width, config['heads']) for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    config = dict(config)
    config['width'] = 256
    config['heads'] = 8
    config['depth'] = 10
    return GPT(config)
