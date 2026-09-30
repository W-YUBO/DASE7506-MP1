"""C2 候选：RoPE + SwiGLU，288/8/8（约 9.2M 参数）。

用法：cp variants/student_c2.py student.py
训练：python train.py --implementation student --device cuda --seed 17 --eval-every 300 --run-dir runs/c2-288 --steps 4200
"""
import torch
from torch import nn
from torch.nn import functional as F


def build_rope_cache(context, head_dim, base=10000.0):
    """预计算 RoPE 每个位置、每对维度的旋转角度。

    theta_i = base^(-2i/head_dim)，第 i 对维度的"转速"；
    freqs[m, i] = m * theta_i，第 m 个位置的旋转角度。
    返回 shape [context, head_dim/2]，只在建模型时算一次。
    """
    i = torch.arange(head_dim // 2, dtype=torch.float32)
    theta = base ** (-2 * i / head_dim)          # [head_dim/2]
    m = torch.arange(context, dtype=torch.float32)
    return torch.outer(m, theta)                 # [context, head_dim/2]


def apply_rope(x, cos, sin):
    """按位置旋转 x 的相邻维度对。

    x:   [..., len, dim]，通常是 attention 的 q 或 k
    cos/sin: [len, dim/2]（从 rope cache 切片而来）
    对每对维度 (x[2j], x[2j+1]) 做 2D 旋转：
      x'[2j]   = x[2j]*cos - x[2j+1]*sin
      x'[2j+1] = x[2j+1]*cos + x[2j]*sin
    """
    even = x[..., 0::2]                           # 偶数下标维度
    odd = x[..., 1::2]                            # 奇数下标维度
    rotated = torch.stack([even * cos - odd * sin,
                           odd * cos + even * sin], dim=-1)
    return rotated.flatten(-2)                    # 还原交错顺序


class Block(nn.Module):
    def __init__(self, width=128, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        # SwiGLU MLP：门控线性单元（hidden=3*width，控制参数增长幅度）
        self.gate = nn.Linear(width, 3 * width, bias=False)
        self.up = nn.Linear(width, 3 * width, bias=False)
        self.down = nn.Linear(3 * width, width, bias=False)

    def forward(self, x, cos, sin):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        # RoPE：只旋转 q 和 k，注意力的"位置信息"就编码在相对角度差里
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # 每个位置只看到自己和之前的 token（is_causal）
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        # 不再有学习式绝对位置嵌入：由 RoPE 的旋转角承担位置信息
        freqs = build_rope_cache(self.context, width // config['heads'])
        self.register_buffer('rope_cos', torch.cos(freqs))   # [context, head_dim/2]
        self.register_buffer('rope_sin', torch.sin(freqs))
        self.blocks = nn.ModuleList(
            [Block(width, config['heads']) for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight          # 输入输出 embedding 共享

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        cos = self.rope_cos[:ids.shape[1]]            # 取本窗口用到的位置
        sin = self.rope_sin[:ids.shape[1]]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def forward(self, ids):
        """训练接口：未归一化 next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """评测接口：归一化的 log 概率，严格因果、无跨窗口状态."""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    config = dict(config)
    config['width'] = 288   # 每层向量宽度（容量↑）
    config['heads'] = 8     # 注意力头数（288/8=36 维/头）
    config['depth'] = 8     # Transformer 层数
    return GPT(config)
