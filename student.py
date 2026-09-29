"""Your algorithm goes here. The default is a complete, runnable baseline.

Required work: diagnose a limitation and implement a structural/training/memory
change. Explain it, measure its cost and perform a mechanism ablation. Merely
renaming the baseline or reporting a lucky seed is not an algorithmic contribution.
You can replace this factory/model completely while keeping the two model interfaces.
"""
from model import GPT


def build_model(config):
    config = dict(config)     # ① 复制一份配置，不改坏原文件
    config['width'] = 192     # ② 每层向量宽度 128 → 192（容量↑）
    config['heads'] = 6       # ③ 注意力头 4 → 6
    config['depth'] = 6       # ④ 层数 4 → 6（模型更深）
    return GPT(config)        # ⑤ 用新配置造出更大的 GPT
