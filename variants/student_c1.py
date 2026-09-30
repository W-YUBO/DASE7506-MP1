"""C1 候选：纯容量放大 288/8/8（无结构改动，用官方 model.py 的 GPT）。

用法：cp variants/student_c1.py student.py
训练：python train.py --implementation student --device cuda --seed 17 --eval-every 300 --run-dir runs/c1-288 --steps 4200
"""
from model import GPT


def build_model(config):
    config = dict(config)
    config['width'] = 288
    config['heads'] = 8
    config['depth'] = 8
    return GPT(config)
