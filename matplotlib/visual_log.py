#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def plot_group(ax, steps, dicts, title, ylabel):
    keys = sorted(dicts[0].keys())          # 1..7
    for k in keys:
        ys = [d[str(k)] if str(k) in d else d[int(k)] for d in dicts]
        ax.plot(steps, ys, label=f'head {k}')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

def main():
    log_file = 'log.jsonl'
    records = load_jsonl(log_file)
    steps = [r['step'] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    plot_group(axes[0], steps,
               [r['ce_losses'] for r in records],
               'CE losses', 'loss')

    plot_group(axes[1], steps,
               [r['contrast_losses'] for r in records],
               'Contrast losses', 'loss')

    plot_group(axes[2], steps,
               [r['total_losses'] for r in records],
               'Total losses', 'loss')

    axes[3].plot(steps, [r['cal_spent'] for r in records], color='black')
    axes[3].set_title('Time per step')
    axes[3].set_ylabel('seconds')
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == '__main__':
    main()