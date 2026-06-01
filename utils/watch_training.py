"""边训练边实时绘图的观察脚本。

用法（另开一个终端）：
    python utils/watch_training.py <model_path> [--interval 10]

原理：
    训练进程每隔 record_interval iter 将指标写入 model_path/training_stats.json。
    本脚本每隔 --interval 秒读一次 JSON，用 matplotlib ion 模式刷新窗口。
    训练结束或 Ctrl+C 后窗口保持打开，再按任意键关闭。
"""

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load(json_path):
    try:
        with open(json_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='训练输出目录')
    parser.add_argument('--interval', type=float, default=10.0, help='刷新间隔（秒）')
    args = parser.parse_args()

    json_path = os.path.join(args.model_path, 'training_stats.json')
    print(f'Watching {json_path}  (refresh every {args.interval}s, Ctrl+C to stop)')

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('Training Statistics (live)', fontsize=13)

    try:
        while True:
            d = load(json_path)
            if d and d.get('iters'):
                iters = d['iters']

                ax1.cla()
                ax1.plot(iters, d['n_gaussians'], color='steelblue', linewidth=1.5)
                ax1.set_ylabel('Active Gaussians')
                ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
                ax1.set_title(f'Gaussian Count  (last iter {iters[-1]}, count {d["n_gaussians"][-1]:,})')
                ax1.grid(True, alpha=0.3)

                ax2.cla()
                ax2.plot(iters, d['vram_alloc_mb'],    label='Allocated', color='tomato', linewidth=1.5)
                ax2.plot(iters, d['vram_reserved_mb'], label='Reserved',  color='orange', linewidth=1.5, linestyle='--')
                ax2.set_ylabel('VRAM (MB)')
                ax2.set_xlabel('Iteration')
                ax2.set_title(f'GPU Memory  (reserved {d["vram_reserved_mb"][-1]:.0f} MB)')
                ax2.legend(loc='upper left')
                ax2.grid(True, alpha=0.3)

                fig.tight_layout()
                plt.pause(0.1)
            else:
                print('Waiting for training_stats.json ...')

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print('\nStopped. Window will stay open.')
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    main()
