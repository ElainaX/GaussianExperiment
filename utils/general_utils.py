#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch


def inverse_sigmoid(x):
    if torch.is_tensor(x):
        return torch.log(x / (1 - x))
    else:
        return np.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device='cuda')

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device='cuda')
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith('\n'):
                    old_f.write(x.replace('\n', ' [{}]\n'.format(str(datetime.now().strftime('%d/%m %H:%M:%S')))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device('cuda:0'))


def create_rotation_matrix_from_direction_vector_batch(direction_vectors):
    # Normalize the batch of direction vectors
    direction_vectors = direction_vectors / torch.norm(direction_vectors, dim=-1, keepdim=True)
    # Create a batch of arbitrary vectors that are not collinear with the direction vectors
    v1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).to(direction_vectors.device).expand(direction_vectors.shape[0], -1).clone()
    is_collinear = torch.all(torch.abs(direction_vectors - v1) < 1e-5, dim=-1)
    v1[is_collinear] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).to(direction_vectors.device)

    # Calculate the first orthogonal vectors
    v1 = torch.cross(direction_vectors, v1)
    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True))
    # Calculate the second orthogonal vectors by taking the cross product
    v2 = torch.cross(direction_vectors, v1)
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True))
    # Create the batch of rotation matrices with the direction vectors as the last columns
    rotation_matrices = torch.stack((v1, v2, direction_vectors), dim=-1)
    return rotation_matrices


# from kornia.geometry import conversions
# def normal_to_rotation(normals):
#     rotations = create_rotation_matrix_from_direction_vector_batch(normals)
#     rotations = conversions.rotation_matrix_to_quaternion(rotations,eps=1e-5, order=conversions.QuaternionCoeffOrder.WXYZ)
#     return rotations


class GaussianTracker:
    """训练过程中定期采样指标，支持训练结束后绘图，也支持边训练边绘图。

    接口：
        tracker = GaussianTracker(model_path, interval=100)
        tracker.record(iteration, gaussians)   # 主循环里每 iter 调用一次，内部按 interval 过滤
        tracker.draw()                          # 训练结束后调用，保存折线图到 model_path

    记录的指标：
        - 有效高斯数量
        - GPU 显存已分配（实际用量）
        - GPU 显存已预留（PyTorch 向驱动申请的总块，是显存压力的真实指标）

    崩溃安全：每次 record 后立即将数据序列化到 model_path/training_stats.json，
    训练中途崩溃后数据不丢失，可用 draw_from_json(json_path) 单独复原图表。

    边训练边绘图：
        训练进程本身不弹窗（无 display 服务器时弹窗会卡死），而是每次 record 后
        写 JSON。在另一个终端运行：
            python utils/watch_training.py <model_path>
        该脚本会每 10 秒读一次 JSON 并刷新 matplotlib 实时窗口。
    """

    def __init__(self, model_path: str, interval: int = 100):
        self.interval = interval
        self.json_path = os.path.join(model_path, 'training_stats.json')
        self.iters: list[int] = []
        self.n_gaussians: list[int] = []
        self.vram_alloc_mb: list[float] = []
        self.vram_reserved_mb: list[float] = []

    def record(self, iteration: int, gaussians) -> None:
        if iteration % self.interval != 0:
            return

        self.iters.append(iteration)
        self.n_gaussians.append(gaussians.get_xyz.shape[0])
        self.vram_alloc_mb.append(round(torch.cuda.memory_allocated() / 1024 ** 2, 1))
        self.vram_reserved_mb.append(round(torch.cuda.memory_reserved() / 1024 ** 2, 1))

        # 崩溃安全：立即持久化
        with open(self.json_path, 'w') as f:
            json.dump({
                'iters': self.iters,
                'n_gaussians': self.n_gaussians,
                'vram_alloc_mb': self.vram_alloc_mb,
                'vram_reserved_mb': self.vram_reserved_mb,
            }, f)

    def draw(self) -> None:
        if not self.iters:
            return
        _plot_stats(
            self.iters, self.n_gaussians, self.vram_alloc_mb, self.vram_reserved_mb,
            out_path=os.path.join(os.path.dirname(self.json_path), 'training_stats.png'),
        )

    @staticmethod
    def draw_from_json(json_path: str) -> None:
        """从 JSON 文件单独复原图表，供训练崩溃后补绘。"""
        with open(json_path) as f:
            d = json.load(f)
        _plot_stats(
            d['iters'], d['n_gaussians'], d['vram_alloc_mb'], d['vram_reserved_mb'],
            out_path=json_path.replace('.json', '.png'),
        )


def _plot_stats(iters, n_gaussians, vram_alloc_mb, vram_reserved_mb, out_path: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('Training Statistics', fontsize=13)

    # ── 高斯数量 ──────────────────────────────────────────────────────────
    ax1.plot(iters, n_gaussians, color='steelblue', linewidth=1.5)
    ax1.set_ylabel('Active Gaussians')
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Gaussian Count')

    # ── 显存 ──────────────────────────────────────────────────────────────
    ax2.plot(iters, vram_alloc_mb,    label='Allocated (MB)', color='tomato',  linewidth=1.5)
    ax2.plot(iters, vram_reserved_mb, label='Reserved (MB)',  color='orange',  linewidth=1.5, linestyle='--')
    ax2.set_ylabel('VRAM (MB)')
    ax2.set_xlabel('Iteration')
    ax2.set_title('GPU Memory Pressure')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[GaussianTracker] stats → {out_path}')


def colormap(img, cmap='jet'):
    import matplotlib.pyplot as plt

    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.0).float().permute(2, 0, 1)
    plt.close()
    return img


def get_minimum_axis(scales, rotations):
    sorted_idx = torch.argsort(scales, descending=False, dim=-1)
    R = build_rotation(rotations)
    R_sorted = torch.gather(R, dim=2, index=sorted_idx[:, None, :].repeat(1, 3, 1)).squeeze()
    x_axis = R_sorted[:, 0, :]  # normalized by defaut

    return x_axis


def flip_align_view(normal, viewdir):
    # normal: (N, 3), viewdir: (N, 3)
    dotprod = torch.sum(normal * -viewdir, dim=-1, keepdims=True)  # (N, 1)
    non_flip = dotprod >= 0  # (N, 1)
    normal_flipped = normal * torch.where(non_flip, 1, -1)  # (N, 3)
    return normal_flipped, non_flip
