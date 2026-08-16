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

from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def l1_loss(network_output, gt, average=True):
    loss = torch.abs((network_output - gt))
    if average:
        return loss.mean()
    else:
        return loss


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def glossy_confidence_gate(score, low=0.08, high=0.20, foreground=None):
    """Convert a fixed glossy score into a smooth, detached confidence map."""
    low = float(low)
    high = max(float(high), low + 1e-6)
    gate = ((score.detach() - low) / (high - low)).clamp(0.0, 1.0)
    gate = gate.square() * (3.0 - 2.0 * gate)
    if foreground is not None:
        gate = gate * foreground.detach().clamp(0.0, 1.0)
    return gate


def glossy_weighted_l1(network_output, gt, confidence):
    """RGB L1 normalized by the amount of valid glossy-region support."""
    confidence = confidence.detach().clamp(0.0, 1.0)
    weighted_error = (network_output - gt).abs() * confidence
    denominator = confidence.sum() * network_output.shape[-3]
    return weighted_error.sum() / denominator.clamp_min(1e-8)


def _haar_high_frequency(image):
    """Return signed LH/HL/HH Haar bands for a CHW image."""
    height = image.shape[-2] - image.shape[-2] % 2
    width = image.shape[-1] - image.shape[-1] % 2
    image = image[..., :height, :width]
    x00 = image[..., 0::2, 0::2]
    x01 = image[..., 0::2, 1::2]
    x10 = image[..., 1::2, 0::2]
    x11 = image[..., 1::2, 1::2]
    lh = 0.5 * (x00 + x01 - x10 - x11)
    hl = 0.5 * (x00 - x01 + x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)
    return torch.cat((lh, hl, hh), dim=-3)


def glossy_weighted_haar_loss(network_output, gt, confidence):
    """Match signed RGB Haar detail only where the glossy prior is confident."""
    height = network_output.shape[-2] - network_output.shape[-2] % 2
    width = network_output.shape[-1] - network_output.shape[-1] % 2
    confidence = confidence.detach()[..., :height, :width].clamp(0.0, 1.0)
    pooled_confidence = F.avg_pool2d(
        confidence.unsqueeze(0), kernel_size=2, stride=2
    ).squeeze(0)
    output_detail = _haar_high_frequency(network_output)
    gt_detail = _haar_high_frequency(gt)
    weighted_error = (output_detail - gt_detail).abs() * pooled_confidence
    denominator = pooled_confidence.sum() * output_detail.shape[-3]
    return weighted_error.sum() / denominator.clamp_min(1e-8)


def glossy_weighted_material_loss(
    roughness,
    reflectance,
    confidence,
    target_roughness=0.15,
    target_reflectance=0.70,
):
    """One-sided soft material prior for confident glossy pixels."""
    confidence = confidence.detach().clamp(0.0, 1.0)
    roughness_excess = torch.relu(roughness - float(target_roughness)).square()
    reflectance_shortfall = torch.relu(
        float(target_reflectance) - reflectance
    ).square()
    weighted_error = (
        roughness_excess + reflectance_shortfall
    ) * confidence
    denominator = confidence.sum() * 2.0
    return weighted_error.sum() / denominator.clamp_min(1e-8)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:, 1:-1, :-2] + disp[:, 1:-1, 2:] - 2 * disp[:, 1:-1, 1:-1])
    grad_disp_y = torch.abs(disp[:, :-2, 1:-1] + disp[:, 2:, 1:-1] - 2 * disp[:, 1:-1, 1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True, average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, average)


def _ssim(img1, img2, window, window_size, channel, size_average=True, average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if not average:
        return ssim_map

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


_lpips_model = None


def lpips(img1, img2):
    global _lpips_model
    if _lpips_model is None:
        from lpipsPyTorch import LPIPS

        _lpips_model = LPIPS(net_type='vgg').cuda()
    return _lpips_model(img1.unsqueeze(0), img2.unsqueeze(0)).squeeze()


def entropy_loss(alpha):
    loss = -alpha * torch.log(alpha + 1e-10) - (1 - alpha) * torch.log(1 - alpha + 1e-10)
    loss = torch.mean(loss)
    return loss


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input + 1e-10) + (1 - target) * torch.log(1 - input + 1e-10)).mean()
