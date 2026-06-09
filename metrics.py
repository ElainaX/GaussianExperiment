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
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from lpipsPyTorch import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import ssim

lpips = LPIPS(net_type='vgg').cuda()


def readImages(renders_dir, gt_dir, masks_dir, vis_dir=None, refl_thresh=0.5):
    renders = []
    gts = []
    masks = []
    refl_masks = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        mask = Image.open(masks_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        mask_tensor = tf.to_tensor(mask.convert('L')).unsqueeze(0).cuda()
        masks.append((mask_tensor > 0.5).float())
        image_names.append(fname)

        # refl_score_XXXXX.png：uint8 灰度图，值域 [0,255] 对应分数 [0,1]
        refl_mask = None
        if vis_dir is not None:
            stem = Path(fname).stem
            refl_path = vis_dir / f'refl_score_{stem}.png'
            if refl_path.exists():
                score_img = tf.to_tensor(Image.open(refl_path).convert('L')).unsqueeze(0).cuda()
                refl_mask = (score_img > refl_thresh).float()  # [1,1,H,W]
        refl_masks.append(refl_mask)

    return renders, gts, masks, refl_masks, image_names


def psnr_region(render, gt, mask):
    """只对 mask>0 的像素计算 PSNR，避免背景零值拉偏 MSE。"""
    pixel_mask = mask[0, 0] > 0  # [H, W]
    if not pixel_mask.any():
        return None
    r = render[0][:, pixel_mask]   # [3, N]
    g = gt[0][:, pixel_mask]       # [3, N]
    mse = ((r - g) ** 2).mean()
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse))


def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}

    for scene_dir in model_paths:
        print('Scene:', scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / 'test'

        for method in os.listdir(test_dir):
            print('Method:', method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / 'gt'
            renders_dir = method_dir / 'renders'
            masks_dir = method_dir / 'transparent_masks'
            vis_dir = method_dir / 'vis'
            renders, gts, masks, refl_masks, image_names = readImages(
                renders_dir, gt_dir, masks_dir,
                vis_dir=vis_dir if vis_dir.exists() else None,
                refl_thresh=args.refl_thresh,
            )

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc='Metric evaluation progress'):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx]))

            print('  SSIM : {:>12.7f}'.format(torch.tensor(ssims).mean(), ))
            print('  PSNR : {:>12.7f}'.format(torch.tensor(psnrs).mean(), ))
            print('  LPIPS: {:>12.7f}'.format(torch.tensor(lpipss).mean(), ))
            print('')

            masked_ssims = []
            masked_psnrs = []
            masked_lpips = []

            for idx in tqdm(range(len(renders)), desc='Metric evaluation progress'):
                if (masks[idx] > 0.5).any():
                    masked_ssims.append(ssim(renders[idx] * masks[idx], gts[idx] * masks[idx]))
                    masked_psnrs.append(psnr(renders[idx] * masks[idx], gts[idx] * masks[idx]))
                    masked_lpips.append(lpips(renders[idx] * masks[idx], gts[idx] * masks[idx]))

            print('  Masked SSIM : {:>12.7f}'.format(torch.tensor(masked_ssims).mean(), ))
            print('  Masked PSNR : {:>12.7f}'.format(torch.tensor(masked_psnrs).mean(), ))
            print('  Masked LPIPS: {:>12.7f}'.format(torch.tensor(masked_lpips).mean(), ))
            print('')

            full_dict[scene_dir][method].update({'SSIM': torch.tensor(ssims).mean().item(), 'PSNR': torch.tensor(psnrs).mean().item(), 'LPIPS': torch.tensor(lpipss).mean().item()})
            per_view_dict[scene_dir][method].update(
                {
                    'SSIM': {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    'PSNR': {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    'LPIPS': {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                }
            )

            full_dict[scene_dir][method].update(
                {'Masked SSIM': torch.tensor(masked_ssims).mean().item(), 'Masked PSNR': torch.tensor(masked_psnrs).mean().item(), 'Masked LPIPS': torch.tensor(masked_lpips).mean().item()}
            )
            per_view_dict[scene_dir][method].update(
                {
                    'Masked SSIM': {name: ssim for ssim, name in zip(torch.tensor(masked_ssims).tolist(), image_names)},
                    'Masked PSNR': {name: psnr for psnr, name in zip(torch.tensor(masked_psnrs).tolist(), image_names)},
                    'Masked LPIPS': {name: lp for lp, name in zip(torch.tensor(masked_lpips).tolist(), image_names)},
                }
            )

            # Refl-region PSNR：只对反射分数 > refl_thresh 的像素区域计算
            refl_psnrs = [psnr_region(renders[idx], gts[idx], refl_masks[idx])
                          for idx in range(len(renders)) if refl_masks[idx] is not None]
            if refl_psnrs:
                refl_psnr_val = torch.stack(refl_psnrs).mean().item()
                print('  Refl PSNR (thresh={:.2f}): {:>12.7f}'.format(args.refl_thresh, refl_psnr_val))
                print('')
                full_dict[scene_dir][method].update({'Refl PSNR': refl_psnr_val})
                per_view_dict[scene_dir][method].update(
                    {'Refl PSNR': {name: v.item() for v, name in zip(refl_psnrs, image_names)
                                   if refl_masks[image_names.index(name)] is not None}}
                )

        with open(scene_dir + '/results.json', 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + '/per_view.json', 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == '__main__':
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description='Training script parameters')
    parser.add_argument('--model_paths', '-m', required=True, nargs='+', type=str, default=[])
    parser.add_argument('--refl_thresh', type=float, default=0.5,
                        help='反射分数阈值，与训练时 --refl_thresh 保持一致')
    args = parser.parse_args()
    evaluate(args.model_paths)
