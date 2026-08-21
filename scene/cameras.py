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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from utils.graphics_utils import getProjectionMatrix, getWorld2View2


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        gt_transparent_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device='cuda',
        prior_depth_path=None,
        prior_normal_path=None,
        prior_roughness_path=None,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f'[Warning] Custom device {data_device} failed, fallback to default cuda device')
            self.data_device = torch.device('cuda')

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # Optional, aligned per-view priors. Keep only paths here and decode on
        # demand so hundreds of prior maps do not remain resident in GPU memory.
        self.prior_depth_path = prior_depth_path
        self.prior_normal_path = prior_normal_path
        self.prior_roughness_path = prior_roughness_path
        # Lazily cache resized priors as uint8 on CPU. Prior supervision may
        # access one view every iteration; caching avoids tens of thousands of
        # repeated PNG decodes without occupying persistent GPU memory.
        self._prior_depth_u8_cache = None
        self._prior_normal_u8_cache = None
        self._prior_roughness_u8_cache = None

        self.gt_transparent_mask = gt_transparent_mask.to(self.data_device)

        if gt_alpha_mask is not None:
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def has_image_priors(self):
        return all(
            path is not None
            for path in (
                self.prior_depth_path,
                self.prior_normal_path,
                self.prior_roughness_path,
            )
        )

    def load_image_priors(self, device=None):
        """Load aligned depth, normal and roughness priors for this view.

        Depth and roughness are returned in [0, 1] with shape [1, H, W].
        RGB-encoded normals are decoded to [-1, 1], normalized, and returned
        with shape [3, H, W]. Missing modalities are returned as ``None``.
        Maps are decoded on first use and retained as compact CPU uint8 tensors;
        only the active view is converted and copied to the requested device.
        """
        target_device = self.data_device if device is None else torch.device(device)
        resolution = (self.image_width, self.image_height)

        def load_scalar_map(path, cache_name):
            if path is None:
                return None
            cached = getattr(self, cache_name)
            if cached is None:
                with Image.open(path) as image:
                    resized = image.convert('L').resize(resolution)
                    array = np.array(resized, dtype=np.uint8, copy=True)
                cached = torch.from_numpy(array)[None].contiguous()
                setattr(self, cache_name, cached)
            return cached.to(
                device=target_device, dtype=torch.float32, non_blocking=True
            ) / 255.0

        depth = load_scalar_map(
            self.prior_depth_path, '_prior_depth_u8_cache'
        )
        roughness = load_scalar_map(
            self.prior_roughness_path, '_prior_roughness_u8_cache'
        )
        normal = self.load_normal_prior(device=target_device)

        return {
            'depth': depth,
            'normal': normal,
            'roughness': roughness,
        }

    def load_normal_prior(self, device=None):
        """Load and decode the aligned camera-space normal prior.

        A resized uint8 copy is retained on CPU and converted to float only for
        the active view, keeping the cache compact and GPU usage temporary.
        """
        if self.prior_normal_path is None:
            return None
        target_device = self.data_device if device is None else torch.device(device)
        if self._prior_normal_u8_cache is None:
            resolution = (self.image_width, self.image_height)
            with Image.open(self.prior_normal_path) as image:
                resized = image.convert('RGB').resize(resolution)
                array = np.array(resized, dtype=np.uint8, copy=True)
            self._prior_normal_u8_cache = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        normal = self._prior_normal_u8_cache.to(
            device=target_device, dtype=torch.float32, non_blocking=True
        ) / 255.0
        return F.normalize(normal * 2.0 - 1.0, dim=0, eps=1e-6)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
