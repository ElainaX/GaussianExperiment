import math
from types import SimpleNamespace

import torch
from torch import nn

from utils.graphics_utils import getProjectionMatrix


class LocalLightProbe(nn.Module):
    """Fixed positive-radiance cubemaps captured around glossy regions."""

    CAPTURE_VERSION = 2

    def __init__(self, count=4, resolution=128, radiance_max=4.0):
        super().__init__()
        self.count = max(1, int(count))
        self.resolution = max(2, int(resolution))
        self.radiance_max = float(radiance_max)
        self.register_buffer('cubemaps', torch.zeros(self.count, 6, 3, self.resolution, self.resolution))
        self.register_buffer('validities', torch.zeros(self.count, 6, 1, self.resolution, self.resolution))
        # Anchors select spatial regions; capture positions are shifted toward
        # a nearby training camera so the virtual camera is outside the glass.
        self.register_buffer('anchors', torch.zeros(self.count, 3))
        self.register_buffer('capture_positions', torch.zeros(self.count, 3))
        self.register_buffer('initialized_count', torch.zeros((), dtype=torch.long))
        self.register_buffer('active_count', torch.zeros((), dtype=torch.long))
        self.register_buffer('capture_version', torch.tensor(self.CAPTURE_VERSION, dtype=torch.long))

    @property
    def is_initialized(self):
        return int(self.initialized_count.item()) > 0

    @property
    def is_active(self):
        return int(self.active_count.item()) > 0

    @staticmethod
    def _distance2(points, centers):
        return (
            (points * points).sum(dim=-1, keepdim=True)
            + (centers * centers).sum(dim=-1).unsqueeze(0)
            - 2.0 * points @ centers.transpose(0, 1)
        ).clamp_min(0.0)

    @torch.no_grad()
    def initialize_regions(self, xyz, scores, threshold=0.15, valid_mask=None,
                           camera_centers=None, surface_offset=0.25):
        """Cluster marked Gaussians and place capture cameras outside them."""
        scores = scores.reshape(-1).clamp(0.0, 1.0)
        eligible = scores >= float(threshold)
        if valid_mask is not None:
            eligible = eligible & valid_mask.reshape(-1).bool()
        candidate_ids = eligible.nonzero(as_tuple=True)[0]
        if candidate_ids.numel() == 0:
            self.initialized_count.zero_()
            self.active_count.zero_()
            return 0

        points = xyz.detach()[candidate_ids]
        weights = scores[candidate_ids].clamp_min(1e-4)
        number = min(self.count, points.shape[0])

        # Weighted farthest-point seeds, followed by weighted Lloyd steps.
        chosen = [int(torch.argmax(weights).item())]
        min_distance2 = ((points - points[chosen[0]]) ** 2).sum(dim=-1)
        for _ in range(1, number):
            priority = min_distance2 * (0.25 + 0.75 * weights)
            next_index = int(torch.argmax(priority).item())
            chosen.append(next_index)
            distance2 = ((points - points[next_index]) ** 2).sum(dim=-1)
            min_distance2 = torch.minimum(min_distance2, distance2)
        anchors = points[torch.tensor(chosen, device=points.device)]

        for _ in range(3):
            region = self._distance2(points, anchors).argmin(dim=-1)
            updated = []
            for region_id in range(number):
                member = region == region_id
                if member.any():
                    member_weight = weights[member].unsqueeze(-1)
                    center = (points[member] * member_weight).sum(dim=0)
                    center = center / member_weight.sum().clamp_min(1e-6)
                else:
                    center = anchors[region_id]
                updated.append(center)
            anchors = torch.stack(updated)

        capture_positions = anchors.clone()
        if camera_centers is not None and camera_centers.numel() > 0:
            cameras = camera_centers.to(device=anchors.device, dtype=anchors.dtype)
            nearest_camera = self._distance2(anchors, cameras).argmin(dim=-1)
            outward = torch.nn.functional.normalize(
                cameras[nearest_camera] - anchors, dim=-1, eps=1e-8
            )
            capture_positions = anchors + float(surface_offset) * outward

        self.anchors[:number].copy_(anchors)
        self.capture_positions[:number].copy_(capture_positions)
        if number < self.count:
            self.anchors[number:].copy_(anchors[-1:].expand(self.count - number, -1))
            self.capture_positions[number:].copy_(capture_positions[-1:].expand(self.count - number, -1))
        self.cubemaps.zero_()
        self.validities.zero_()
        self.initialized_count.fill_(number)
        self.active_count.zero_()
        return number

    def assign_regions(self, positions, initialized=False):
        count = int((self.initialized_count if initialized else self.active_count).item())
        if count == 0 or positions.numel() == 0:
            index = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)
            distance = positions.new_full((positions.shape[0],), float('inf'))
            return index, distance
        distance2 = self._distance2(positions, self.anchors[:count])
        distance2, index = distance2.min(dim=-1)
        return index, distance2.sqrt()

    @staticmethod
    def direction_to_face_uv(directions):
        """Map directions to the same face bases used by probe cameras."""
        d = torch.nn.functional.normalize(directions, dim=-1, eps=1e-8)
        x, y, z = d.unbind(dim=-1)
        ax, ay, az = x.abs(), y.abs(), z.abs()
        dx, dy, dz = ax.clamp_min(1e-8), ay.clamp_min(1e-8), az.clamp_min(1e-8)
        major_x = (ax >= ay) & (ax >= az)
        major_y = (~major_x) & (ay >= az)
        major_z = ~(major_x | major_y)

        face = torch.empty_like(x, dtype=torch.long)
        u, v = torch.empty_like(x), torch.empty_like(x)
        mask = major_x & (x >= 0)
        face[mask], u[mask], v[mask] = 0, z[mask] / dx[mask], -y[mask] / dx[mask]
        mask = major_x & (x < 0)
        face[mask], u[mask], v[mask] = 1, -z[mask] / dx[mask], -y[mask] / dx[mask]
        mask = major_y & (y >= 0)
        face[mask], u[mask], v[mask] = 2, x[mask] / dy[mask], -z[mask] / dy[mask]
        mask = major_y & (y < 0)
        face[mask], u[mask], v[mask] = 3, x[mask] / dy[mask], z[mask] / dy[mask]
        mask = major_z & (z >= 0)
        face[mask], u[mask], v[mask] = 4, x[mask] / dz[mask], y[mask] / dz[mask]
        mask = major_z & (z < 0)
        face[mask], u[mask], v[mask] = 5, -x[mask] / dz[mask], y[mask] / dz[mask]
        return face, torch.stack((u, v), dim=-1).clamp(-1.0, 1.0)

    def _sample(self, texture, probe_index, directions):
        face, uv = self.direction_to_face_uv(directions)
        pixel = (uv + 1.0) * 0.5 * (self.resolution - 1)
        x0, y0 = pixel[:, 0].floor().long(), pixel[:, 1].floor().long()
        x1 = (x0 + 1).clamp_max(self.resolution - 1)
        y1 = (y0 + 1).clamp_max(self.resolution - 1)
        wx = (pixel[:, 0] - x0).unsqueeze(-1)
        wy = (pixel[:, 1] - y0).unsqueeze(-1)

        channels = texture.shape[2]
        flat = texture.permute(0, 1, 3, 4, 2).reshape(-1, channels)
        offset = (probe_index * 6 + face) * self.resolution * self.resolution
        c00 = flat[offset + y0 * self.resolution + x0]
        c10 = flat[offset + y0 * self.resolution + x1]
        c01 = flat[offset + y1 * self.resolution + x0]
        c11 = flat[offset + y1 * self.resolution + x1]
        top = c00 * (1.0 - wx) + c10 * wx
        bottom = c01 * (1.0 - wx) + c11 * wx
        return top * (1.0 - wy) + bottom * wy

    def forward(self, positions, directions):
        """Return positive local radiance, alpha validity, region and distance."""
        probe_index, distance = self.assign_regions(positions)
        if not self.is_active or positions.numel() == 0:
            radiance = positions.new_zeros((positions.shape[0], 3))
            validity = positions.new_zeros((positions.shape[0], 1))
            return radiance, validity, probe_index, distance
        radiance = self._sample(self.cubemaps, probe_index, directions)
        validity = self._sample(self.validities, probe_index, directions)
        return radiance.clamp(0.0, self.radiance_max), validity.clamp(0.0, 1.0), probe_index, distance

    @torch.no_grad()
    def set_face(self, probe_index, face_index, radiance, validity):
        self.cubemaps[probe_index, face_index].copy_(radiance.clamp(0.0, self.radiance_max))
        self.validities[probe_index, face_index].copy_(validity.clamp(0.0, 1.0))

    @torch.no_grad()
    def activate(self):
        self.active_count.copy_(self.initialized_count)


def make_cubemap_camera(center, face_index, resolution, znear=0.01, zfar=1000.0):
    """Construct a 90-degree virtual camera matching direction_to_face_uv."""
    device, dtype = center.device, center.dtype
    bases = (
        ((0, 0, 1), (0, -1, 0), (1, 0, 0)),
        ((0, 0, -1), (0, -1, 0), (-1, 0, 0)),
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
    )
    right, down, forward = [torch.tensor(axis, device=device, dtype=dtype) for axis in bases[int(face_index)]]
    c2w = torch.eye(4, device=device, dtype=dtype)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, center
    world_view = torch.linalg.inv(c2w).transpose(0, 1).contiguous()
    fov = math.pi * 0.5
    projection = getProjectionMatrix(znear, zfar, fov, fov).to(device=device, dtype=dtype).transpose(0, 1)
    return SimpleNamespace(
        image_width=int(resolution), image_height=int(resolution), FoVx=fov, FoVy=fov,
        znear=float(znear), zfar=float(zfar), world_view_transform=world_view,
        projection_matrix=projection, full_proj_transform=world_view @ projection,
        camera_center=center,
    )
