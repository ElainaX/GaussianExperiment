import torch
from torch import nn


class LocalLightProbe(nn.Module):
    """Small learnable cubemap residuals anchored at glossy scene regions."""

    def __init__(self, count=8, resolution=32, max_residual=0.5):
        super().__init__()
        self.count = max(1, int(count))
        self.resolution = max(2, int(resolution))
        self.max_residual = float(max_residual)
        # [probe, face, RGB, H, W]. Zero initialization makes this branch an
        # exact no-op before the RGB reconstruction loss learns a correction.
        self.cubemaps = nn.Parameter(
            torch.zeros(self.count, 6, 3, self.resolution, self.resolution)
        )
        self.register_buffer('centers', torch.zeros(self.count, 3))
        self.register_buffer('active_count', torch.zeros((), dtype=torch.long))

    @property
    def is_active(self):
        return int(self.active_count.item()) > 0

    @torch.no_grad()
    def initialize_centers(self, xyz, scores, threshold=0.15, valid_mask=None):
        """Spatially partition marked Gaussians with weighted farthest points."""
        scores = scores.reshape(-1).clamp(0.0, 1.0)
        eligible = scores >= float(threshold)
        if valid_mask is not None:
            eligible = eligible & valid_mask.reshape(-1).bool()
        candidate_ids = eligible.nonzero(as_tuple=True)[0]
        if candidate_ids.numel() == 0:
            self.active_count.zero_()
            return 0

        points = xyz.detach()[candidate_ids]
        weights = scores[candidate_ids]
        number = min(self.count, points.shape[0])
        chosen = [int(torch.argmax(weights).item())]
        min_distance2 = ((points - points[chosen[0]]) ** 2).sum(dim=-1)
        for _ in range(1, number):
            priority = min_distance2 * (0.25 + 0.75 * weights)
            next_index = int(torch.argmax(priority).item())
            chosen.append(next_index)
            distance2 = ((points - points[next_index]) ** 2).sum(dim=-1)
            min_distance2 = torch.minimum(min_distance2, distance2)

        selected = points[torch.tensor(chosen, device=points.device)]
        self.centers[:number].copy_(selected)
        if number < self.count:
            self.centers[number:].copy_(selected[-1:].expand(self.count - number, -1))
        self.active_count.fill_(number)
        return number

    @staticmethod
    def direction_to_face_uv(directions):
        """Map normalized XYZ directions to cubemap face and UV in [-1, 1]."""
        d = torch.nn.functional.normalize(directions, dim=-1, eps=1e-8)
        x, y, z = d.unbind(dim=-1)
        ax, ay, az = x.abs(), y.abs(), z.abs()
        dx = ax.clamp_min(1e-8)
        dy = ay.clamp_min(1e-8)
        dz = az.clamp_min(1e-8)
        major_x = (ax >= ay) & (ax >= az)
        major_y = (~major_x) & (ay >= az)
        major_z = ~(major_x | major_y)

        face = torch.empty_like(x, dtype=torch.long)
        u = torch.empty_like(x)
        v = torch.empty_like(x)

        positive = x >= 0
        mask = major_x & positive
        face[mask], u[mask], v[mask] = 0, -z[mask] / dx[mask], -y[mask] / dx[mask]
        mask = major_x & ~positive
        face[mask], u[mask], v[mask] = 1, z[mask] / dx[mask], -y[mask] / dx[mask]

        positive = y >= 0
        mask = major_y & positive
        face[mask], u[mask], v[mask] = 2, x[mask] / dy[mask], z[mask] / dy[mask]
        mask = major_y & ~positive
        face[mask], u[mask], v[mask] = 3, x[mask] / dy[mask], -z[mask] / dy[mask]

        positive = z >= 0
        mask = major_z & positive
        face[mask], u[mask], v[mask] = 4, x[mask] / dz[mask], -y[mask] / dz[mask]
        mask = major_z & ~positive
        face[mask], u[mask], v[mask] = 5, -x[mask] / dz[mask], -y[mask] / dz[mask]
        return face, torch.stack((u, v), dim=-1).clamp(-1.0, 1.0)

    def _sample(self, probe_index, directions):
        face, uv = self.direction_to_face_uv(directions)
        pixel = (uv + 1.0) * 0.5 * (self.resolution - 1)
        x0 = pixel[:, 0].floor().long()
        y0 = pixel[:, 1].floor().long()
        x1 = (x0 + 1).clamp_max(self.resolution - 1)
        y1 = (y0 + 1).clamp_max(self.resolution - 1)
        wx = (pixel[:, 0] - x0).unsqueeze(-1)
        wy = (pixel[:, 1] - y0).unsqueeze(-1)

        # Flatten before indexing. Selecting ``cubemaps[probe, face]`` first
        # would materialize an entire R×R face for every visible pixel.
        texture = self.cubemaps.permute(0, 1, 3, 4, 2).reshape(-1, 3)
        face_offset = (probe_index * 6 + face) * self.resolution * self.resolution
        c00 = texture[face_offset + y0 * self.resolution + x0]
        c10 = texture[face_offset + y0 * self.resolution + x1]
        c01 = texture[face_offset + y1 * self.resolution + x0]
        c11 = texture[face_offset + y1 * self.resolution + x1]
        top = c00 * (1.0 - wx) + c10 * wx
        bottom = c01 * (1.0 - wx) + c11 * wx
        return top * (1.0 - wy) + bottom * wy

    def forward(self, positions, directions):
        """Query the nearest active probe and return a bounded RGB residual."""
        active_count = int(self.active_count.item())
        if active_count == 0 or positions.numel() == 0:
            empty = positions.new_zeros((positions.shape[0], 3))
            index = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)
            return empty, index
        centers = self.centers[:active_count]
        distance2 = (
            (positions * positions).sum(dim=-1, keepdim=True)
            + (centers * centers).sum(dim=-1).unsqueeze(0)
            - 2.0 * positions @ centers.transpose(0, 1)
        )
        probe_index = distance2.argmin(dim=-1)
        residual = self.max_residual * torch.tanh(
            self._sample(probe_index, directions)
        )
        return residual, probe_index

    def regularization(self):
        return self.cubemaps.square().mean()
