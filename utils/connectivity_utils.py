import torch
import numpy as np
from scipy.spatial import KDTree


def build_edge_candidates(
    vertices: torch.Tensor,
    triangle_indices: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """
    为每条三角形边（共 3P 条）找 k 个最近邻边（按边中点 3D 距离）。
    同一三角形内的边会被排除。

    Args:
        vertices:         [V, 3] GPU tensor（detach 后转 CPU 处理）
        triangle_indices: [P, 3] int tensor
        k:                每条边的候选邻居数

    Returns:
        candidates: [3P, k] LongTensor（CPU），
                    edge j 的候选集为 candidates[j]，存储全局边编号 0..3P-1
    """
    V = vertices.detach().cpu().numpy()
    T = triangle_indices.long().cpu().numpy()
    P = T.shape[0]

    v0, v1, v2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    edge_a = np.concatenate([v0, v1, v2], axis=0)   # [3P, 3]
    edge_b = np.concatenate([v1, v2, v0], axis=0)   # [3P, 3]
    midpoints = (edge_a + edge_b) / 2               # [3P, 3]

    tri_owner = np.tile(np.arange(P), 3)  # [3P]

    query_k = min(k * 4 + 3, 3 * P)
    tree = KDTree(midpoints)
    _, nn_idx = tree.query(midpoints, k=query_k)

    candidates = np.full((3 * P, k), -1, dtype=np.int64)
    for i in range(3 * P):
        valid = [
            j for j in nn_idx[i]
            if j != i and tri_owner[j] != tri_owner[i]
        ]
        n = min(k, len(valid))
        candidates[i, :n] = valid[:n]
        if 0 < len(valid) < k:
            candidates[i, len(valid):] = valid[-1]
        elif len(valid) == 0:
            candidates[i, :] = i  # fallback: 自身，D=0，梯度为 0

    return torch.tensor(candidates, dtype=torch.long)  # CPU tensor


def connectivity_loss(
    vertices: torch.Tensor,
    triangle_indices: torch.Tensor,
    edge_candidates: torch.Tensor,
    importance_score: torch.Tensor,
    vertex_weight: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """
    边级软连通 loss：

        L_conn = mean_e [ omega_e * sum_f P_ef * D_edge(e, f) ]

    D_edge(e=(a,b), f=(c,d)) = min( ||a-c||^2+||b-d||^2, ||a-d||^2+||b-c||^2 )
    P_ef = softmax(-D / tau)    （matching 权重，stop-grad）
    omega_e = importance[tri] * min_opacity[tri]   （置信权重，stop-grad）

    梯度仅通过 D_edge 中的顶点坐标传播。
    """
    T = triangle_indices.long()
    P = T.shape[0]
    dev = vertices.device

    # Shape guard：candidates 与当前三角形数不一致时跳过（prune/densify 后可能过期）
    if edge_candidates.shape[0] != 3 * P:
        return torch.tensor(0.0, device=dev)

    v0 = vertices[T[:, 0]]
    v1 = vertices[T[:, 1]]
    v2 = vertices[T[:, 2]]
    edge_a = torch.cat([v0, v1, v2], dim=0)   # [3P, 3]
    edge_b = torch.cat([v1, v2, v0], dim=0)   # [3P, 3]

    opacity = torch.sigmoid(vertex_weight.squeeze(-1)[T]).min(dim=1).values  # [P]
    imp = importance_score.squeeze()[:P]                                       # [P]
    omega = (imp * opacity).detach()
    tri_owner = torch.arange(P, device=dev).repeat(3)
    omega_per_edge = omega[tri_owner]                                          # [3P]

    cands = edge_candidates.to(dev)   # [3P, k]
    ne_a = edge_a[cands]              # [3P, k, 3]
    ne_b = edge_b[cands]              # [3P, k, 3]

    ea = edge_a.unsqueeze(1)
    eb = edge_b.unsqueeze(1)

    d1 = (ea - ne_a).pow(2).sum(-1) + (eb - ne_b).pow(2).sum(-1)   # [3P, k]
    d2 = (ea - ne_b).pow(2).sum(-1) + (eb - ne_a).pow(2).sum(-1)   # [3P, k]
    D = torch.minimum(d1, d2)                                        # [3P, k]

    P_ef = torch.softmax(-D.detach() / tau, dim=-1)  # stop-grad on weights

    return (omega_per_edge * (P_ef * D).sum(-1)).mean()


def opacity_solidification_loss(
    vertex_weight: torch.Tensor,
    triangle_indices: torch.Tensor,
    importance_score: torch.Tensor,
    importance_threshold: float,
) -> torch.Tensor:
    """
    实体化 loss：

        L_opaque = mean_i [ m_i * (1 - o_i)^2 ]

    m_i = 1 若 importance_score[i] > importance_threshold，否则 0。
    只对高置信三角形施加"变不透明"的压力。
    """
    T = triangle_indices.long()
    P = T.shape[0]
    o_per_tri = torch.sigmoid(vertex_weight.squeeze(-1)[T]).min(dim=1).values  # [P]
    mask = (importance_score.squeeze()[:P] > importance_threshold).float().detach()
    return (mask * (1.0 - o_per_tri).pow(2)).mean()
