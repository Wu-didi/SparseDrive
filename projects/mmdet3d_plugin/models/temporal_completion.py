# Copyright (c) 2024. All rights reserved.
# Motion-Compensated Temporal Completion Module
# 运动补偿时序补全模块

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Dict, Optional, Tuple

__all__ = ['MotionCompensatedTemporalCompletion']


class FeatureQueue:
    """
    历史帧特征队列

    维护多帧历史特征和对应的元数据（用于计算 T_temp2cur）
    """
    def __init__(self, queue_length: int = 3, max_time_interval: float = 2.0):
        self.queue_length = queue_length
        self.max_time_interval = max_time_interval
        self.reset()

    def reset(self):
        """重置队列"""
        self.feature_queue: List[List[torch.Tensor]] = []
        self.T_global_queue: List[np.ndarray] = []
        self.timestamp_queue: List[float] = []

    def push(self, feats: List[torch.Tensor], metas: Dict):
        """添加新帧到队列"""
        # 获取 T_global
        if 'img_metas' in metas and metas['img_metas'] is not None:
            img_metas = metas['img_metas']
            if isinstance(img_metas, list) and len(img_metas) > 0:
                T_global = img_metas[0].get('T_global', np.eye(4))
            else:
                T_global = np.eye(4)
        else:
            T_global = np.eye(4)

        timestamp = metas.get('timestamp', 0.0)
        if isinstance(timestamp, torch.Tensor):
            timestamp = timestamp.item() if timestamp.numel() == 1 else timestamp[0].item()

        # detach 避免梯度累积，不使用 clone 节省内存
        self.feature_queue.append([f.detach() for f in feats])
        self.T_global_queue.append(T_global.copy() if isinstance(T_global, np.ndarray) else T_global)
        self.timestamp_queue.append(timestamp)

        if len(self.feature_queue) > self.queue_length:
            self.feature_queue.pop(0)
            self.T_global_queue.pop(0)
            self.timestamp_queue.pop(0)

    def get(self) -> Tuple[List[List[torch.Tensor]], List[np.ndarray], List[float]]:
        return self.feature_queue, self.T_global_queue, self.timestamp_queue

    def __len__(self):
        return len(self.feature_queue)


class ImageLevelMotionWarp(nn.Module):
    """
    图像级运动补偿 Warp

    使用多深度假设 + 可学习残差偏移
    """
    def __init__(self,
                 embed_dims: int = 256,
                 reference_depths: List[float] = [5, 10, 20, 40],
                 depth_weights: List[float] = None,
                 learnable_offset: bool = True,
                 offset_scale: float = 0.1):
        super().__init__()
        self.reference_depths = reference_depths
        self.learnable_offset = learnable_offset
        self.offset_scale = offset_scale

        if depth_weights is None:
            weights = [1.0 / (d ** 0.5) for d in reference_depths]
            total = sum(weights)
            depth_weights = [w / total for w in weights]
        self.register_buffer('depth_weights', torch.tensor(depth_weights, dtype=torch.float32))

        if learnable_offset:
            self.offset_net = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dims // 2, 2, 3, padding=1),
            )
            nn.init.zeros_(self.offset_net[-1].weight)
            nn.init.zeros_(self.offset_net[-1].bias)

    def compute_base_grid(self, T_temp2cur, lidar2img, H, W, img_shape):
        B = T_temp2cur.shape[0]
        device = T_temp2cur.device
        H_img, W_img = img_shape

        scale_h = H_img / H
        scale_w = W_img / W

        y_feat = torch.arange(H, device=device, dtype=torch.float32)
        x_feat = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_feat, x_feat, indexing='ij')

        xx_img = (xx + 0.5) * scale_w
        yy_img = (yy + 0.5) * scale_h

        grids = []
        for depth in self.reference_depths:
            u_norm = xx_img / W_img * 2 - 1
            v_norm = yy_img / H_img * 2 - 1

            fov_factor = 0.8
            pts_3d = torch.stack([
                torch.full_like(u_norm, depth),
                u_norm * depth * fov_factor,
                v_norm * depth * fov_factor * 0.6,
            ], dim=-1)

            pts_3d = pts_3d.unsqueeze(0).expand(B, -1, -1, -1)
            pts_3d_homo = torch.cat([pts_3d, torch.ones_like(pts_3d[..., :1])], dim=-1)
            pts_3d_homo = pts_3d_homo.view(B, -1, 4)

            pts_cur = torch.bmm(pts_3d_homo, T_temp2cur.transpose(1, 2))
            pts_cur = pts_cur.view(B, H, W, 4)[..., :3]

            pts_cur_homo = torch.cat([pts_cur, torch.ones_like(pts_cur[..., :1])], dim=-1)
            pts_cur_homo = pts_cur_homo.view(B, -1, 4)

            pts_img = torch.bmm(pts_cur_homo, lidar2img.transpose(1, 2))
            pts_img = pts_img.view(B, H, W, 4)

            depth_proj = pts_img[..., 2:3].clamp(min=1e-3)
            pts_2d = pts_img[..., :2] / depth_proj

            grid = torch.stack([
                pts_2d[..., 0] / W_img * 2 - 1,
                pts_2d[..., 1] / H_img * 2 - 1,
            ], dim=-1)

            grids.append(grid)

        grids = torch.stack(grids, dim=0)
        weights = self.depth_weights.view(-1, 1, 1, 1, 1)
        base_grid = (grids * weights).sum(dim=0)

        return base_grid

    def forward(self, feat, T_temp2cur, lidar2img, img_shape):
        B, C, H, W = feat.shape
        base_grid = self.compute_base_grid(T_temp2cur, lidar2img, H, W, img_shape)

        if self.learnable_offset:
            offset = self.offset_net(feat)
            offset = offset.permute(0, 2, 3, 1) * self.offset_scale
            grid = base_grid + offset
        else:
            grid = base_grid

        grid = grid.clamp(-2, 2)
        warped = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return warped


class TemporalCrossAttention(nn.Module):
    """
    时序跨相机注意力

    支持跨相机历史融合，使用 gradient checkpointing 减少显存
    """
    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_cameras: int = 6,
                 num_history: int = 3,
                 dropout: float = 0.1,
                 use_checkpoint: bool = True):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_cameras = num_cameras
        self.num_history = num_history
        self.use_checkpoint = use_checkpoint

        self.query_embed = nn.Parameter(torch.randn(1, embed_dims, 1, 1) * 0.02)
        self.camera_pos_embed = nn.Parameter(torch.randn(num_cameras, embed_dims) * 0.02)
        self.temporal_pos_embed = nn.Parameter(torch.randn(num_history, embed_dims) * 0.02)
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, embed_dims, 1, 1) * 0.02)

        self.register_buffer('camera_adjacency', self._create_camera_adjacency())

        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dims)

    def _create_camera_adjacency(self):
        adjacency = torch.ones(6, 6) * 0.5
        adjacency.fill_diagonal_(1.0)
        adjacency[0, 1] = adjacency[1, 0] = 0.8
        adjacency[0, 2] = adjacency[2, 0] = 0.8
        adjacency[5, 3] = adjacency[3, 5] = 0.8
        adjacency[5, 4] = adjacency[4, 5] = 0.8
        adjacency[1, 3] = adjacency[3, 1] = 0.7
        adjacency[2, 4] = adjacency[4, 2] = 0.7
        return adjacency

    def _attention_forward(self, query_flat, kv_flat, H, W, C):
        """注意力计算，用于 checkpoint"""
        B = query_flat.shape[0]

        q = self.q_proj(query_flat)
        k = self.k_proj(kv_flat)
        v = self.v_proj(kv_flat)

        scale = (C // self.num_heads) ** -0.5

        q = q.view(B, H * W, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, H * W, C)

        out = self.out_proj(out)
        out = self.norm(out)
        return out

    def forward(self, query_cam_idx, history_feats, H, W):
        B, V, T, C, H_in, W_in = history_feats.shape

        query = self.query_embed.expand(B, -1, H, W)
        query = query + self.camera_pos_embed[query_cam_idx].view(1, -1, 1, 1)
        query = query + self.spatial_pos_embed.expand(B, -1, H, W)
        query_flat = query.flatten(2).permute(0, 2, 1)

        # 添加位置编码（不使用 clone，直接加）
        kv = history_feats
        # 创建位置编码张量并广播相加
        temporal_pe = self.temporal_pos_embed[:T].view(1, 1, T, -1, 1, 1)
        camera_pe = self.camera_pos_embed[:V].view(1, V, 1, -1, 1, 1)
        kv = kv + temporal_pe + camera_pe

        kv_flat = kv.permute(0, 1, 2, 4, 5, 3).reshape(B, V * T * H_in * W_in, C)

        # 使用 checkpoint 减少显存
        if self.use_checkpoint and self.training:
            out = checkpoint(self._attention_forward, query_flat, kv_flat, H, W, C, use_reentrant=False)
        else:
            out = self._attention_forward(query_flat, kv_flat, H, W, C)

        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return out


class SpatialDecoder(nn.Module):
    """空间细化解码器"""
    def __init__(self, embed_dims: int = 256, hidden_dims: int = 256, num_res_blocks: int = 2):
        super().__init__()

        layers = [
            nn.Conv2d(embed_dims, hidden_dims, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_res_blocks):
            layers.append(ResBlock(hidden_dims))

        layers.append(nn.Conv2d(hidden_dims, embed_dims, 1))

        self.decoder = nn.Sequential(*layers)
        nn.init.normal_(self.decoder[-1].weight, mean=0, std=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x):
        return self.decoder(x)


class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))


class MotionCompensatedTemporalCompletion(nn.Module):
    """
    运动补偿时序补全模块（完整版）

    核心特性：
    1. 多尺度处理：所有尺度都进行补全
    2. 运动补偿：使用 T_temp2cur + 多深度假设 + 可学习偏移
    3. 跨相机融合：失效相机可从所有相机历史获取信息
    4. Gradient Checkpointing：减少显存占用但不影响精度
    """
    def __init__(self,
                 ch_per_scale: List[int],
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 queue_length: int = 3,
                 num_cameras: int = 6,
                 reference_depths: List[float] = [5, 10, 20, 40],
                 use_checkpoint: bool = True,
                 enable: bool = True):
        super().__init__()
        self.enable = enable
        self.num_scales = len(ch_per_scale)
        self.num_cameras = num_cameras
        self.queue_length = queue_length
        self.use_checkpoint = use_checkpoint

        if not enable:
            return

        self.feature_queue = FeatureQueue(queue_length=queue_length)

        # 每个尺度的模块
        self.motion_warps = nn.ModuleList([
            ImageLevelMotionWarp(embed_dims=c, reference_depths=reference_depths, learnable_offset=True)
            for c in ch_per_scale
        ])

        self.temporal_attention = TemporalCrossAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_cameras=num_cameras,
            num_history=queue_length,
            use_checkpoint=use_checkpoint,
        )

        self.feat_adapters = nn.ModuleList([
            nn.Conv2d(c, embed_dims, 1) if c != embed_dims else nn.Identity()
            for c in ch_per_scale
        ])

        self.out_adapters = nn.ModuleList([
            nn.Conv2d(embed_dims, c, 1) if c != embed_dims else nn.Identity()
            for c in ch_per_scale
        ])

        self.spatial_decoder = SpatialDecoder(embed_dims=embed_dims)

        self.gate = nn.Sequential(
            nn.Conv2d(embed_dims * 2, embed_dims // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims // 2, 1, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[-2].bias, -2.0)

        self.img_shape = (900, 1600)

    def compute_T_temp2cur(self, T_global_hist, T_global_cur, device):
        T_global_inv_cur = np.linalg.inv(T_global_cur)
        T_temp2cur = T_global_inv_cur @ T_global_hist
        return torch.tensor(T_temp2cur, dtype=torch.float32, device=device)

    def forward(self, current_feats, cam_mask, metas):
        if not self.enable:
            return current_feats

        if cam_mask is None or not cam_mask.any():
            with torch.no_grad():
                self.feature_queue.push(current_feats, metas)
            return current_feats

        B, V = cam_mask.shape
        device = cam_mask.device

        history_feats, T_global_queue, _ = self.feature_queue.get()

        if len(history_feats) == 0:
            with torch.no_grad():
                self.feature_queue.push(current_feats, metas)
            return current_feats

        T_global_cur = metas['img_metas'][0].get('T_global', np.eye(4))

        if 'lidar2img' in metas['img_metas'][0]:
            lidar2img = metas['img_metas'][0]['lidar2img']
            if isinstance(lidar2img, np.ndarray):
                lidar2img = torch.tensor(lidar2img, dtype=torch.float32, device=device)
            elif isinstance(lidar2img, list):
                lidar2img = torch.tensor(np.stack(lidar2img), dtype=torch.float32, device=device)
        else:
            lidar2img = torch.eye(4, device=device).unsqueeze(0).expand(V, -1, -1)

        if lidar2img.dim() == 3:
            lidar2img = lidar2img.unsqueeze(0).expand(B, -1, -1, -1)

        T_temp2cur_list = []
        for T_global_hist in T_global_queue:
            T_temp2cur = self.compute_T_temp2cur(T_global_hist, T_global_cur, device)
            T_temp2cur = T_temp2cur.unsqueeze(0).expand(B, -1, -1)
            T_temp2cur_list.append(T_temp2cur)

        outputs = []

        for scale_idx, feat_cur in enumerate(current_feats):
            B, V, C, H, W = feat_cur.shape
            feat_out = feat_cur.clone()

            warped_history = []

            for t, T_temp2cur in enumerate(T_temp2cur_list):
                if t >= len(history_feats):
                    continue

                hist_feat = history_feats[t][scale_idx]

                if hist_feat.shape[0] != B:
                    continue

                warped_cams = []
                for v in range(V):
                    warped = self.motion_warps[scale_idx](
                        hist_feat[:, v], T_temp2cur, lidar2img[:, v], self.img_shape
                    )
                    warped_cams.append(warped)

                warped_hist = torch.stack(warped_cams, dim=1)
                warped_history.append(warped_hist)

            if len(warped_history) == 0:
                outputs.append(feat_out)
                continue

            warped_history = torch.stack(warped_history, dim=2)
            T_actual = warped_history.shape[2]

            warped_history_adapted = self.feat_adapters[scale_idx](
                warped_history.reshape(B * V * T_actual, C, H, W)
            )
            embed_dims = warped_history_adapted.shape[1]
            warped_history_adapted = warped_history_adapted.reshape(B, V, T_actual, embed_dims, H, W)

            for v in range(V):
                missing_mask = cam_mask[:, v]
                if not missing_mask.any():
                    continue

                completed = self.temporal_attention(
                    query_cam_idx=v,
                    history_feats=warped_history_adapted,
                    H=H, W=W,
                )

                completed = self.spatial_decoder(completed)

                feat_cur_adapted = self.feat_adapters[scale_idx](feat_cur[:, v])
                gate_input = torch.cat([feat_cur_adapted, completed], dim=1)
                gate = self.gate(gate_input)

                fused = feat_cur_adapted * (1 - gate) + completed * gate
                fused = self.out_adapters[scale_idx](fused)

                mask = missing_mask.view(B, 1, 1, 1).float()
                feat_out[:, v] = feat_out[:, v] * (1 - mask) + fused * mask

            outputs.append(feat_out)

        with torch.no_grad():
            self.feature_queue.push(current_feats, metas)

        return outputs

    def reset(self):
        """重置历史队列"""
        if hasattr(self, 'feature_queue'):
            self.feature_queue.reset()
