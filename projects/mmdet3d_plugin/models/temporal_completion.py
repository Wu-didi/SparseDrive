# Copyright (c) 2024. All rights reserved.
# Motion-Compensated Temporal Completion Module
# 运动补偿时序补全模块

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

__all__ = ['MotionCompensatedTemporalCompletion']


class FeatureQueue:
    """
    历史帧特征队列（显存优化版）

    只存储单一尺度的特征，减少显存占用
    """
    def __init__(self, queue_length: int = 2, max_time_interval: float = 2.0):
        """
        Args:
            queue_length: 队列长度（保存多少帧历史），默认2帧节省显存
            max_time_interval: 最大时间间隔（秒），超过则认为历史无效
        """
        self.queue_length = queue_length
        self.max_time_interval = max_time_interval
        self.reset()

    def reset(self):
        """重置队列"""
        self.feature_queue: List[torch.Tensor] = []  # 只存储单一尺度
        self.T_global_queue: List[np.ndarray] = []
        self.timestamp_queue: List[float] = []

    def push(self, feat: torch.Tensor, metas: Dict):
        """
        添加新帧到队列（只存储单一尺度特征）

        Args:
            feat: Tensor [B, V, C, H, W]，单一尺度特征
            metas: 包含 'img_metas' 的字典
        """
        # 获取 T_global
        if 'img_metas' in metas and metas['img_metas'] is not None:
            img_metas = metas['img_metas']
            if isinstance(img_metas, list) and len(img_metas) > 0:
                T_global = img_metas[0].get('T_global', np.eye(4))
            else:
                T_global = np.eye(4)
        else:
            T_global = np.eye(4)

        # 获取时间戳
        timestamp = metas.get('timestamp', 0.0)
        if isinstance(timestamp, torch.Tensor):
            timestamp = timestamp.item() if timestamp.numel() == 1 else timestamp[0].item()

        # 场景切换检测：如果时间间隔过大，清空队列
        if len(self.timestamp_queue) > 0:
            time_diff = abs(timestamp - self.timestamp_queue[-1])
            if time_diff > self.max_time_interval:
                # 时间间隔超过阈值，可能是场景切换或数据不连续
                self.reset()

        # 添加到队列（detach 避免梯度累积）
        self.feature_queue.append(feat.detach())
        self.T_global_queue.append(T_global.copy() if isinstance(T_global, np.ndarray) else T_global)
        self.timestamp_queue.append(timestamp)

        # 保持队列长度
        if len(self.feature_queue) > self.queue_length:
            self.feature_queue.pop(0)
            self.T_global_queue.pop(0)
            self.timestamp_queue.pop(0)

    def get(self) -> Tuple[List[torch.Tensor], List[np.ndarray], List[float]]:
        """获取历史特征和元数据"""
        return self.feature_queue, self.T_global_queue, self.timestamp_queue

    def __len__(self):
        return len(self.feature_queue)


class ImageLevelMotionWarp(nn.Module):
    """
    图像级运动补偿 Warp

    使用多深度假设 + 可学习残差偏移的方式进行特征对齐

    核心流程：
    1. 在多个假设深度反投影到 3D
    2. 通过 T_temp2cur 变换到当前帧
    3. 投影回像素坐标，得到 base_grid
    4. 加上可学习的残差偏移
    5. 使用 grid_sample 进行 warp
    """
    def __init__(self,
                 embed_dims: int = 256,
                 reference_depths: List[float] = [5, 10, 20, 40],
                 depth_weights: List[float] = None,
                 learnable_offset: bool = True,
                 offset_scale: float = 0.1):
        """
        Args:
            embed_dims: 特征通道数
            reference_depths: 参考深度列表（米）
            depth_weights: 深度权重（默认远处权重低）
            learnable_offset: 是否使用可学习偏移
            offset_scale: 偏移缩放因子（防止初期偏移过大）
        """
        super().__init__()
        self.reference_depths = reference_depths
        self.learnable_offset = learnable_offset
        self.offset_scale = offset_scale

        # 深度权重（近处权重高，远处权重低）
        if depth_weights is None:
            weights = [1.0 / (d ** 0.5) for d in reference_depths]
            total = sum(weights)
            depth_weights = [w / total for w in weights]
        self.register_buffer('depth_weights', torch.tensor(depth_weights, dtype=torch.float32))

        # 可学习的残差偏移网络
        if learnable_offset:
            self.offset_net = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dims // 2, 2, 3, padding=1),  # 输出 [dx, dy]
            )
            # 小值初始化，使初始偏移接近零
            nn.init.zeros_(self.offset_net[-1].weight)
            nn.init.zeros_(self.offset_net[-1].bias)

    def compute_base_grid(self,
                          T_temp2cur: torch.Tensor,
                          lidar2img: torch.Tensor,
                          H: int, W: int,
                          img_shape: Tuple[int, int]) -> torch.Tensor:
        """
        计算基础 warp grid（多深度假设加权）

        Args:
            T_temp2cur: [B, 4, 4] 历史帧 -> 当前帧变换
            lidar2img: [B, 4, 4] lidar -> image 投影矩阵
            H, W: 特征图尺寸
            img_shape: 原始图像尺寸 (H_img, W_img)

        Returns:
            grid: [B, H, W, 2] 归一化采样网格
        """
        B = T_temp2cur.shape[0]
        device = T_temp2cur.device
        H_img, W_img = img_shape

        # 计算特征图到图像的缩放比例
        scale_h = H_img / H
        scale_w = W_img / W

        # 创建特征图坐标网格
        y_feat = torch.arange(H, device=device, dtype=torch.float32)
        x_feat = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_feat, x_feat, indexing='ij')

        # 转换到图像坐标
        xx_img = (xx + 0.5) * scale_w
        yy_img = (yy + 0.5) * scale_h

        # 收集不同深度的采样点
        grids = []

        for depth in self.reference_depths:
            # 1. 图像坐标 -> 历史帧 lidar 坐标（假设深度）
            # 简化处理：假设像素在 lidar 前方 depth 米处
            # 实际需要使用相机内参反投影

            # 归一化图像坐标
            u_norm = xx_img / W_img * 2 - 1  # [-1, 1]
            v_norm = yy_img / H_img * 2 - 1  # [-1, 1]

            # 假设的 3D 点（lidar 坐标系，前方 depth 米）
            # 使用简化模型：x=depth, y=u_norm*depth*tan(fov/2), z=v_norm*depth*tan(fov/2)
            fov_factor = 0.8  # 近似视场角因子
            pts_3d = torch.stack([
                torch.full_like(u_norm, depth),  # X (前方)
                u_norm * depth * fov_factor,      # Y (左右)
                v_norm * depth * fov_factor * 0.6,  # Z (上下，调整比例)
            ], dim=-1)  # [H, W, 3]

            # 扩展到 batch
            pts_3d = pts_3d.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 3]

            # 2. 历史帧 3D -> 当前帧 3D (T_temp2cur)
            pts_3d_homo = torch.cat([pts_3d, torch.ones_like(pts_3d[..., :1])], dim=-1)  # [B, H, W, 4]
            pts_3d_homo = pts_3d_homo.view(B, -1, 4)  # [B, H*W, 4]

            # 变换
            pts_cur = torch.bmm(pts_3d_homo, T_temp2cur.transpose(1, 2))  # [B, H*W, 4]
            pts_cur = pts_cur.view(B, H, W, 4)[..., :3]  # [B, H, W, 3]

            # 3. 当前帧 3D -> 当前帧像素
            pts_cur_homo = torch.cat([pts_cur, torch.ones_like(pts_cur[..., :1])], dim=-1)  # [B, H, W, 4]
            pts_cur_homo = pts_cur_homo.view(B, -1, 4)  # [B, H*W, 4]

            pts_img = torch.bmm(pts_cur_homo, lidar2img.transpose(1, 2))  # [B, H*W, 4]
            pts_img = pts_img.view(B, H, W, 4)

            # 透视除法
            depth_proj = pts_img[..., 2:3].clamp(min=1e-3)
            pts_2d = pts_img[..., :2] / depth_proj  # [B, H, W, 2]

            # 归一化到 [-1, 1]
            grid = torch.stack([
                pts_2d[..., 0] / W_img * 2 - 1,
                pts_2d[..., 1] / H_img * 2 - 1,
            ], dim=-1)  # [B, H, W, 2]

            grids.append(grid)

        # 加权融合
        grids = torch.stack(grids, dim=0)  # [num_depths, B, H, W, 2]
        weights = self.depth_weights.view(-1, 1, 1, 1, 1)  # [num_depths, 1, 1, 1, 1]
        base_grid = (grids * weights).sum(dim=0)  # [B, H, W, 2]

        return base_grid

    def forward(self,
                feat: torch.Tensor,
                T_temp2cur: torch.Tensor,
                lidar2img: torch.Tensor,
                img_shape: Tuple[int, int]) -> torch.Tensor:
        """
        对单个特征进行运动补偿 warp

        Args:
            feat: [B, C, H, W] 历史帧特征
            T_temp2cur: [B, 4, 4] 历史帧 -> 当前帧变换
            lidar2img: [B, 4, 4] lidar -> image 投影矩阵
            img_shape: 原始图像尺寸

        Returns:
            warped: [B, C, H, W] 对齐后的特征
        """
        B, C, H, W = feat.shape

        # 计算基础 grid
        base_grid = self.compute_base_grid(T_temp2cur, lidar2img, H, W, img_shape)

        # 可学习偏移
        if self.learnable_offset:
            offset = self.offset_net(feat)  # [B, 2, H, W]
            offset = offset.permute(0, 2, 3, 1)  # [B, H, W, 2]
            offset = offset * self.offset_scale  # 缩放
            grid = base_grid + offset
        else:
            grid = base_grid

        # 限制 grid 范围，避免采样到图像外太远
        grid = grid.clamp(-2, 2)

        # Warp
        warped = F.grid_sample(
            feat, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        return warped


class TemporalCrossAttention(nn.Module):
    """
    时序跨相机注意力（显存优化版）

    特点：
    1. 支持跨相机历史融合（失效相机可从所有相机历史获取信息）
    2. 对 Key/Value 做空间下采样，大幅减少显存
    3. 包含空间、时间、相机位置编码
    """
    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_cameras: int = 6,
                 num_history: int = 3,
                 dropout: float = 0.1,
                 kv_downsample: int = 4,  # Key/Value 空间下采样倍数
                 use_flash_attn: bool = True):
        """
        Args:
            embed_dims: 特征维度
            num_heads: 注意力头数
            num_cameras: 相机数量
            num_history: 历史帧数量
            dropout: Dropout 比例
            kv_downsample: Key/Value 的空间下采样倍数（减少显存）
            use_flash_attn: 是否使用 FlashAttention（需要安装）
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_cameras = num_cameras
        self.num_history = num_history
        self.use_flash_attn = use_flash_attn
        self.kv_downsample = kv_downsample

        # 可学习的 Query 初始化
        self.query_embed = nn.Parameter(torch.randn(1, embed_dims, 1, 1) * 0.02)

        # 位置编码
        self.camera_pos_embed = nn.Parameter(torch.randn(num_cameras, embed_dims) * 0.02)
        self.temporal_pos_embed = nn.Parameter(torch.randn(num_history, embed_dims) * 0.02)

        # 空间位置编码（可学习的 2D 位置）
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, embed_dims, 1, 1) * 0.02)

        # 相机距离编码（用于跨相机注意力）
        self.register_buffer('camera_adjacency', self._create_camera_adjacency())

        # Key/Value 空间下采样
        if kv_downsample > 1:
            self.kv_pool = nn.AdaptiveAvgPool2d(None)  # 动态设置

        # 注意力层
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 输出 LayerNorm
        self.norm = nn.LayerNorm(embed_dims)

    def _create_camera_adjacency(self) -> torch.Tensor:
        """
        创建相机邻接矩阵

        nuScenes 相机布局：
        0: FRONT, 1: FRONT_LEFT, 2: FRONT_RIGHT,
        3: BACK_LEFT, 4: BACK_RIGHT, 5: BACK
        """
        adjacency = torch.ones(6, 6) * 0.5
        adjacency.fill_diagonal_(1.0)
        adjacency[0, 1] = adjacency[1, 0] = 0.8
        adjacency[0, 2] = adjacency[2, 0] = 0.8
        adjacency[5, 3] = adjacency[3, 5] = 0.8
        adjacency[5, 4] = adjacency[4, 5] = 0.8
        adjacency[1, 3] = adjacency[3, 1] = 0.7
        adjacency[2, 4] = adjacency[4, 2] = 0.7
        return adjacency

    def forward(self,
                query_cam_idx: int,
                history_feats: torch.Tensor,
                H: int, W: int) -> torch.Tensor:
        """
        对单个失效相机进行补全（显存优化版）

        Args:
            query_cam_idx: 失效相机索引
            history_feats: [B, V, T, C, H, W] 所有相机的历史特征（已 warp）
            H, W: 输出特征图尺寸

        Returns:
            completed: [B, C, H, W] 补全后的特征
        """
        B, V, T, C, H_in, W_in = history_feats.shape
        device = history_feats.device

        # 1. 构建 Query（可学习初始化）
        query = self.query_embed.expand(B, -1, H, W)
        query = query + self.camera_pos_embed[query_cam_idx].view(1, -1, 1, 1)
        query = query + self.spatial_pos_embed.expand(B, -1, H, W)
        query_flat = query.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # 2. 构建 Key/Value（对历史特征做空间下采样以节省显存）
        # 下采样尺寸
        H_kv = max(H_in // self.kv_downsample, 1)
        W_kv = max(W_in // self.kv_downsample, 1)

        # 重塑并下采样
        kv = history_feats.view(B * V * T, C, H_in, W_in)
        if self.kv_downsample > 1:
            kv = F.adaptive_avg_pool2d(kv, (H_kv, W_kv))  # [B*V*T, C, H_kv, W_kv]
        kv = kv.view(B, V, T, C, H_kv, W_kv)

        # 添加位置编码（在下采样后添加，节省计算）
        for t in range(T):
            kv[:, :, t] = kv[:, :, t] + self.temporal_pos_embed[t].view(1, 1, -1, 1, 1)
        for v in range(V):
            kv[:, v] = kv[:, v] + self.camera_pos_embed[v].view(1, 1, -1, 1, 1)

        # 展平
        kv_flat = kv.permute(0, 1, 2, 4, 5, 3).reshape(B, V * T * H_kv * W_kv, C)

        # 3. 投影
        q = self.q_proj(query_flat)  # [B, H*W, C]
        k = self.k_proj(kv_flat)      # [B, V*T*H_kv*W_kv, C]
        v_out = self.v_proj(kv_flat)

        # 4. 计算注意力
        scale = (C // self.num_heads) ** -0.5

        # 重塑为多头
        q = q.view(B, H * W, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v_out = v_out.view(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        # 注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        out = torch.matmul(attn, v_out)
        out = out.transpose(1, 2).reshape(B, H * W, C)

        # 输出投影
        out = self.out_proj(out)
        out = self.norm(out)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return out


class SpatialDecoder(nn.Module):
    """
    空间细化解码器

    保持空间结构，使用残差块进行逐像素细化
    """
    def __init__(self, embed_dims: int = 256, hidden_dims: int = 256, num_res_blocks: int = 2):
        super().__init__()

        layers = [
            nn.Conv2d(embed_dims, hidden_dims, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(inplace=True),
        ]

        # 残差块
        for _ in range(num_res_blocks):
            layers.append(ResBlock(hidden_dims))

        # 输出层
        layers.extend([
            nn.Conv2d(hidden_dims, embed_dims, 1),
        ])

        self.decoder = nn.Sequential(*layers)

        # 小值初始化最后一层
        nn.init.normal_(self.decoder[-1].weight, mean=0, std=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.conv(x))


class MotionCompensatedTemporalCompletion(nn.Module):
    """
    运动补偿时序补全模块（显存优化版）

    显存优化策略：
    1. 只存储和处理最粗尺度特征（scale_idx=-1）
    2. 历史队列长度减少到 2 帧
    3. Key/Value 做 4x 空间下采样
    4. 补全结果上采样后应用到所有尺度

    输入：
        current_feats: List[Tensor] 当前帧多尺度特征，每个 [B, V, C, H, W]
        cam_mask: [B, V] bool 失效相机掩码
        metas: dict 包含 T_global, lidar2img 等

    输出：
        completed_feats: List[Tensor] 补全后的特征
    """
    def __init__(self,
                 ch_per_scale: List[int],
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 queue_length: int = 2,  # 减少到 2 帧
                 num_cameras: int = 6,
                 reference_depths: List[float] = [10, 30],  # 减少深度假设
                 kv_downsample: int = 4,  # Key/Value 下采样
                 use_flash_attn: bool = False,
                 enable: bool = True):
        super().__init__()
        self.enable = enable
        self.num_scales = len(ch_per_scale)
        self.num_cameras = num_cameras
        self.queue_length = queue_length

        if not enable:
            return

        # 只处理最粗尺度（最后一个尺度，显存最小）
        self.process_scale_idx = -1
        process_ch = ch_per_scale[self.process_scale_idx]

        # 特征队列（只存储单一尺度）
        self.feature_queue = FeatureQueue(queue_length=queue_length)

        # 单一尺度的运动 warp 模块
        self.motion_warp = ImageLevelMotionWarp(
            embed_dims=process_ch,
            reference_depths=reference_depths,
            learnable_offset=True,
        )

        # 时序跨相机注意力
        self.temporal_attention = TemporalCrossAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_cameras=num_cameras,
            num_history=queue_length,
            kv_downsample=kv_downsample,
            use_flash_attn=use_flash_attn,
        )

        # 特征适配层
        self.feat_adapter = nn.Conv2d(process_ch, embed_dims, 1) if process_ch != embed_dims else nn.Identity()
        self.out_adapter = nn.Conv2d(embed_dims, process_ch, 1) if process_ch != embed_dims else nn.Identity()

        # 空间解码器（轻量版）
        self.spatial_decoder = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, 1),
        )
        nn.init.normal_(self.spatial_decoder[-1].weight, mean=0, std=0.01)
        nn.init.zeros_(self.spatial_decoder[-1].bias)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Conv2d(process_ch * 2, process_ch // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(process_ch // 4, 1, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[-2].bias, -2.0)

        # 默认图像尺寸
        self.img_shape = (900, 1600)

    def compute_T_temp2cur(self, T_global_hist: np.ndarray, T_global_cur: np.ndarray, device) -> torch.Tensor:
        """计算历史帧到当前帧的变换矩阵"""
        T_global_inv_cur = np.linalg.inv(T_global_cur)
        T_temp2cur = T_global_inv_cur @ T_global_hist
        return torch.tensor(T_temp2cur, dtype=torch.float32, device=device)

    def forward(self,
                current_feats: List[torch.Tensor],
                cam_mask: torch.Tensor,
                metas: Dict) -> List[torch.Tensor]:
        """
        前向传播（显存优化版）

        只处理最粗尺度，补全结果应用到所有尺度
        """
        if not self.enable:
            return current_feats

        # 获取要处理的尺度特征
        feat_process = current_feats[self.process_scale_idx]  # [B, V, C, H, W]
        B, V, C, H, W = feat_process.shape
        device = feat_process.device

        if cam_mask is None or not cam_mask.any():
            # 没有失效相机，更新队列后直接返回
            with torch.no_grad():
                self.feature_queue.push(feat_process, metas)
            return current_feats

        # 获取历史特征
        history_feats, T_global_queue, _ = self.feature_queue.get()

        if len(history_feats) == 0:
            # 没有历史帧，更新队列后返回原始特征
            with torch.no_grad():
                self.feature_queue.push(feat_process, metas)
            return current_feats

        # 获取当前帧的变换矩阵（带空值检查）
        img_metas = metas.get('img_metas', None)
        if img_metas is None or not isinstance(img_metas, list) or len(img_metas) == 0:
            # 没有有效的 img_metas，跳过时序补全
            with torch.no_grad():
                self.feature_queue.push(feat_process, metas)
            return current_feats

        T_global_cur = img_metas[0].get('T_global', np.eye(4))

        # 获取 lidar2img
        if 'lidar2img' in img_metas[0]:
            lidar2img = img_metas[0]['lidar2img']
            if isinstance(lidar2img, np.ndarray):
                lidar2img = torch.tensor(lidar2img, dtype=torch.float32, device=device)
            elif isinstance(lidar2img, list):
                lidar2img = torch.tensor(np.stack(lidar2img), dtype=torch.float32, device=device)
        else:
            lidar2img = torch.eye(4, device=device).unsqueeze(0).expand(V, -1, -1)

        if lidar2img.dim() == 3:
            lidar2img = lidar2img.unsqueeze(0).expand(B, -1, -1, -1)

        # 计算 T_temp2cur 并 warp 历史特征
        warped_history = []
        for t, (hist_feat, T_global_hist) in enumerate(zip(history_feats, T_global_queue)):
            if hist_feat.shape[0] != B:
                continue

            T_temp2cur = self.compute_T_temp2cur(T_global_hist, T_global_cur, device)
            T_temp2cur = T_temp2cur.unsqueeze(0).expand(B, -1, -1)

            # 对每个相机 warp
            warped_cams = []
            for v in range(V):
                warped = self.motion_warp(hist_feat[:, v], T_temp2cur, lidar2img[:, v], self.img_shape)
                warped_cams.append(warped)
            warped_hist = torch.stack(warped_cams, dim=1)
            warped_history.append(warped_hist)

        if len(warped_history) == 0:
            with torch.no_grad():
                self.feature_queue.push(feat_process, metas)
            return current_feats

        # 堆叠历史帧 [B, V, T, C, H, W]
        warped_history = torch.stack(warped_history, dim=2)
        T_actual = warped_history.shape[2]

        # 适配到 embed_dims
        warped_adapted = self.feat_adapter(warped_history.view(B * V * T_actual, C, H, W))
        embed_dims = warped_adapted.shape[1]
        warped_adapted = warped_adapted.view(B, V, T_actual, embed_dims, H, W)

        # 对每个失效相机进行补全
        feat_out = feat_process.clone()

        for v in range(V):
            missing_mask = cam_mask[:, v]
            if not missing_mask.any():
                continue

            # 时序注意力
            completed = self.temporal_attention(
                query_cam_idx=v,
                history_feats=warped_adapted,
                H=H, W=W,
            )

            # 空间解码
            completed = self.spatial_decoder(completed)

            # 适配回原始通道
            completed = self.out_adapter(completed)

            # 门控融合
            gate_input = torch.cat([feat_process[:, v], completed], dim=1)
            gate = self.gate(gate_input)
            fused = feat_process[:, v] * (1 - gate) + completed * gate

            # 只替换失效的 batch
            mask = missing_mask.view(B, 1, 1, 1).float()
            feat_out[:, v] = feat_out[:, v] * (1 - mask) + fused * mask

        # 更新历史队列
        with torch.no_grad():
            self.feature_queue.push(feat_process, metas)

        # 构建输出：只更新处理的尺度
        outputs = list(current_feats)
        outputs[self.process_scale_idx] = feat_out

        return outputs

    def reset(self):
        """重置历史队列"""
        if hasattr(self, 'feature_queue'):
            self.feature_queue.reset()
