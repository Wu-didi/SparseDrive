import copy
import math
import warnings
from inspect import signature
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except Exception:
    DAF_VALID = False

__all__ = ["SparseDrive"]

from einops import rearrange  # 目前未用到，保留
from .temporal_completion import MotionCompensatedTemporalCompletion


# ================================
# 1) 随机相机失效（可复现）
# ================================
class RandCamMask(torch.nn.Module):
    r"""
    随机 blackout 相机张量，可复现
      Args
        p_missing : float   每条样本触发缺失概率 (0~1)
        n_min,n_max: int    一次 blackout 的相机数范围 [n_min, n_max]
        train_only: bool    True=>仅训练期启用，False=>推理期也启用
        seed      : int     随机种子，保证遮挡可复现
    Input / Output
        x :  [B, V, 3, H, W]  float32  (0~1)
    Return
        (x_masked, cam_mask)  其中 cam_mask: [B, V] bool，True=该相机被遮
    """
    def __init__(self,
                 p_missing=0.5,
                 n_min=1,
                 n_max=2,
                 train_only=True,
                 seed: int = 42,
                 sticky_fault: bool = False,
                 sticky_min_frames: int = 2,
                 sticky_max_frames: int = 6,
                 curriculum_steps: int = 0,
                 p_missing_start: float = None,
                 n_max_start: int = None,
                 state_cache_size: int = 256):
        super().__init__()
        assert 0.0 <= p_missing <= 1.0
        assert 1 <= n_min <= n_max
        assert sticky_min_frames >= 1
        assert sticky_min_frames <= sticky_max_frames
        self.p_missing = p_missing
        self.n_min = n_min
        self.n_max = n_max
        self.train_only = train_only
        self._seed = seed
        self.sticky_fault = sticky_fault
        self.sticky_min_frames = sticky_min_frames
        self.sticky_max_frames = sticky_max_frames
        self.curriculum_steps = max(int(curriculum_steps), 0)
        self.p_missing_start = p_missing if p_missing_start is None else float(p_missing_start)
        self.n_max_start = n_min if n_max_start is None else int(n_max_start)
        self.state_cache_size = int(state_cache_size)
        self._global_step = 0

        # 独立生成器，确保复现且不扰动全局 RNG
        self._g = torch.Generator()
        self._g.manual_seed(seed)
        self._scene_fault_state = OrderedDict()

    def _sample_cam_mask(self, n_cam, n_max, device):
        n_drop = torch.randint(
            self.n_min,
            n_max + 1,
            (1,),
            generator=self._g,
            device=device,
        ).item()
        drop_ids = torch.randperm(
            n_cam,
            generator=self._g,
            device=device,
        )[:n_drop]
        cam_mask = torch.zeros(n_cam, device=device, dtype=torch.bool)
        cam_mask[drop_ids] = True
        return cam_mask

    def _extract_scene_tokens(self, metas, batch_size):
        scene_tokens = [None for _ in range(batch_size)]
        img_metas = metas
        if isinstance(metas, dict):
            img_metas = metas.get("img_metas", None)
        if isinstance(img_metas, list):
            for i in range(min(batch_size, len(img_metas))):
                token = img_metas[i].get("scene_token", None)
                scene_tokens[i] = str(token) if token is not None else None
        return scene_tokens

    def _get_drop_policy(self, n_cam):
        p_missing = self.p_missing
        n_max = min(self.n_max, n_cam)
        if self.training and self.curriculum_steps > 0:
            ratio = min(1.0, self._global_step / float(self.curriculum_steps))
            p_missing = self.p_missing_start + ratio * (self.p_missing - self.p_missing_start)
            n_max_curr = int(round(self.n_max_start + ratio * (self.n_max - self.n_max_start)))
            n_max = min(max(self.n_min, n_max_curr), n_cam)
        p_missing = float(max(0.0, min(1.0, p_missing)))
        return p_missing, n_max

    def _get_cached_mask(self, scene_token, device):
        state = self._scene_fault_state.get(scene_token, None)
        if state is None:
            return None
        remain = state["remain"]
        if remain <= 0:
            self._scene_fault_state.pop(scene_token, None)
            return None
        state["remain"] = remain - 1
        self._scene_fault_state.move_to_end(scene_token)
        return state["mask"].to(device=device, dtype=torch.bool)

    def _set_cached_mask(self, scene_token, cam_mask):
        duration = torch.randint(
            self.sticky_min_frames,
            self.sticky_max_frames + 1,
            (1,),
            generator=self._g,
            device=cam_mask.device,
        ).item()
        self._scene_fault_state[scene_token] = {
            "mask": cam_mask.detach().cpu(),
            "remain": max(duration - 1, 0),
        }
        self._scene_fault_state.move_to_end(scene_token)
        while len(self._scene_fault_state) > self.state_cache_size:
            self._scene_fault_state.popitem(last=False)

    @torch.no_grad()
    def forward(self, x, return_mask: bool = False, metas=None):
        if self.train_only and not self.training:
            return (x, None) if return_mask else x

        B, N_cam, C, H, W = x.shape
        device = x.device
        if self.training:
            self._global_step += 1

        # 生成器设备对齐（避免 cpu/gpu 不匹配报错）
        if getattr(self._g, "device", torch.device("cpu")).type != device.type:
            self._g = torch.Generator(device=device)
            self._g.manual_seed(self._seed)

        p_missing, n_max = self._get_drop_policy(N_cam)
        scene_tokens = self._extract_scene_tokens(metas, B) if self.sticky_fault else [None] * B
        cam_mask = torch.zeros(B, N_cam, device=device, dtype=torch.bool)
        for b in range(B):
            scene_token = scene_tokens[b]
            if self.sticky_fault and scene_token is not None:
                cached_mask = self._get_cached_mask(scene_token, device)
                if cached_mask is not None:
                    cam_mask[b] = cached_mask
                    continue

            trigger = torch.rand(1, generator=self._g, device=device).item() < p_missing
            if not trigger:
                continue

            sampled = self._sample_cam_mask(N_cam, n_max, device)
            cam_mask[b] = sampled
            if self.sticky_fault and scene_token is not None:
                self._set_cached_mask(scene_token, sampled)

        # 应用遮挡（被遮挡视角直接置 0）
        x = x * (~cam_mask).view(B, N_cam, 1, 1, 1).to(x.dtype)
        return (x, cam_mask) if return_mask else x


# ===========================================
# 2) 视角级特征补全（VAE / Flow Matching）
# ===========================================
class SimpleFeatureVAE(nn.Module):
    """
    一个轻量卷积 VAE，用在单尺度特征图上：
      输入:  [B, C, H, W]
      输出:  [B, C, H, W]  (重建特征), 以及 mu, logvar
    """
    def __init__(self, in_channels, latent_channels=64):
        super().__init__()
        # encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )
        # 变分瓶颈
        self.mu_conv = nn.Conv2d(latent_channels, latent_channels, 1, bias=True)
        self.logvar_conv = nn.Conv2d(latent_channels, latent_channels, 1, bias=True)

        # decoder
        self.dec = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, in_channels, 1, bias=True),
        )

    def encode(self, x):
        h = self.enc(x)
        mu = self.mu_conv(h)
        logvar = self.logvar_conv(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return:
          x_rec: [B, C, H, W]
          mu, logvar: [B, C_latent, H, W]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar


class PVReconVAE(nn.Module):
    """
    多尺度、多视角 VAE 补全模块

    输入:
      feature_maps: List[Tensor]，每尺度 [B, V, C, H, W]
      cam_mask    : [B, V] bool，True=该视角缺失

    输出:
      outs        : List[Tensor]，与 feature_maps 同形状，
                    cam_mask=True 的位置用 VAE 重建特征替换
      loss_dict   : {'loss_pv_vae_rec': ..., 'loss_pv_vae_kl': ...}
    """
    def __init__(self,
                 ch_per_scale,
                 latent_channels=64,
                 lambda_rec=1.0,
                 lambda_kl=1e-4):
        super().__init__()
        assert isinstance(ch_per_scale, (list, tuple))
        self.vaes = nn.ModuleList([
            SimpleFeatureVAE(c, latent_channels=latent_channels)
            for c in ch_per_scale
        ])
        self.lambda_rec = lambda_rec
        self.lambda_kl = lambda_kl

    def forward(
        self,
        feature_maps,
        cam_mask,
        metas=None,
        camera_weights=None,
        target_feature_maps=None,
    ):
        assert cam_mask is not None, "PVReconVAE 需要 cam_mask（可以全 0）"
        B_mask, V_mask = cam_mask.shape
        outs = []

        total_rec_loss = 0.0
        total_kl_loss = 0.0

        for i, Fm in enumerate(feature_maps):
            # Fm: [B, V, C, H, W]
            B, V, C, H, W = Fm.shape
            assert B == B_mask and V == V_mask, "cam_mask 维度需与特征一致"

            # 展平视角维度，喂入 VAE
            F_in = Fm.reshape(B * V, C, H, W)           # [B*V, C, H, W]
            F_detach = F_in.detach()                 # 只训练 VAE，不回传到 backbone

            # 用 detached 输入训练补全器，避免 VAE loss 反向拉扯 backbone
            x_rec, mu, logvar = self.vaes[i](F_detach)   # [B*V, C, H, W], [B*V, C_lat, H, W]

            # ---------- VAE 损失 ----------
            # 仅在缺失相机上统计重建/KL，target 优先使用 full/teacher 特征
            target = F_detach
            if target_feature_maps is not None:
                target_fm = target_feature_maps[i]
                if target_fm.shape != Fm.shape:
                    raise ValueError("target_feature_maps scale shape mismatch")
                target = target_fm.detach().reshape(B * V, C, H, W)

            rec_per_view = (x_rec - target).pow(2).reshape(B, V, -1).mean(dim=-1)  # [B, V]
            kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
            kl_per_view = kl.reshape(B, V, -1).mean(dim=-1)  # [B, V]

            miss = cam_mask.to(rec_per_view.device, dtype=rec_per_view.dtype)
            has_missing = bool(miss.any().item())
            if has_missing:
                weight = miss

                # 可选：按相机重要性做加权（[B, V]）
                if camera_weights is not None:
                    cam_w = camera_weights.to(weight.device, dtype=weight.dtype)
                    if cam_w.shape != weight.shape:
                        raise ValueError("camera_weights shape should be [B, V]")
                    weight = weight * (1.0 + cam_w.clamp(min=0.0))

                normalizer = weight.sum().clamp(min=1.0)
                rec_loss = (rec_per_view * weight).sum() / normalizer
                kl_loss = (kl_per_view * weight).sum() / normalizer
            else:
                rec_loss = rec_per_view.new_zeros(())
                kl_loss = kl_per_view.new_zeros(())

            total_rec_loss = total_rec_loss + rec_loss
            total_kl_loss = total_kl_loss + kl_loss

            # ---------- 视角级补全 ----------
            F_rec = x_rec.reshape(B, V, C, H, W)
            miss = cam_mask.view(B, V, 1, 1, 1).to(Fm.dtype)  # 1=缺失
            F_out = Fm * (1.0 - miss) + F_rec * miss

            outs.append(F_out)

        loss_dict = {
            "loss_pv_vae_rec": self.lambda_rec * total_rec_loss,
            "loss_pv_vae_kl": self.lambda_kl * total_kl_loss,
        }
        return outs, loss_dict


def _build_group_norm(num_channels, max_groups=8):
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dims=128):
        super().__init__()
        if embed_dims % 2 != 0:
            raise ValueError("SinusoidalTimeEmbedding requires an even embed_dims")
        self.embed_dims = embed_dims

    def forward(self, t):
        t = t.reshape(-1, 1)
        half = self.embed_dims // 2
        if half <= 0:
            raise ValueError("embed_dims must be >= 2")
        if half == 1:
            freq = torch.ones(1, device=t.device, dtype=t.dtype)
        else:
            freq = torch.exp(
                torch.arange(half, device=t.device, dtype=t.dtype)
                * (-math.log(10000.0) / float(half - 1))
            )
        angles = t * freq.unsqueeze(0)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class SimpleFeatureFlow(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        time_embed_dim=128,
        num_cameras=6,
        use_camera_embed=True,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.camera_embed = (
            nn.Embedding(num_cameras, hidden_channels)
            if use_camera_embed
            else None
        )

        input_channels = in_channels * 3 + 1
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1, bias=False),
            _build_group_norm(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            _build_group_norm(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            _build_group_norm(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1),
        )

    def forward(self, x_t, source, context, miss_mask, time, camera_ids=None):
        hidden = self.stem(torch.cat([x_t, source, context, miss_mask], dim=1))
        time_feat = self.time_proj(self.time_embed(time)).unsqueeze(-1).unsqueeze(-1)
        time_feat = time_feat.to(hidden.dtype)
        hidden = hidden + time_feat
        if self.camera_embed is not None and camera_ids is not None:
            camera_feat = self.camera_embed(camera_ids).unsqueeze(-1).unsqueeze(-1)
            hidden = hidden + camera_feat.to(hidden.dtype)
        return self.head(hidden)


class FeatureFlowReconstructor(nn.Module):
    """
    多尺度、多视角条件 flow matching 重建模块。

    输入:
      feature_maps: List[Tensor]，每尺度 [B, V, C, H, W]
      cam_mask    : [B, V] bool，True=该视角缺失

    输出:
      outs        : List[Tensor]，与 feature_maps 同形状，
                    cam_mask=True 的位置用 flow 重建特征替换
      loss_dict   : {'loss_pv_flow': ...}
    """

    def __init__(
        self,
        ch_per_scale,
        hidden_channels=128,
        time_embed_dim=128,
        lambda_flow=0.01,
        num_integration_steps=4,
        num_cameras=6,
        use_camera_embed=True,
        detach_inputs=True,
    ):
        super().__init__()
        if not isinstance(ch_per_scale, (list, tuple)):
            raise TypeError("ch_per_scale should be a list or tuple")
        self.predictors = nn.ModuleList(
            [
                SimpleFeatureFlow(
                    c,
                    hidden_channels=hidden_channels,
                    time_embed_dim=time_embed_dim,
                    num_cameras=num_cameras,
                    use_camera_embed=use_camera_embed,
                )
                for c in ch_per_scale
            ]
        )
        self.lambda_flow = lambda_flow
        self.num_integration_steps = max(int(num_integration_steps), 1)
        self.detach_inputs = detach_inputs

    @staticmethod
    def _flatten_views(feat):
        B, V, C, H, W = feat.shape
        return feat.reshape(B * V, C, H, W)

    @staticmethod
    def _build_camera_ids(batch_size, num_cameras, device):
        return (
            torch.arange(num_cameras, device=device, dtype=torch.long)
            .view(1, num_cameras)
            .expand(batch_size, -1)
            .reshape(-1)
        )

    def _build_context(self, feat, cam_mask):
        visible = (~cam_mask).to(feat.dtype).view(feat.shape[0], feat.shape[1], 1, 1, 1)
        denom = visible.sum(dim=1, keepdim=True).clamp(min=1.0)
        context = (feat * visible).sum(dim=1, keepdim=True) / denom
        return context.expand_as(feat)

    def _build_loss_weight(self, cam_mask, camera_weights, dtype, device):
        weight = cam_mask.to(device=device, dtype=dtype)
        if camera_weights is not None:
            cam_w = camera_weights.to(device=device, dtype=dtype)
            if cam_w.shape != weight.shape:
                raise ValueError("camera_weights shape should be [B, V]")
            weight = weight * (1.0 + cam_w.clamp(min=0.0))
        return weight

    def _integrate(self, predictor, source, context, camera_ids):
        state = source.clone()
        miss_mask = source.new_ones((source.shape[0], 1, source.shape[2], source.shape[3]))
        dt = 1.0 / float(self.num_integration_steps)
        for step in range(self.num_integration_steps):
            time = source.new_full((source.shape[0],), step * dt)
            velocity = predictor(state, source, context, miss_mask, time, camera_ids)
            state = state + velocity * dt
        return state

    def forward(
        self,
        feature_maps,
        cam_mask,
        metas=None,
        camera_weights=None,
        target_feature_maps=None,
    ):
        assert cam_mask is not None, "FeatureFlowReconstructor requires cam_mask"
        cam_mask = cam_mask.to(dtype=torch.bool)
        B_mask, V_mask = cam_mask.shape
        outs = []

        total_flow_loss = feature_maps[0].new_zeros(())

        for i, Fm in enumerate(feature_maps):
            B, V, C, H, W = Fm.shape
            if B != B_mask or V != V_mask:
                raise ValueError("cam_mask shape must match feature maps")

            source = Fm.detach() if self.detach_inputs else Fm
            target = source
            if target_feature_maps is not None:
                target_fm = target_feature_maps[i]
                if target_fm.shape != Fm.shape:
                    raise ValueError("target_feature_maps scale shape mismatch")
                target = target_fm.detach()

            miss = cam_mask.to(Fm.device)
            if miss.any():
                context = self._build_context(source, miss)
                flat_source = self._flatten_views(source)
                flat_target = self._flatten_views(target)
                flat_context = self._flatten_views(context)
                flat_feat = self._flatten_views(Fm)
                miss_flat = miss.reshape(B * V)
                camera_ids = self._build_camera_ids(B, V, Fm.device)[miss_flat]

                source_miss = flat_source[miss_flat]
                target_miss = flat_target[miss_flat]
                context_miss = flat_context[miss_flat]
                time = torch.rand(source_miss.shape[0], device=Fm.device, dtype=Fm.dtype)
                x_t = torch.lerp(
                    source_miss,
                    target_miss,
                    time.view(-1, 1, 1, 1),
                )
                miss_map = source_miss.new_ones((source_miss.shape[0], 1, H, W))
                pred_velocity = self.predictors[i](
                    x_t,
                    source_miss,
                    context_miss,
                    miss_map,
                    time,
                    camera_ids,
                )
                target_velocity = target_miss - source_miss
                loss_per_view = (pred_velocity - target_velocity).pow(2).flatten(1).mean(dim=1)

                view_weight = self._build_loss_weight(
                    miss,
                    camera_weights,
                    dtype=Fm.dtype,
                    device=Fm.device,
                ).reshape(B * V)[miss_flat]
                normalizer = view_weight.sum().clamp(min=1.0)
                total_flow_loss = total_flow_loss + (loss_per_view * view_weight).sum() / normalizer

                recon_miss = self._integrate(
                    self.predictors[i],
                    source_miss,
                    context_miss,
                    camera_ids,
                )
                flat_feat = flat_feat.clone()
                flat_feat[miss_flat] = recon_miss
                F_out = flat_feat.reshape(B, V, C, H, W)
            else:
                F_out = Fm

            outs.append(F_out)

        return outs, {"loss_pv_flow": self.lambda_flow * total_flow_loss}


class IdentityPVReconstructor(nn.Module):
    def forward(
        self,
        feature_maps,
        cam_mask,
        metas=None,
        camera_weights=None,
        target_feature_maps=None,
    ):
        return feature_maps, {}


def build_pv_reconstructor(cfg=None):
    cfg = copy.deepcopy(cfg) if cfg is not None else {}
    recon_type = cfg.pop("type", "vae")
    default_channels = [256, 256, 256, 256]

    if recon_type in ("identity", "none", None):
        return IdentityPVReconstructor()

    if recon_type == "vae":
        defaults = dict(
            ch_per_scale=default_channels,
            latent_channels=64,
            lambda_rec=0.01,
            lambda_kl=1e-4,
        )
        defaults.update(cfg)
        return PVReconVAE(**defaults)

    if recon_type == "flow":
        defaults = dict(
            ch_per_scale=default_channels,
            hidden_channels=128,
            time_embed_dim=128,
            lambda_flow=0.01,
            num_integration_steps=4,
            num_cameras=6,
            use_camera_embed=True,
            detach_inputs=True,
        )
        defaults.update(cfg)
        return FeatureFlowReconstructor(**defaults)

    raise ValueError(f"Unsupported pv_recon type: {recon_type}")


# ===========================================
# 时序补全模块：利用历史帧预测当前缺失相机
# ===========================================
class TemporalFeatureCompletion(nn.Module):
    """
    时序补全模块

    核心思想：
    - 端到端模型有历史帧信息（queue_length=4）
    - 当相机失效时，用历史轨迹预测当前帧的特征
    - 比单帧VAE更合理，利用了时序连续性

    输入：
        current_feats: List[Tensor] 当前帧特征 [[B,V,C,H,W], ...]
        history_queue: List[List[Tensor]] 历史帧特征队列
        cam_mask: [B, V] bool，True=该相机失效

    输出：
        补全后的特征，与current_feats同结构
    """
    def __init__(self,
                 ch_per_scale,
                 hidden_dim=128,
                 num_layers=2,
                 enable=True):
        super().__init__()
        self.enable = enable
        if not enable:
            return

        assert isinstance(ch_per_scale, (list, tuple))
        self.num_scales = len(ch_per_scale)

        # 为每个尺度创建一个GRU预测器
        self.predictors = nn.ModuleList()
        for c in ch_per_scale:
            # 输入：时序特征，输出：预测的当前帧特征
            predictor = nn.GRU(
                input_size=c,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0.0
            )
            self.predictors.append(predictor)

        # 解码器：从hidden state到特征
        self.decoders = nn.ModuleList()
        for c in ch_per_scale:
            decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, c),
            )
            self.decoders.append(decoder)

    def forward(self, current_feats, history_queue, cam_mask):
        """
        current_feats: List[Tensor]，每个 [B, V, C, H, W]
        history_queue: List[List[Tensor]]，外层是时间步，内层是尺度
                       例如：[[scale0_t0, scale1_t0, ...], [scale0_t1, ...], ...]
        cam_mask: [B, V] bool
        """
        if not self.enable:
            return current_feats

        if cam_mask is None or not cam_mask.any():
            return current_feats

        # 如果没有历史帧，无法做时序预测
        if history_queue is None or len(history_queue) == 0:
            return current_feats

        outputs = []

        for scale_idx, feat_cur in enumerate(current_feats):
            B, V, C, H, W = feat_cur.shape
            feat_out = feat_cur.clone()

            # 提取这个尺度的历史特征
            history_feats = []
            for t in range(len(history_queue)):
                if scale_idx < len(history_queue[t]):
                    history_feats.append(history_queue[t][scale_idx])

            if len(history_feats) == 0:
                outputs.append(feat_out)
                continue

            # 对每个相机单独处理
            for v in range(V):
                # 找出这个相机失效的batch
                cam_v_mask = cam_mask[:, v]  # [B]
                if not cam_v_mask.any():
                    continue

                # 提取历史序列 [T, B, C, H, W]
                hist_seq = [h[:, v] for h in history_feats]  # List of [B, C, H, W]

                # 过滤掉batch size不匹配的历史帧
                hist_seq_filtered = []
                for h in hist_seq:
                    if h.shape[0] == B:  # 只保留batch size匹配的
                        hist_seq_filtered.append(h)

                if len(hist_seq_filtered) == 0:
                    # 没有匹配的历史帧，跳过这个相机
                    continue

                hist_seq = torch.stack(hist_seq_filtered, dim=0)  # [T, B, C, H, W]
                T = hist_seq.shape[0]

                # 全局平均池化：[T, B, C, H, W] -> [T, B, C]
                hist_seq_pooled = hist_seq.mean(dim=(-1, -2))  # [T, B, C]
                hist_seq_pooled = hist_seq_pooled.permute(1, 0, 2)  # [B, T, C]

                # GRU预测
                gru_out, hidden = self.predictors[scale_idx](hist_seq_pooled)  # [B, T, H]

                # 取最后一个时间步的hidden state
                last_hidden = hidden[-1]  # [B, H]

                # 解码到特征空间
                predicted_feat = self.decoders[scale_idx](last_hidden)  # [B, C]

                # 扩展到空间维度（简单broadcast）
                predicted_feat = predicted_feat.view(B, C, 1, 1).expand(B, C, H, W)

                # 只替换失效的相机
                mask_v = cam_v_mask.view(B, 1, 1, 1).float()
                feat_out[:, v] = feat_out[:, v] * (1 - mask_v) + predicted_feat * mask_v

            outputs.append(feat_out)

        return outputs


# ===========================================
# 规划导向的特征加权模块
# ===========================================
class PlanningGuidedWeighting(nn.Module):
    """
    规划导向的特征重要性加权

    核心思想：
    - 不是所有相机区域都同等重要
    - 前视相机对规划最关键，后视相机影响较小
    - 自车速度越快，前方越重要

    nuScenes相机布局（6个相机）：
    0: CAM_FRONT - 前视（最重要）
    1: CAM_FRONT_LEFT - 前左
    2: CAM_FRONT_RIGHT - 前右
    3: CAM_BACK_LEFT - 后左
    4: CAM_BACK_RIGHT - 后右
    5: CAM_BACK - 后视（最不重要）
    """
    def __init__(self,
                 num_cameras=6,
                 use_ego_state=True,
                 base_weights=None):
        super().__init__()
        self.num_cameras = num_cameras
        self.use_ego_state = use_ego_state

        # 基础权重（可配置）
        if base_weights is None:
            # 默认权重：前视最高，后视最低
            self.register_buffer('base_weights', torch.tensor([
                2.0,   # CAM_FRONT
                1.5,   # CAM_FRONT_LEFT
                1.5,   # CAM_FRONT_RIGHT
                1.0,   # CAM_BACK_LEFT
                1.0,   # CAM_BACK_RIGHT
                0.5,   # CAM_BACK
            ]))
        else:
            self.register_buffer('base_weights', torch.tensor(base_weights))

        # 根据自车状态调整权重（可选）
        if use_ego_state:
            # 简单的MLP：ego_state -> 权重调整系数
            self.ego_adapter = nn.Sequential(
                nn.Linear(3, 16),  # 输入：速度、加速度、角速度
                nn.ReLU(inplace=True),
                nn.Linear(16, num_cameras),
                nn.Sigmoid(),  # 输出 [0, 1]，用于调整base_weights
            )
        else:
            self.ego_adapter = None

    def forward(self, cam_mask, ego_state=None):
        """
        cam_mask: [B, V] bool，True=失效
        ego_state: [B, 3] 可选，自车状态 (速度, 加速度, 角速度)

        返回：
            importance_weights: [B, V] 每个相机的重要性权重
        """
        B, V = cam_mask.shape
        device = cam_mask.device

        # 基础权重
        weights = self.base_weights[:V].unsqueeze(0).expand(B, V).clone()  # [B, V]

        # 根据ego状态动态调整（可选）
        if self.ego_adapter is not None and ego_state is not None:
            ego_adj = self.ego_adapter(ego_state)  # [B, V]
            # 速度越快，前方权重增加更多
            weights = weights * (1.0 + ego_adj)

        # 归一化（保持总权重和不变）
        weights = weights / weights.mean(dim=1, keepdim=True) * V

        return weights

    def get_spatial_importance(self, V, H, W, device='cuda'):
        """
        返回空间重要性图 [V, H, W]

        在图像空间中，中心区域通常比边缘更重要
        """
        importance = torch.ones(V, H, W, device=device)

        # 为每个相机创建中心加权的importance map
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        # 中心权重高，边缘权重低
        spatial_weight = torch.exp(-(x**2 + y**2) / 0.5)  # 高斯权重

        # 应用到每个相机（可以为不同相机设置不同的spatial pattern）
        for v in range(V):
            camera_weight = self.base_weights[v].item()
            importance[v] = spatial_weight * camera_weight

        return importance


# ===========================================
# 规划引导的特征补全模块（创新点）
# ===========================================
class PlanningGuidedCompletion(nn.Module):
    """
    规划引导的特征补全

    核心创新：
    1. 补全目标不仅是重建，更重要的是提升规划性能
    2. 规划轨迹经过的区域需要更精确的补全
    3. 规划损失可以反传到补全模块，端到端优化

    输入：
        feats: List[Tensor], 每个 [B, V, C, H, W] - 被遮挡后的特征
        cam_mask: [B, V] bool - 遮挡掩码
        ego_trajectory: [B, T, 2] 可选 - 规划轨迹 (x, y)
        cam_params: dict 可选 - 相机参数

    输出：
        completed_feats: List[Tensor] - 补全后的特征（梯度可从规划反传）
        importance_maps: List[Tensor] - 各区域重要性（用于加权损失）
    """
    def __init__(self,
                 ch_per_scale,
                 hidden_dim=256,
                 use_trajectory_guidance=True,
                 use_cross_camera=True,
                 use_conditional_completion=True,
                 enable=True):
        super().__init__()
        self.enable = enable
        self.use_trajectory_guidance = use_trajectory_guidance
        self.use_cross_camera = use_cross_camera
        self.use_conditional_completion = use_conditional_completion
        self.num_scales = len(ch_per_scale)

        # 补全网络（每个尺度独立）
        self.completion_nets = nn.ModuleList()
        for c in ch_per_scale:
            self.completion_nets.append(nn.Sequential(
                nn.Conv2d(c, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, c, 1),
            ))

        # 门控网络：决定补全特征和原始特征的融合比例
        self.gate_nets = nn.ModuleList()
        for c in ch_per_scale:
            self.gate_nets.append(nn.Sequential(
                nn.Conv2d(c * 2, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, 1),
                nn.Sigmoid()
            ))

        # 显式条件补全：missing-view-like + valid-view 条件
        self.conditional_nets = nn.ModuleList()
        for c in ch_per_scale:
            self.conditional_nets.append(nn.Sequential(
                nn.Conv2d(c * 2, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, c, 1),
            ))

        # missing-aware 表征：显式告诉网络该视角是否缺失
        self.valid_view_embeds = nn.ParameterList()
        self.missing_view_embeds = nn.ParameterList()
        for c in ch_per_scale:
            self.valid_view_embeds.append(nn.Parameter(torch.zeros(1, 1, c, 1, 1)))
            self.missing_view_embeds.append(nn.Parameter(torch.zeros(1, 1, c, 1, 1)))

        # 跨相机注意力（利用其他相机的当前帧信息）
        if use_cross_camera:
            self.cross_cam_attention = nn.ModuleList()
            for c in ch_per_scale:
                self.cross_cam_attention.append(
                    CrossCameraAttention(c, num_heads=8)
                )

        # 注：轨迹重要性使用启发式方法计算（基于轨迹方向），不需要可学习参数

        # 精细补全网络（用于轨迹区域）
        self.fine_completion_nets = nn.ModuleList()
        for c in ch_per_scale:
            self.fine_completion_nets.append(nn.Sequential(
                nn.Conv2d(c, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, c, 1),
            ))

        # 初始化：让补全网络初始输出接近零，门控网络初始接近0.5
        self._init_weights()

    def _init_weights(self):
        """初始化权重，确保训练初期稳定"""
        for net in self.completion_nets:
            # 最后一层卷积使用小值初始化（不是零，避免梯度消失）
            if hasattr(net[-1], 'weight'):
                nn.init.normal_(net[-1].weight, mean=0, std=0.01)
            if hasattr(net[-1], 'bias') and net[-1].bias is not None:
                nn.init.zeros_(net[-1].bias)

        for net in self.fine_completion_nets:
            if hasattr(net[-1], 'weight'):
                nn.init.normal_(net[-1].weight, mean=0, std=0.01)
            if hasattr(net[-1], 'bias') and net[-1].bias is not None:
                nn.init.zeros_(net[-1].bias)

        for net in self.conditional_nets:
            if hasattr(net[-1], 'weight'):
                nn.init.normal_(net[-1].weight, mean=0, std=0.01)
            if hasattr(net[-1], 'bias') and net[-1].bias is not None:
                nn.init.zeros_(net[-1].bias)

        for net in self.gate_nets:
            # 门控网络最后一层bias初始化为负值，使初始 sigmoid(x) ≈ 0.1
            # 这样初期主要使用原始特征，补全特征只有小权重
            # sigmoid(-2.2) ≈ 0.1
            if hasattr(net[-2], 'bias') and net[-2].bias is not None:
                nn.init.constant_(net[-2].bias, -2.0)

        for valid_embed, missing_embed in zip(self.valid_view_embeds, self.missing_view_embeds):
            nn.init.zeros_(valid_embed)
            nn.init.normal_(missing_embed, mean=0, std=0.02)

    def forward(self, feats, cam_mask, ego_trajectory=None, cam_params=None):
        """
        前向传播

        注意：不使用 detach()，让梯度可以从规划模块反传回来
        """
        if not self.enable:
            return feats, None

        if cam_mask is None or not cam_mask.any():
            return feats, None

        B, V = cam_mask.shape
        device = cam_mask.device

        # 计算轨迹重要性图（如果提供了轨迹）
        traj_importance = None
        if self.use_trajectory_guidance and ego_trajectory is not None and cam_params is not None:
            traj_importance = self.compute_trajectory_importance(
                ego_trajectory, cam_params, feats[0].shape[-2:]
            )  # [B, V, H, W]

        outputs = []
        importance_maps = []

        for scale_idx, feat in enumerate(feats):
            B, V, C, H, W = feat.shape
            feat_base = feat

            # missing-aware：给每个视角注入缺失状态 embedding
            cam_mask_float = cam_mask.view(B, V, 1, 1, 1).to(dtype=feat.dtype)
            feat = (
                feat_base
                + (1.0 - cam_mask_float) * self.valid_view_embeds[scale_idx]
                + cam_mask_float * self.missing_view_embeds[scale_idx]
            )

            # 1. 跨相机注意力（利用有效相机的信息）
            if self.use_cross_camera:
                feat_cross = self.cross_camera_completion(
                    feat, cam_mask, scale_idx
                )
            else:
                feat_cross = feat

            # 2. 展平处理（base: 保持原始分布；feat_flat: 补全网络输入）
            feat_base_flat = feat_base.view(B * V, C, H, W)
            feat_flat = feat_cross.view(B * V, C, H, W)

            # 3. 基础补全
            completed_coarse = self.completion_nets[scale_idx](feat_flat)

            # 4. 精细补全（用于轨迹区域）
            completed_fine = self.fine_completion_nets[scale_idx](feat_flat)

            # 4.1 显式条件补全：利用 valid-view 的条件统计
            if self.use_conditional_completion:
                valid = (~cam_mask).view(B, V, 1, 1, 1).to(dtype=feat_cross.dtype)
                valid_count = valid.sum(dim=1).clamp(min=1.0)
                cond_view = (feat_cross * valid).sum(dim=1) / valid_count  # [B, C, H, W]
                cond_view = cond_view.unsqueeze(1).expand(-1, V, -1, -1, -1)
                cond_flat = cond_view.reshape(B * V, C, H, W)
                cond_completed = self.conditional_nets[scale_idx](
                    torch.cat([feat_flat, cond_flat], dim=1)
                )
                completed_coarse = completed_coarse + cond_completed
                completed_fine = completed_fine + cond_completed

            # 5. 根据轨迹重要性混合粗糙/精细补全
            if traj_importance is not None:
                # 调整重要性图尺寸
                importance = F.interpolate(
                    traj_importance, size=(H, W), mode='bilinear', align_corners=False
                )  # [B, V, H, W]
                importance_flat = importance.view(B * V, 1, H, W)

                # 混合：轨迹区域用精细补全，其他区域用粗糙补全
                completed = completed_coarse * (1 - importance_flat) + completed_fine * importance_flat
            else:
                completed = completed_coarse
                importance = torch.ones(B, V, H, W, device=device) * 0.5

            # 6. 门控融合（学习原始特征和补全特征的最优混合）
            gate_input = torch.cat([feat_base_flat, completed], dim=1)
            gate = self.gate_nets[scale_idx](gate_input)  # [B*V, 1, H, W]

            fused = feat_base_flat * (1 - gate) + completed * gate

            # 7. 只对被遮挡的相机应用补全
            mask = cam_mask.view(B * V, 1, 1, 1).float()
            output = feat_base_flat * (1 - mask) + fused * mask

            outputs.append(output.view(B, V, C, H, W))
            importance_maps.append(importance)

        return outputs, importance_maps

    def cross_camera_completion(self, feat, cam_mask, scale_idx):
        """
        跨相机注意力补全：缺失相机从其他有效相机获取信息
        """
        B, V, C, H, W = feat.shape

        # 有效相机的mask（未被遮挡）
        valid_mask = ~cam_mask  # [B, V]

        # 使用跨相机注意力
        feat_out = self.cross_cam_attention[scale_idx](feat, valid_mask, cam_mask)

        return feat_out

    def compute_trajectory_importance(self, trajectory, cam_params, feat_size):
        """
        计算规划轨迹在图像上的重要性图

        trajectory: [B, T, 2] 未来轨迹点 (x, y) 在自车坐标系
        cam_params: dict with 'intrinsics', 'extrinsics', 'img2lidars' 等
        feat_size: (H, W) 特征图尺寸

        返回: [B, V, H, W] 每个像素的重要性（轨迹经过的区域重要性高）
        """
        B, T, _ = trajectory.shape
        H, W = feat_size

        # 获取相机数量
        if 'intrinsics' in cam_params:
            V = cam_params['intrinsics'].shape[1] if cam_params['intrinsics'].dim() > 2 else 6
        else:
            V = 6

        device = trajectory.device

        # 简化实现：基于轨迹点的方向生成重要性
        # 前方轨迹 -> 前视相机重要
        # 左转轨迹 -> 左侧相机重要
        # 右转轨迹 -> 右侧相机重要

        importance_maps = []

        for b in range(B):
            traj_b = trajectory[b]  # [T, 2]

            # 计算轨迹的主要方向
            if T > 1:
                traj_direction = traj_b[-1] - traj_b[0]  # [2] 最终位移
                traj_direction = traj_direction / (traj_direction.norm() + 1e-6)
            else:
                traj_direction = torch.tensor([1.0, 0.0], device=device)

            # 根据轨迹方向分配相机重要性
            # nuScenes 相机布局：
            # 0: FRONT, 1: FRONT_LEFT, 2: FRONT_RIGHT, 3: BACK_LEFT, 4: BACK_RIGHT, 5: BACK
            cam_importance = torch.zeros(V, device=device)

            # 前向运动
            forward_score = traj_direction[0].clamp(min=0)  # x > 0 表示前进
            if V > 0:
                cam_importance[0] = forward_score * 2.0  # FRONT
            if V > 1:
                cam_importance[1] = forward_score * 1.0  # FRONT_LEFT
            if V > 2:
                cam_importance[2] = forward_score * 1.0  # FRONT_RIGHT

            # 左转
            left_score = traj_direction[1].clamp(min=0)  # y > 0 表示左转
            if V > 1:
                cam_importance[1] += left_score * 1.5  # FRONT_LEFT
            if V > 3:
                cam_importance[3] += left_score * 1.0  # BACK_LEFT

            # 右转
            right_score = (-traj_direction[1]).clamp(min=0)  # y < 0 表示右转
            if V > 2:
                cam_importance[2] += right_score * 1.5  # FRONT_RIGHT
            if V > 4:
                cam_importance[4] += right_score * 1.0  # BACK_RIGHT

            # 归一化
            cam_importance = cam_importance / (cam_importance.sum() + 1e-6) * V

            # 扩展到空间维度 [V, H, W]
            importance_b = cam_importance.view(V, 1, 1).expand(V, H, W)

            # 添加空间高斯权重（中心更重要）
            y_coords = torch.linspace(-1, 1, H, device=device)
            x_coords = torch.linspace(-1, 1, W, device=device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            spatial_weight = torch.exp(-(xx**2 + yy**2) / 1.0)  # 高斯

            importance_b = importance_b * spatial_weight.unsqueeze(0)
            importance_maps.append(importance_b)

        importance = torch.stack(importance_maps, dim=0)  # [B, V, H, W]

        # 归一化到 [0, 1]
        importance = importance / (importance.max() + 1e-6)

        return importance


class CrossCameraAttention(nn.Module):
    """
    跨相机注意力：缺失相机从有效相机获取信息

    采用低分辨率空间 token 的 cross-attn，保留空间结构而非全局广播。
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        max_cameras=6,
        residual_scale=0.1,
        token_hw=(8, 8),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_cameras = max_cameras
        self.residual_scale = residual_scale  # 残差缩放，防止初期输出过大
        self.token_h = int(token_hw[0])
        self.token_w = int(token_hw[1])

        # 位置编码（每个相机有独立的位置编码）
        self.cam_pos_embed = nn.Parameter(torch.randn(max_cameras, embed_dim) * 0.02)
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, self.token_h * self.token_w, embed_dim) * 0.02
        )
        self.mask_status_embed = nn.Parameter(torch.randn(2, embed_dim) * 0.02)

        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # 特征投影
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

        # 空间降维（减少计算量）
        self.spatial_pool = nn.AdaptiveAvgPool2d((self.token_h, self.token_w))

    def _camera_pos(self, num_cams):
        if num_cams <= self.max_cameras:
            return self.cam_pos_embed[:num_cams]
        return self.cam_pos_embed.repeat((num_cams // self.max_cameras) + 1, 1)[:num_cams]

    def forward(self, feat, valid_mask, cam_mask):
        """
        feat: [B, V, C, H, W]
        valid_mask: [B, V] True = 有效相机
        cam_mask: [B, V] True = 被遮挡相机
        """
        B, V, C, H, W = feat.shape
        device = feat.device

        # 空间降维并转 token：[B, V, C, h, w] -> [B, V, S, C]
        feat_pooled = self.spatial_pool(feat.reshape(B * V, C, H, W))
        feat_pooled = feat_pooled.reshape(B, V, C, self.token_h * self.token_w).permute(0, 1, 3, 2)

        cam_pos = self._camera_pos(V).to(device=device, dtype=feat_pooled.dtype)  # [V, C]
        tokens = feat_pooled + cam_pos.view(1, V, 1, C) + self.spatial_pos_embed.to(
            device=device, dtype=feat_pooled.dtype
        )
        status_embed = self.mask_status_embed[
            cam_mask.to(device=device, dtype=torch.long).clamp(min=0, max=1)
        ].unsqueeze(2)
        tokens = tokens + status_embed.to(dtype=tokens.dtype)

        outputs = []
        for b in range(B):
            valid_cams = valid_mask[b]  # [V]
            missing_cams = cam_mask[b]  # [V]

            if not missing_cams.any() or not valid_cams.any():
                outputs.append(feat[b])
                continue

            q_indices = torch.where(missing_cams)[0]
            kv_indices = torch.where(valid_cams)[0]

            # Key/Value: 有效相机全部空间 token
            kv_tokens = tokens[b, kv_indices].reshape(-1, C)  # [N_valid*S, C]
            k = self.proj_k(kv_tokens).unsqueeze(0)
            v = self.proj_v(kv_tokens).unsqueeze(0)

            feat_b = feat[b].clone()  # [V, C, H, W]
            for cam_idx in q_indices:
                # Query: 缺失相机的空间 token
                q_tokens = tokens[b, cam_idx]  # [S, C]
                q = self.proj_q(q_tokens).unsqueeze(0)

                attn_out, _ = self.attention(q, k, v)  # [1, S, C]
                attn_out = self.proj_out(attn_out.squeeze(0))  # [S, C]

                # token -> 空间图 -> 上采样回原分辨率
                attn_map = attn_out.transpose(0, 1).reshape(C, self.token_h, self.token_w)
                attn_map = F.interpolate(
                    attn_map.unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

                feat_b[cam_idx] = feat_b[cam_idx] + self.residual_scale * attn_map

            outputs.append(feat_b)

        return torch.stack(outputs, dim=0)


class TrajectoryImportanceEncoder(nn.Module):
    """
    轨迹重要性编码器：将规划轨迹编码为图像空间的重要性图
    """
    def __init__(self, traj_dim=2, hidden_dim=64, output_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(traj_dim * 12, hidden_dim),  # 12个未来时间步
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6 * output_dim),  # 6个相机
        )

    def forward(self, trajectory):
        """
        trajectory: [B, T, 2]
        返回: [B, 6] 每个相机的重要性
        """
        B, T, D = trajectory.shape
        traj_flat = trajectory.view(B, -1)  # [B, T*2]

        # 补齐到12个时间步
        if T < 12:
            padding = torch.zeros(B, (12 - T) * D, device=trajectory.device)
            traj_flat = torch.cat([traj_flat, padding], dim=1)

        importance = self.encoder(traj_flat)  # [B, 6]
        importance = torch.sigmoid(importance)  # [0, 1]

        return importance


class PlanningFeedbackLoss(nn.Module):
    """
    规划反馈损失

    核心思想：补全质量由规划性能来评判
    - 传统损失：||补全特征 - 真实特征||² (重建损失)
    - 规划反馈：规划损失越小，说明补全越好
    """
    def __init__(self,
                 lambda_recon=0.1,      # 重建损失权重（降低）
                 lambda_planning=1.0,   # 规划损失权重（提高）
                 lambda_importance=0.5, # 重要性加权
                 loss_scale=1.0):       # 损失缩放因子
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_planning = lambda_planning
        self.lambda_importance = lambda_importance
        self.loss_scale = loss_scale

        # 使用 SmoothL1Loss，比 L1 更平滑，比 L2 更鲁棒
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=1.0)

    def forward(self,
                completed_feats,      # 补全后的特征
                original_feats,       # 原始完整特征（GT）
                cam_mask,             # 遮挡mask
                importance_maps=None, # 重要性图
                planning_loss=None):  # 规划损失（从规划模块传入）
        """
        计算规划引导的补全损失
        """
        losses = {}

        # 1. 重建损失（使用 SmoothL1，兼顾 L1 的鲁棒性和 L2 的平滑性）
        recon_loss = 0
        for scale_idx, (comp, orig) in enumerate(zip(completed_feats, original_feats)):
            B, V, C, H, W = comp.shape
            mask = cam_mask.view(B, V, 1, 1, 1).float()

            # SmoothL1: 小误差用 L2，大误差用 L1
            diff = self.smooth_l1(comp, orig.detach())

            # 应用重要性加权
            if importance_maps is not None and scale_idx < len(importance_maps):
                importance = importance_maps[scale_idx].unsqueeze(2)  # [B, V, 1, H, W]
                diff = diff * (1 + self.lambda_importance * importance)

            # 只计算被遮挡区域的损失
            masked_diff = diff * mask
            scale_loss = masked_diff.sum() / (mask.sum() * C * H * W + 1e-6)
            recon_loss = recon_loss + scale_loss

        recon_loss = recon_loss / len(completed_feats)

        # 使用 log 软缩放代替硬裁剪，保持梯度连续
        # log(1 + x) 在 x 大时增长缓慢，但梯度不会突然变为0
        recon_loss = torch.log1p(recon_loss) * self.loss_scale
        losses['loss_completion_recon'] = recon_loss * self.lambda_recon

        # 2. 规划反馈损失（主要损失）
        # 注意：规划损失在主forward中计算，会自动反传到补全模块
        # 这里只是记录，实际反传通过计算图自动完成
        if planning_loss is not None:
            losses['loss_completion_planning_feedback'] = planning_loss * self.lambda_planning

        return losses


class LightDreamerRSSM(nn.Module):
    """
    轻量级 Dreamer/RSSM：把多视角特征聚合成 token 序列，按视角维度滚动，
    预测缺失视角的稠密 token，并以 teacher（完整图像）特征监督。
    """

    def __init__(
        self,
        ch_per_scale,
        deter_dim=128,
        stoch_dim=32,
        hidden_dim=128,
        lambda_rec=1.0,
        lambda_kl=1e-4,
    ):
        super().__init__()
        if isinstance(ch_per_scale, (list, tuple)):
            self.obs_dim = sum(ch_per_scale)
        else:
            self.obs_dim = int(ch_per_scale)
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.lambda_rec = lambda_rec
        self.lambda_kl = lambda_kl

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, self.obs_dim),
        )
        self.gru = nn.GRUCell(stoch_dim, deter_dim)

    def _pool_multi_scale(self, feats):
        """
        feats: list[Tensor] -> List[[B,V,C,H,W]]
        return [B,V,obs_dim]
        """
        pooled = []
        for feat in feats:
            if not torch.is_tensor(feat):
                continue
            if feat.dim() == 5:
                pooled.append(feat.mean(dim=(-1, -2)))
            elif feat.dim() == 4:
                pooled.append(feat.mean(dim=(-1, -2)).unsqueeze(1))
        if len(pooled) == 0:
            return None
        return torch.cat(pooled, dim=-1)

    @staticmethod
    def _split_stats(stats):
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def _reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _kl_normal(mu_q, logvar_q, mu_p, logvar_p, eps=1e-5):
        var_q = (logvar_q.exp()).clamp_min(eps)
        var_p = (logvar_p.exp()).clamp_min(eps)
        kl = 0.5 * (
            (var_q + (mu_q - mu_p) ** 2) / var_p
            - 1.0
            + logvar_p
            - logvar_q
        )
        return kl.sum(dim=-1, keepdim=True)

    def _rollout(self, obs_tokens, target_tokens, cam_mask=None):
        B, T, _ = obs_tokens.shape
        deter = obs_tokens.new_zeros(B, self.deter_dim)
        rec_loss = obs_tokens.new_zeros(())
        kl_loss = obs_tokens.new_zeros(())
        rec_steps = 0.0
        kl_steps = 0.0

        for t in range(T):
            obs_t = obs_tokens[:, t]
            target_t = target_tokens[:, t]

            obs_embed = self.obs_encoder(obs_t)
            prior_mu, prior_logvar = self._split_stats(self.prior_net(deter))
            post_inp = torch.cat([deter, obs_embed], dim=-1)
            post_mu, post_logvar = self._split_stats(self.post_net(post_inp))

            if cam_mask is not None:
                mask_t = cam_mask[:, t].float().unsqueeze(-1)
                use_prior = mask_t.bool().expand_as(prior_mu)
                z_post = self._reparameterize(post_mu, post_logvar)
                z_prior = self._reparameterize(prior_mu, prior_logvar)
                z = torch.where(use_prior, z_prior, z_post)
            else:
                mask_t = None
                z = self._reparameterize(post_mu, post_logvar)

            deter = self.gru(z, deter)
            decoded = self.decoder(torch.cat([deter, z], dim=-1))

            recon = F.mse_loss(
                decoded, target_t.detach(), reduction="none"
            ).mean(dim=-1, keepdim=True)
            if mask_t is None:
                rec_loss = rec_loss + recon.mean()
                rec_steps += 1.0
            else:
                miss = mask_t
                miss_sum = miss.sum()
                if miss_sum > 0:
                    rec_loss = rec_loss + (recon * miss).sum() / miss_sum
                    rec_steps += 1.0

            if mask_t is None:
                kl = self._kl_normal(post_mu, post_logvar, prior_mu, prior_logvar)
                kl_loss = kl_loss + kl.mean()
                kl_steps += 1.0
            else:
                obs_mask = 1.0 - mask_t
                obs_sum = obs_mask.sum()
                if obs_sum > 0:
                    kl = self._kl_normal(
                        post_mu, post_logvar, prior_mu, prior_logvar
                    )
                    kl_loss = kl_loss + (kl * obs_mask).sum() / obs_sum
                    kl_steps += 1.0

        rec_den = obs_tokens.new_tensor(rec_steps if rec_steps > 0 else 1.0)
        kl_den = obs_tokens.new_tensor(kl_steps if kl_steps > 0 else 1.0)
        rec_loss = rec_loss / rec_den
        kl_loss = kl_loss / kl_den
        return rec_loss, kl_loss

    def forward(self, student_feats, teacher_feats, cam_mask=None):
        student_tokens = self._pool_multi_scale(student_feats)
        teacher_tokens = self._pool_multi_scale(teacher_feats)
        if student_tokens is None or teacher_tokens is None:
            return None

        if student_tokens.shape != teacher_tokens.shape:
            raise ValueError(
                f"student_tokens shape {student_tokens.shape} != teacher_tokens {teacher_tokens.shape}"
            )

        if cam_mask is not None:
            cam_mask = cam_mask.to(student_tokens.device, dtype=torch.bool)
            if cam_mask.dim() != 2:
                raise ValueError("cam_mask should be [B, V]")

        rec_loss, kl_loss = self._rollout(
            student_tokens, teacher_tokens.detach(), cam_mask
        )
        return {
            "loss_world_rec": rec_loss * self.lambda_rec,
            "loss_world_kl": kl_loss * self.lambda_kl,
        }


# ============================
# 3) 主模型：接入视角重建 + 自监督
# ============================
@DETECTORS.register_module()
class SparseDrive(BaseDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
        ssl_weight=0.01,       # 自监督损失权重
        world_model_cfg=None,  # Dreamer 风格潜世界模型
        test_cam_missing=False,  # 测试时是否模拟相机缺失
        cam_dropout_cfg=None,
        pv_recon_cfg=None,  # 视角特征重建配置（VAE / Flow Matching）
        temporal_completion_cfg=None,  # 时序补全配置
        planning_guided_completion_cfg=None,  # 规划引导补全配置
        frozen_modules=None,   # 冻结模块名列表，支持 "a.b.c" 点分路径
    ):
        super(SparseDrive, self).__init__(init_cfg=init_cfg)

        self.img_backbone = build_backbone(img_backbone)
        if pretrained is not None:
            # 修正原代码里的 backbone 引用错误
            self.img_backbone.pretrained = pretrained

        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        else:
            self.img_neck = None

        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask

        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func

        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None

        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

        if cam_dropout_cfg is None:
            cam_dropout_cfg = {}
        default_cam_dropout_cfg = dict(
            p_missing=0.6,
            n_min=1,
            n_max=2,
            train_only=False,
            seed=42,
            sticky_fault=True,
            sticky_min_frames=2,
            sticky_max_frames=6,
            curriculum_steps=8000,
            p_missing_start=0.15,
            n_max_start=1,
        )
        default_cam_dropout_cfg.update(cam_dropout_cfg)
        self.cam_dropout = RandCamMask(**default_cam_dropout_cfg)

        # 是否在 test 阶段模拟相机缺失
        self.test_cam_missing = test_cam_missing

        # 视角级特征重建模块，默认使用 VAE；可切换到 flow matching。
        self.pv_recon = build_pv_reconstructor(pv_recon_cfg)

        # 存储重建损失
        self.pv_recon_loss_dict = None

        self.ssl_weight = ssl_weight

        # 轻量 Dreamer 世界模型
        enable_world_model = True
        if world_model_cfg is None:
            world_model_cfg = dict(
                ch_per_scale=[256, 256, 256, 256],
                deter_dim=128,
                stoch_dim=32,
                hidden_dim=128,
                lambda_rec=0.01,  # 降低权重：从1.0改为0.01，避免世界模型损失主导训练
                lambda_kl=1e-4,
            )
        else:
            enable_world_model = world_model_cfg.pop("enabled", True)
        self.world_model = (
            LightDreamerRSSM(**world_model_cfg) if enable_world_model else None
        )

        # ===== 新增：时序补全模块（运动补偿版本，显存优化）=====
        if temporal_completion_cfg is None:
            temporal_completion_cfg = {}
        tc_enable = temporal_completion_cfg.get('enable', True)
        self.temporal_completion = MotionCompensatedTemporalCompletion(
            ch_per_scale=temporal_completion_cfg.get('ch_per_scale', [256, 256, 256, 256]),
            embed_dims=temporal_completion_cfg.get('embed_dims', 256),
            num_heads=temporal_completion_cfg.get('num_heads', 8),
            queue_length=temporal_completion_cfg.get('queue_length', 2),  # 显存优化：2帧
            num_cameras=temporal_completion_cfg.get('num_cameras', 6),
            reference_depths=temporal_completion_cfg.get('reference_depths', [10, 30]),  # 显存优化：2个深度
            kv_downsample=temporal_completion_cfg.get('kv_downsample', 4),  # 显存优化：4x下采样
            use_flash_attn=temporal_completion_cfg.get('use_flash_attn', False),
            enable=tc_enable,
        )

        # ===== 新增：规划导向加权模块 =====
        self.planning_weighting = PlanningGuidedWeighting(
            num_cameras=6,
            use_ego_state=True,  # 利用ego状态动态调整
            base_weights=None,  # 使用默认权重
        )

        # ===== 新增：规划引导补全模块 =====
        if planning_guided_completion_cfg is None:
            planning_guided_completion_cfg = {}
        pgc_enable = planning_guided_completion_cfg.get('enable', True)
        self.trajectory_source = planning_guided_completion_cfg.get('trajectory_source', 'gt')
        if self.trajectory_source not in ("none", "gt", "pred"):
            raise ValueError("planning_guided_completion_cfg.trajectory_source must be 'none'/'gt'/'pred'")
        self._pred_trajectory_train_fallback_warned = False
        self.planning_guided_completion = PlanningGuidedCompletion(
            ch_per_scale=planning_guided_completion_cfg.get('ch_per_scale', [256, 256, 256, 256]),
            hidden_dim=planning_guided_completion_cfg.get('hidden_dim', 256),
            use_trajectory_guidance=planning_guided_completion_cfg.get('use_trajectory_guidance', True),
            use_cross_camera=planning_guided_completion_cfg.get('use_cross_camera', True),
            use_conditional_completion=planning_guided_completion_cfg.get('use_conditional_completion', True),
            enable=pgc_enable,
        )

        # ===== 新增：规划反馈损失 =====
        self.planning_feedback_loss = PlanningFeedbackLoss(
            lambda_recon=0.1,      # 重建损失权重（较低）
            lambda_planning=1.0,   # 规划损失权重（主要）
            lambda_importance=0.5, # 重要性加权
        )

        # ===== 模块冻结 =====
        self._frozen_modules = frozen_modules or []
        if self._frozen_modules:
            self._freeze_modules()

    def _get_submodule(self, name):
        """通过点分路径获取子模块，如 'head.det_head'"""
        obj = self
        for attr in name.split('.'):
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj

    def _freeze_modules(self):
        for name in self._frozen_modules:
            module = self._get_submodule(name)
            if module is None:
                import warnings
                warnings.warn(f"[SparseDrive] frozen_modules: '{name}' not found, skipped.")
                continue
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

    def train(self, mode=True):
        super(SparseDrive, self).train(mode)
        # 保证冻结模块始终处于 eval 模式（避免 BN 统计被更新）
        for name in self._frozen_modules:
            module = self._get_submodule(name)
            if module is not None:
                module.eval()
        return self

    # -----------------------
    # 仅 backbone + neck 的特征提取
    # -----------------------
    @auto_fp16(apply_to=("img",), out_fp32=True)
    def _extract_backbone_neck(self, img, metas=None, enable_deform=False):
        """
        只跑 backbone + neck（可选 deformable），不做视角重建、不算 depth。
        输入 img: [B, V, 3, H, W] 或 [B, 3, H, W]
        输出: list of [B, V, C, H, W]
        """
        bs = img.shape[0]
        if img.dim() == 5:
            num_cams = img.shape[1]
            img_flat = img.flatten(0, 1)  # [B*V, 3, H, W]
        else:
            num_cams = 1
            img_flat = img

        if self.use_grid_mask:
            img_flat = self.grid_mask(img_flat)

        if "metas" in signature(self.img_backbone.forward).parameters:
            feats = self.img_backbone(img_flat, num_cams, metas=metas)
        else:
            feats = self.img_backbone(img_flat)

        if self.img_neck is not None:
            feats = list(self.img_neck(feats))
        else:
            feats = list(feats) if isinstance(feats, (list, tuple)) else [feats]

        # 还原成 [B, V, C, H, W]
        for i, f in enumerate(feats):
            feats[i] = f.view(bs, num_cams, *f.shape[1:])

        if self.use_deformable_func and enable_deform:
            feats = feature_maps_format(feats)

        return feats

    # -----------------------
    # 推理/测试用的 extract_feat（带 pv_recon / depth）
    # -----------------------
    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None, cam_mask=None):
        """
        仅在 eval/test 中使用；训练阶段 forward_train 不走这条
        """
        self.pv_recon_loss_dict = None  # 测试阶段不记录重建损失

        if self.training:
            # 训练阶段不应调用这个接口
            raise RuntimeError("extract_feat should not be used in training mode.")

        bs = img.shape[0]
        if img.dim() == 5:
            num_cams = img.shape[1]
            img_flat = img.flatten(0, 1)
        else:
            num_cams = 1
            img_flat = img

        if self.use_grid_mask:
            img_flat = self.grid_mask(img_flat)

        if "metas" in signature(self.img_backbone.forward).parameters:
            feats = self.img_backbone(img_flat, num_cams, metas=metas)
        else:
            feats = self.img_backbone(img_flat)

        if self.img_neck is not None:
            feats = list(self.img_neck(feats))
        else:
            feats = list(feats) if isinstance(feats, (list, tuple)) else [feats]

        # 还原成 [B, V, C, H, W]
        for i, f in enumerate(feats):
            feats[i] = f.view(bs, num_cams, *f.shape[1:])

        depths = None
        if return_depth and self.depth_branch is not None:
            focal = None
            if metas is not None and isinstance(metas, dict):
                focal = metas.get("focal", None)
            depths = self.depth_branch(feats, focal)

        # 注意：不在这里进行 feature_maps_format，
        # 因为 simple_test 中需要在格式化之前进行时序补全
        # feature_maps_format 将在 simple_test 中调用

        if return_depth:
            return feats, depths
        return feats

    # -----------------------
    # 自监督损失（特征一致性）
    # -----------------------
    def compute_ssl_loss(self, feats_full, feats_masked, cam_mask):
        """
        feats_full   : list of [B, V, C, H, W]，来自 full images（no_grad）
        feats_masked : list of [B, V, C, H, W]，来自 masked images（带梯度）
        cam_mask     : [B, V] bool，True=该相机被 blackout
        损失：仅在 cam_mask=True 的视角上，对齐 masked 与 full 的特征（MSE）
        """
        if cam_mask is None or (not cam_mask.any()):
            return feats_masked[0].new_tensor(0.0)

        cam_mask = cam_mask.to(feats_masked[0].device)
        B, V = cam_mask.shape
        mask = cam_mask.view(B, V, 1, 1, 1).float()  # [B,V,1,1,1]

        total_loss = 0.0
        num_scales = 0

        for f_full, f_mask in zip(feats_full, feats_masked):
            if f_full.dim() != 5 or f_mask.dim() != 5:
                continue

            Bf, Vf, C, H, W = f_full.shape
            if Bf != B or Vf != V:
                continue

            diff = (f_mask - f_full) ** 2  # [B,V,C,H,W]
            masked_diff = diff * mask
            valid = mask.sum()  # 被遮挡的 (B,V) 视角数

            if valid > 0:
                loss_scale = masked_diff.sum() / (valid * C * H * W)
                total_loss = total_loss + loss_scale
                num_scales += 1

        if num_scales > 0:
            total_loss = total_loss / num_scales
        else:
            total_loss = feats_masked[0].new_tensor(0.0)

        return total_loss * self.ssl_weight

    def _normalize_cam_mask(self, cam_mask, batch_size, num_cams, device):
        if cam_mask is None:
            return None

        if hasattr(cam_mask, "data") and not torch.is_tensor(cam_mask):
            cam_mask = cam_mask.data
        if isinstance(cam_mask, (list, tuple)):
            if len(cam_mask) == 1:
                cam_mask = cam_mask[0]
                if hasattr(cam_mask, "data") and not torch.is_tensor(cam_mask):
                    cam_mask = cam_mask.data
            elif all(torch.is_tensor(x) for x in cam_mask):
                cam_mask = torch.stack([x.to(device=device) for x in cam_mask], dim=0)
            else:
                cam_mask = torch.as_tensor(cam_mask, device=device)
        elif not torch.is_tensor(cam_mask):
            cam_mask = torch.as_tensor(cam_mask, device=device)
        else:
            cam_mask = cam_mask.to(device=device)

        if cam_mask.dim() == 3 and cam_mask.shape[0] == 1:
            cam_mask = cam_mask.squeeze(0)
        if cam_mask.dim() == 1:
            if cam_mask.shape[0] != num_cams:
                raise ValueError("cam_mask length should match num_cams")
            cam_mask = cam_mask.unsqueeze(0).expand(batch_size, -1)
        elif cam_mask.dim() == 2 and cam_mask.shape[0] == 1 and batch_size > 1:
            cam_mask = cam_mask.expand(batch_size, -1)

        if cam_mask.dim() != 2 or cam_mask.shape != (batch_size, num_cams):
            raise ValueError("cam_mask should have shape [B, V]")
        return cam_mask.to(device=device, dtype=torch.bool)

    def _apply_cam_mask_to_images(self, img, cam_mask):
        if cam_mask is None:
            return img
        if img.dim() == 5:
            return img * (~cam_mask).view(img.shape[0], img.shape[1], 1, 1, 1).to(img.dtype)
        if img.dim() == 4 and cam_mask.shape[1] == 1:
            return img * (~cam_mask).view(img.shape[0], 1, 1, 1).to(img.dtype)
        return img

    def _resolve_cam_mask_input(self, img, data, sample_if_missing=False):
        batch_size = img.shape[0]
        num_cams = img.shape[1] if img.dim() == 5 else 1
        cam_mask = self._normalize_cam_mask(
            data.get("cam_mask", None),
            batch_size=batch_size,
            num_cams=num_cams,
            device=img.device,
        )
        if cam_mask is not None:
            return self._apply_cam_mask_to_images(img, cam_mask), cam_mask
        if sample_if_missing:
            img_masked, cam_mask = self.cam_dropout(
                img,
                return_mask=True,
                metas=data.get("img_metas", None),
            )
            return img_masked, cam_mask.to(img.device, dtype=torch.bool)
        return img, None

    def _extract_ego_state(self, data, device):
        ego_state = None
        if "can_bus" in data and data["can_bus"] is not None:
            can_bus = data["can_bus"]
            if isinstance(can_bus, torch.Tensor):
                can_bus = can_bus.to(device)
                if can_bus.dim() >= 2 and can_bus.shape[-1] >= 3:
                    ego_state = can_bus[:, :3]
        return ego_state

    def _extract_cam_params(self, data, device):
        cam_params = None
        img_metas = data.get("img_metas", None)
        if isinstance(img_metas, list) and len(img_metas) > 0 and "intrinsics" in img_metas[0]:
            intrinsics = []
            for meta in img_metas:
                intr = meta.get("intrinsics", None)
                if intr is None:
                    intrinsics = []
                    break
                intrinsics.append(torch.as_tensor(intr, dtype=torch.float32, device=device))
            if len(intrinsics) == len(img_metas):
                cam_params = {"intrinsics": torch.stack(intrinsics, dim=0)}
        return cam_params

    def _clone_runtime_obj(self, obj):
        if torch.is_tensor(obj):
            return obj.clone()
        if isinstance(obj, list):
            return [self._clone_runtime_obj(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(self._clone_runtime_obj(x) for x in obj)
        if isinstance(obj, dict):
            return {k: self._clone_runtime_obj(v) for k, v in obj.items()}
        return copy.deepcopy(obj)

    def _capture_head_runtime_state(self):
        runtime_state = {}
        head = getattr(self, "head", None)
        if head is None:
            return runtime_state

        def capture_instance_bank(sub_head, key_prefix):
            if sub_head is None or not hasattr(sub_head, "instance_bank"):
                return
            bank = sub_head.instance_bank
            bank_keys = [
                "cached_feature", "cached_anchor", "metas", "mask",
                "confidence", "temp_confidence", "instance_id", "prev_id",
            ]
            instance_state = {}
            for k in bank_keys:
                value = getattr(bank, k, None)
                instance_state[k] = value if k == "metas" else self._clone_runtime_obj(value)
            runtime_state[f"{key_prefix}_instance_bank"] = instance_state

        def capture_sampler(sub_head, key_prefix):
            if sub_head is None or not hasattr(sub_head, "sampler"):
                return
            runtime_state[f"{key_prefix}_sampler_dn_metas"] = self._clone_runtime_obj(
                getattr(sub_head.sampler, "dn_metas", None)
            )

        det_head = getattr(head, "det_head", None)
        map_head = getattr(head, "map_head", None)
        capture_instance_bank(det_head, "det")
        capture_instance_bank(map_head, "map")
        capture_sampler(det_head, "det")
        capture_sampler(map_head, "map")

        motion_head = getattr(head, "motion_plan_head", None)
        if motion_head is not None and hasattr(motion_head, "instance_queue"):
            queue = motion_head.instance_queue
            queue_keys = [
                "metas", "prev_instance_id", "prev_confidence", "period",
                "instance_feature_queue", "anchor_queue", "prev_ego_status",
                "ego_period", "ego_feature_queue", "ego_anchor_queue",
            ]
            queue_state = {}
            for k in queue_keys:
                value = getattr(queue, k, None)
                queue_state[k] = value if k == "metas" else self._clone_runtime_obj(value)
            runtime_state["motion_instance_queue"] = queue_state

        return runtime_state

    def _restore_head_runtime_state(self, runtime_state):
        head = getattr(self, "head", None)
        if head is None or not runtime_state:
            return

        def restore_instance_bank(sub_head, key_prefix):
            key = f"{key_prefix}_instance_bank"
            if sub_head is None or not hasattr(sub_head, "instance_bank") or key not in runtime_state:
                return
            for attr, value in runtime_state[key].items():
                setattr(sub_head.instance_bank, attr, value)

        def restore_sampler(sub_head, key_prefix):
            key = f"{key_prefix}_sampler_dn_metas"
            if sub_head is None or not hasattr(sub_head, "sampler") or key not in runtime_state:
                return
            setattr(sub_head.sampler, "dn_metas", runtime_state[key])

        det_head = getattr(head, "det_head", None)
        map_head = getattr(head, "map_head", None)
        restore_instance_bank(det_head, "det")
        restore_instance_bank(map_head, "map")
        restore_sampler(det_head, "det")
        restore_sampler(map_head, "map")

        motion_head = getattr(head, "motion_plan_head", None)
        if (
            motion_head is not None
            and hasattr(motion_head, "instance_queue")
            and "motion_instance_queue" in runtime_state
        ):
            for attr, value in runtime_state["motion_instance_queue"].items():
                setattr(motion_head.instance_queue, attr, value)

    def _decode_predicted_trajectory(self, model_outs):
        if not isinstance(model_outs, (list, tuple)) or len(model_outs) < 4:
            return None
        planning_output = model_outs[3]
        if not isinstance(planning_output, dict):
            return None
        cls_seq = planning_output.get("classification", None)
        reg_seq = planning_output.get("prediction", None)
        if not cls_seq or not reg_seq:
            return None

        plan_cls = cls_seq[-1]
        plan_reg = reg_seq[-1]
        if plan_cls is None or plan_reg is None:
            return None

        if plan_cls.dim() == 3:
            plan_cls = plan_cls.squeeze(1)
        elif plan_cls.dim() > 2:
            plan_cls = plan_cls.reshape(plan_cls.shape[0], -1)

        if plan_reg.dim() == 5:
            plan_reg = plan_reg.squeeze(1)
        elif plan_reg.dim() > 4:
            plan_reg = plan_reg.reshape(plan_reg.shape[0], -1, plan_reg.shape[-2], plan_reg.shape[-1])

        if plan_cls.dim() != 2 or plan_reg.dim() != 4:
            return None

        num_modes = min(plan_cls.shape[1], plan_reg.shape[1])
        if num_modes <= 0:
            return None

        plan_cls = plan_cls[:, :num_modes]
        plan_reg = plan_reg[:, :num_modes]

        mode_idx = plan_cls.sigmoid().argmax(dim=-1)
        batch_idx = torch.arange(plan_cls.shape[0], device=plan_cls.device)
        traj_delta = plan_reg[batch_idx, mode_idx]  # [B, T, 2]
        return traj_delta.cumsum(dim=-2).detach()

    def _extract_predicted_trajectory(self, feature_maps, data):
        if feature_maps is None:
            return None

        runtime_state = self._capture_head_runtime_state()
        feats_for_head = feature_maps
        if self.use_deformable_func:
            feats_for_head = feature_maps_format(feats_for_head)

        try:
            cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                with torch.no_grad():
                    model_outs = self.head(feats_for_head, data)
                    traj = self._decode_predicted_trajectory(model_outs)
        finally:
            self._restore_head_runtime_state(runtime_state)

        return traj

    def _extract_gt_completion_trajectory(self, data, device):
        traj = data.get("ego_fut_trajs", None)
        if traj is None:
            traj = data.get("gt_ego_fut_trajs", None)
        if traj is None:
            return None
        return torch.as_tensor(traj, dtype=torch.float32, device=device)

    def _extract_completion_trajectory(self, data, device, feature_maps=None):
        if self.trajectory_source == "none":
            return None

        if self.trajectory_source == "pred":
            if self.training:
                traj_gt = self._extract_gt_completion_trajectory(data, device)
                if not self._pred_trajectory_train_fallback_warned:
                    warnings.warn(
                        "[SparseDrive] trajectory_source='pred' falls back to GT/none during training "
                        "to avoid an extra detached head forward."
                    )
                    self._pred_trajectory_train_fallback_warned = True
                return traj_gt
            traj_pred = self._extract_predicted_trajectory(feature_maps, data)
            if traj_pred is None:
                return None
            return traj_pred.to(device=device, dtype=torch.float32)

        # trajectory_source == "gt"
        return self._extract_gt_completion_trajectory(data, device)

    # -----------------------
    # 主 forward
    # -----------------------
    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    # -----------------------
    # 训练前向：带 SSL + 视角重建 + 时序补全 + 规划导向加权
    # -----------------------
    def forward_train(self, img, **data):
        # img: [B, V, 3, H, W]
        B, V = img.shape[:2]

        # 1) 保留一份 full images，用于 SSL teacher
        img_full = img.clone()

        # 2) 优先使用外部相机可用性标注；缺失时再采样模拟缺失
        img_masked, cam_mask = self._resolve_cam_mask_input(
            img,
            data,
            sample_if_missing=True,
        )
        if cam_mask is None:
            cam_mask = torch.zeros((B, V), dtype=torch.bool, device=img.device)

        # 3) masked 分支：带梯度的 backbone+neck（student）
        feats_mask_base = self._extract_backbone_neck(
            img_masked, metas=data, enable_deform=False
        )  # list of [B,V,C,H,W]

        # 4) full 分支：no_grad 的 backbone+neck（teacher）
        with torch.no_grad():
            feats_full_base = self._extract_backbone_neck(
                img_full, metas=data, enable_deform=False
            )

        # 5) 自监督损失：只对被 blackout 的视角做特征一致性
        loss_ssl = self.compute_ssl_loss(feats_full_base, feats_mask_base, cam_mask)

        # 5.1) Dreamer 风格潜世界模型监督
        world_loss_dict = None
        if self.world_model is not None:
            world_loss_dict = self.world_model(
                feats_mask_base, feats_full_base, cam_mask
            )

        # ===== 新增：时序补全（运动补偿版本）=====
        # 6) 时序补全：利用历史帧预测缺失相机的特征
        # 新版本模块内置特征队列，自动管理历史帧
        feats_temporal = self.temporal_completion(
            feats_mask_base, cam_mask, metas=data
        )

        # ===== 新增：规划导向加权 =====
        # 7) 计算相机重要性权重
        ego_state = self._extract_ego_state(data, img.device)
        importance_weights = self.planning_weighting(cam_mask, ego_state)  # [B, V]

        # 8) 视角特征重建（应用规划导向加权）
        feature_maps, pv_recon_loss = self.pv_recon(
            feats_temporal,
            cam_mask,
            metas=data,
            camera_weights=importance_weights,
            target_feature_maps=feats_full_base,
        )
        self.pv_recon_loss_dict = pv_recon_loss  # 用于后面合并 loss

        # ===== 新增：规划引导补全 =====
        # 9) 轨迹来源由 trajectory_source 控制：none / gt / pred（pred 使用 stop-grad）
        ego_trajectory = self._extract_completion_trajectory(
            data,
            img.device,
            feature_maps=feature_maps,
        )
        cam_params = self._extract_cam_params(data, img.device)

        # 10) 应用规划引导补全
        planning_guided_loss = {}
        importance_maps = None

        if cam_mask.any() and self.planning_guided_completion.enable:
            # 使用规划引导补全
            feature_maps_guided, importance_maps = self.planning_guided_completion(
                feature_maps, cam_mask, ego_trajectory, cam_params
            )

            # 如果补全成功，使用补全后的特征
            if feature_maps_guided is not None:
                # 计算规划引导补全的损失
                planning_guided_loss = self.planning_feedback_loss(
                    completed_feats=feature_maps_guided,
                    original_feats=feats_full_base,  # GT 特征
                    cam_mask=cam_mask,
                    importance_maps=importance_maps,
                    planning_loss=None  # 规划损失将在后面计算
                )

                # 使用补全后的特征继续
                feature_maps = feature_maps_guided

        # 11) 深度分支（如果有）
        depths = None
        if self.depth_branch is not None and "gt_depth" in data:
            focal = None
            if isinstance(data, dict):
                focal = data.get("focal", None)
            depths = self.depth_branch(feature_maps, focal)

        # 12) deformable 聚合给 head 用
        feats_for_head = feature_maps
        if self.use_deformable_func:
            feats_for_head = feature_maps_format(feats_for_head)

        # 13) head 前向 + 监督 loss
        model_outs = self.head(feats_for_head, data)
        output = self.head.loss(model_outs, data)

        # ===== 端到端规划反馈说明 =====
        # 规划损失会自动反传到补全模块，因为：
        # 1. feature_maps_guided 没有使用 detach()
        # 2. 梯度路径: planning_loss -> head -> feature_maps_guided -> planning_guided_completion
        # 不需要额外添加 loss_planning_feedback，否则会重复计算

        if depths is not None:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )

        # 合并视角重建损失
        if self.pv_recon_loss_dict is not None:
            output.update(self.pv_recon_loss_dict)

        if world_loss_dict is not None:
            output.update(world_loss_dict)

        # 合并规划引导补全损失
        if planning_guided_loss:
            output.update(planning_guided_loss)

        # 合并自监督损失
        output["loss_pv_ssl"] = loss_ssl

        return output

    # -----------------------
    # 测试前向
    # -----------------------
    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        img, cam_mask = self._resolve_cam_mask_input(
            img,
            data,
            sample_if_missing=getattr(self, "test_cam_missing", False),
        )

        # 提取特征
        feature_maps = self.extract_feat(
            img,
            metas=data.get("img_metas", None),
            cam_mask=cam_mask,
            return_depth=False,
        )

        # ===== 新增：测试时的时序补全 =====
        # 每帧都调用以维护内部历史队列；无缺失时模块会直接旁路返回。
        feature_maps_list = list(feature_maps) if isinstance(feature_maps, tuple) else feature_maps
        with torch.no_grad():
            feature_maps = self.temporal_completion(
                feature_maps_list, cam_mask, metas=data
            )

        # ===== 新增：测试时的视角重建 + 规划引导补全 =====
        if cam_mask is not None and cam_mask.any():
            feature_maps_list = list(feature_maps) if isinstance(feature_maps, tuple) else feature_maps
            ego_state = self._extract_ego_state(data, img.device)
            importance_weights = self.planning_weighting(cam_mask.to(img.device), ego_state)
            cam_params = self._extract_cam_params(data, img.device)
            with torch.no_grad():
                feature_maps_list, _ = self.pv_recon(
                    feature_maps_list,
                    cam_mask.to(img.device),
                    metas=data,
                    camera_weights=importance_weights,
                )
                ego_trajectory = self._extract_completion_trajectory(
                    data,
                    img.device,
                    feature_maps=feature_maps_list,
                )
                feature_maps_guided, _ = self.planning_guided_completion(
                    feature_maps_list,
                    cam_mask.to(img.device),
                    ego_trajectory=ego_trajectory,
                    cam_params=cam_params,
                )
            if feature_maps_guided is not None:
                feature_maps = feature_maps_guided

        # 在时序补全之后，进行特征格式化（如果需要）
        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)

        # head 推理
        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in list(data.keys()):
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
