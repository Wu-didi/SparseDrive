from inspect import signature

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
                 seed: int = 42):
        super().__init__()
        assert 0.0 <= p_missing <= 1.0
        assert 1 <= n_min <= n_max
        self.p_missing = p_missing
        self.n_min = n_min
        self.n_max = n_max
        self.train_only = train_only
        self._seed = seed

        # 独立生成器，确保复现且不扰动全局 RNG
        self._g = torch.Generator()
        self._g.manual_seed(seed)

    @torch.no_grad()
    def forward(self, x, return_mask: bool = False):
        if self.train_only and not self.training:
            return (x, None) if return_mask else x

        B, N_cam, C, H, W = x.shape
        device = x.device

        # 生成器设备对齐（避免 cpu/gpu 不匹配报错）
        if getattr(self._g, "device", torch.device("cpu")).type != device.type:
            self._g = torch.Generator(device=device)
            self._g.manual_seed(self._seed)

        # [B] 哪些样本触发缺失（直接在 device 上采样）
        mask_flag = torch.rand(B, generator=self._g, device=device) < self.p_missing

        cam_mask = torch.zeros(B, N_cam, device=device, dtype=torch.bool)
        if mask_flag.any():
            n_max = min(self.n_max, N_cam)
            for b in torch.nonzero(mask_flag, as_tuple=False).squeeze(1):
                n_drop = torch.randint(
                    self.n_min,
                    n_max + 1,
                    (1,),
                    generator=self._g,
                    device=device,
                ).item()
                drop_ids = torch.randperm(
                    N_cam,
                    generator=self._g,
                    device=device,
                )[:n_drop]
                cam_mask[b, drop_ids] = True

        # 应用遮挡（被遮挡视角直接置 0）
        x = x * (~cam_mask).view(B, N_cam, 1, 1, 1).to(x.dtype)
        return (x, cam_mask) if return_mask else x


# ===========================================
# 2) 视角级特征补全（VAE）
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

    def forward(self, feature_maps, cam_mask, metas=None):
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
            F_in = Fm.view(B * V, C, H, W)           # [B*V, C, H, W]
            F_detach = F_in.detach()                 # 只训练 VAE，不回传到 backbone

            x_rec, mu, logvar = self.vaes[i](F_in)   # [B*V, C, H, W], [B*V, C_lat, H, W]

            # ---------- VAE 损失 ----------
            # 重建损失：MSE
            rec_loss = F.mse_loss(x_rec, F_detach)

            # KL 散度（per-pixel）
            kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
            kl_loss = kl.mean()

            total_rec_loss = total_rec_loss + rec_loss
            total_kl_loss = total_kl_loss + kl_loss

            # ---------- 视角级补全 ----------
            F_rec = x_rec.view(B, V, C, H, W)
            miss = cam_mask.view(B, V, 1, 1, 1).to(Fm.dtype)  # 1=缺失
            F_out = Fm * (1.0 - miss) + F_rec * miss

            outs.append(F_out)

        loss_dict = {
            "loss_pv_vae_rec": self.lambda_rec * total_rec_loss,
            "loss_pv_vae_kl": self.lambda_kl * total_kl_loss,
        }
        return outs, loss_dict


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
# 3) 主模型：接入 VAE + 自监督
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
        ssl_weight=1.0,        # 自监督损失权重
        world_model_cfg=None,  # Dreamer 风格潜世界模型
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

        # 随机相机失效
        self.cam_dropout = RandCamMask(
            p_missing=0.6, n_min=1, n_max=2, train_only=False, seed=42
        )

        # 是否在 test 阶段模拟相机缺失
        self.test_cam_missing = False  # 可在外部设置为 True

        # VAE 视角级特征补全模块
        # 注意 ch_per_scale 要和 neck 输出通道对齐
        self.pv_recon = PVReconVAE(
            ch_per_scale=[256, 256, 256, 256],
            latent_channels=64,
            lambda_rec=1.0,
            lambda_kl=1e-4,
        )

        # 存储 VAE loss
        self.vae_loss_dict = None

        # 自监督损失权重
        self.ssl_weight = ssl_weight

        # 轻量 Dreamer 世界模型
        enable_world_model = True
        if world_model_cfg is None:
            world_model_cfg = dict(
                ch_per_scale=[256, 256, 256, 256],
                deter_dim=128,
                stoch_dim=32,
                hidden_dim=128,
                lambda_rec=1.0,
                lambda_kl=1e-4,
            )
        else:
            enable_world_model = world_model_cfg.pop("enabled", True)
        self.world_model = (
            LightDreamerRSSM(**world_model_cfg) if enable_world_model else None
        )

    # -----------------------
    # 仅 backbone + neck 的特征提取
    # -----------------------
    @auto_fp16(apply_to=("img",), out_fp32=True)
    def _extract_backbone_neck(self, img, metas=None, enable_deform=False):
        """
        只跑 backbone + neck（可选 deformable），不做 VAE 补全、不算 depth。
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
    # 推理/测试用的 extract_feat（带 VAE / depth）
    # -----------------------
    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None, cam_mask=None):
        """
        仅在 eval/test 中使用；训练阶段 forward_train 不走这条
        """
        self.vae_loss_dict = None  # 测试阶段不记录 VAE 损失

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

        # 测试阶段若 cam_mask 有 True，可以用 VAE 补全
        if cam_mask is not None:
            cam_mask = cam_mask.to(feats[0].device, dtype=torch.bool)
            if cam_mask.any():
                feats, _ = self.pv_recon(feats, cam_mask, metas)

        depths = None
        if return_depth and self.depth_branch is not None:
            focal = None
            if metas is not None and isinstance(metas, dict):
                focal = metas.get("focal", None)
            depths = self.depth_branch(feats, focal)

        if self.use_deformable_func:
            feats = feature_maps_format(feats)

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
    # 训练前向：带 SSL + VAE
    # -----------------------
    def forward_train(self, img, **data):
        # img: [B, V, 3, H, W]
        B, V = img.shape[:2]

        # 1) 保留一份 full images，用于 SSL teacher
        img_full = img.clone()

        # 2) 随机相机遮挡，得到 masked images + cam_mask
        img_masked, cam_mask = self.cam_dropout(img, return_mask=True)
        cam_mask = cam_mask.to(img.device)

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

        # 6) VAE 视角补全：只作用在 masked 分支特征上
        feature_maps, vae_loss = self.pv_recon(feats_mask_base, cam_mask, metas=data)
        self.vae_loss_dict = vae_loss  # 用于后面合并 loss

        # 7) 深度分支（如果有）
        depths = None
        if self.depth_branch is not None and "gt_depth" in data:
            focal = None
            if isinstance(data, dict):
                focal = data.get("focal", None)
            depths = self.depth_branch(feature_maps, focal)

        # 8) deformable 聚合给 head 用
        feats_for_head = feature_maps
        if self.use_deformable_func:
            feats_for_head = feature_maps_format(feats_for_head)

        # 9) head 前向 + 监督 loss
        model_outs = self.head(feats_for_head, data)
        output = self.head.loss(model_outs, data)

        if depths is not None:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )

        # 合并 VAE 损失
        if self.vae_loss_dict is not None:
            output.update(self.vae_loss_dict)

        if world_loss_dict is not None:
            output.update(world_loss_dict)

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
        cam_mask = None
        # 仅在配置里开启时，才在测试阶段模拟相机缺失
        if getattr(self, "test_cam_missing", False):
            img, cam_mask = self.cam_dropout(img, return_mask=True)

        feature_maps = self.extract_feat(
            img,
            metas=data.get("img_metas", None),
            cam_mask=cam_mask,
            return_depth=False,
        )

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
