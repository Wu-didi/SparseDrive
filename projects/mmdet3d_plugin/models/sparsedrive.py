from inspect import signature

import torch
import torch.nn as nn
import torch.nn.functional as function

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

from einops import rearrange


# ================================
# 1) 随机相机失效（仅用于采样 cam_mask）
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
        x :  [B, V, C, H, W]  float32
    Return
        (x_masked, cam_mask)  其中 cam_mask: [B, V] bool，True=该相机被“选中缺失”
    """
    def __init__(
        self,
        p_missing: float = 0.5,
        n_min: int = 1,
        n_max: int = 2,
        train_only: bool = True,
        seed: int = 42,
    ):
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

        # 这里虽然返回了 x_masked，但在方案 B 中，我们只真正用 cam_mask
        x = x * (~cam_mask).view(B, N_cam, 1, 1, 1).to(x.dtype)
        return (x, cam_mask) if return_mask else x


# ===========================================
# 2) 视角级特征 VAE（单尺度）
# ===========================================
class SimpleFeatureVAE(nn.Module):
    """
    一个简单的卷积 VAE，用于对单尺度特征图做重建：
      - 输入:  [B, C, H, W]
      - 输出:  x_rec, mu, logvar  其中 x_rec 与输入同形状
    """
    def __init__(self, in_channels: int, latent_channels: int = 64):
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
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar


# ===========================================
# 3) 多尺度、多视角 VAE 自监督模块（方案 B）
# ===========================================
class PVReconVAE(nn.Module):
    """
    多尺度、多视角 VAE 自监督补全模块（方案 B）：
    - 不在图像上做缺失，只在特征层做 masked reconstruction
    - target 是完整特征 F.detach()，mask 后输入 VAE 重建
    - loss 只在 cam_mask=True 的视角上计算
    """
    def __init__(
        self,
        ch_per_scale,
        latent_channels: int = 64,
        lambda_rec: float = 1.0,
        lambda_kl: float = 1e-4,
    ):
        super().__init__()
        assert isinstance(ch_per_scale, (list, tuple))
        self.vaes = nn.ModuleList(
            [SimpleFeatureVAE(c, latent_channels=latent_channels) for c in ch_per_scale]
        )
        self.lambda_rec = lambda_rec
        self.lambda_kl = lambda_kl

    def forward(self, feature_maps, cam_mask, metas=None):
        """
        Args:
            feature_maps: list of tensors
                其中我们只对形状为 [B, V, C, H, W] 的层做自监督
            cam_mask    : [B, V] bool，True 表示该视角“被当作缺失”，
                          仅在这些位置计算重建 + KL 损失
        Returns:
            outs:      list of 原始 feature_maps（当前版本不改动，只算 loss）
            loss_dict: {'loss_pv_vae_rec': ..., 'loss_pv_vae_kl': ...}
        """
        assert cam_mask is not None, "PVReconVAE 需要 cam_mask（可以全 0）"

        outs = []
        total_rec_loss = 0.0
        total_kl_loss = 0.0
        B_mask, V_mask = cam_mask.shape

        for i, F in enumerate(feature_maps):
            # 只在 5D 特征上做多视角自监督，其它维度（比如 4D/3D）直接略过
            if F.dim() == 5:
                B, V, C, H, W = F.shape
                assert B == B_mask and V == V_mask, "cam_mask 维度需与特征一致"

                F_target = F.detach()  # 完整特征作为重建 target

                # 构造 mask：True=缺失，需要重建
                miss = cam_mask.view(B, V, 1, 1, 1).to(F.dtype)  # [B,V,1,1,1]
                keep = 1.0 - miss

                # 仅在 miss==1 的视角位置，将输入特征置 0，模拟“特征缺失”
                F_in = F * keep  # [B, V, C, H, W]

                # 展平视角维度喂入 VAE
                F_in_flat = F_in.view(B * V, C, H, W)
                x_rec, mu, logvar = self.vaes[i](F_in_flat)

                # 还原视角维
                F_rec = x_rec.view(B, V, C, H, W)

                # ========= 重建误差（只在 miss==1 的位置上算） =========
                num_missing = miss.sum()
                if num_missing > 0:
                    diff = (F_rec - F_target) ** 2  # [B,V,C,H,W]
                    rec_loss = (diff * miss).sum() / (num_missing * C * H * W)
                else:
                    rec_loss = function.mse_loss(F_rec, F_target)

                # ========= KL 散度（同样只在 miss==1 上统计平均） =========
                kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)  # [B*V,Lat,H,W]
                Bv, Lat, Hk, Wk = kl.shape
                kl = kl.view(B, V, Lat, Hk, Wk)
                if num_missing > 0:
                    kl_loss = (
                        kl * miss.view(B, V, 1, 1, 1)
                    ).sum() / (num_missing * Lat * Hk * Wk)
                else:
                    kl_loss = kl.mean()

                total_rec_loss = total_rec_loss + rec_loss
                total_kl_loss = total_kl_loss + kl_loss

                outs.append(F)
            else:
                # 比如某些特殊层是 [B,C,H,W] 或 [C,H,W]，这里不做自监督，直接 passthrough
                outs.append(F)

        # 把 total_rec_loss / total_kl_loss 转成 tensor，避免某些 batch 是纯 float(0.0)
        if isinstance(feature_maps, (list, tuple)) and len(feature_maps) > 0:
            ref_feat = feature_maps[0]
            if isinstance(ref_feat, torch.Tensor):
                ref_device = ref_feat.device
            else:
                ref_device = torch.device("cpu")
        else:
            ref_device = torch.device("cpu")

        total_rec_loss = torch.as_tensor(
            total_rec_loss, device=ref_device, dtype=torch.float32
        )
        total_kl_loss = torch.as_tensor(
            total_kl_loss, device=ref_device, dtype=torch.float32
        )

        loss_dict = {
            "loss_pv_vae_rec": self.lambda_rec * total_rec_loss,
            "loss_pv_vae_kl": self.lambda_kl * total_kl_loss,
        }
        return outs, loss_dict


# ============================
# 4) 主模型：SparseDrive
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
        use_grid_mask: bool = True,
        use_deformable_func: bool = False,
        depth_branch=None,
    ):
        super(SparseDrive, self).__init__(init_cfg=init_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is not None:
            backbone.pretrained = pretrained  # type: ignore[name-defined]

        self.img_backbone = build_backbone(img_backbone)
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

        # ---------------- 随机相机“缺失” mask（训练用于自监督） ----------------
        self.cam_dropout = RandCamMask(
            p_missing=0.6,
            n_min=1,
            n_max=2,
            train_only=True,
            seed=42,
        )

        # ------------ VAE 视角级特征自监督模块 ------------
        # 注意：这里 ch_per_scale 要和 neck 输出通道对齐
        # 例如 FPN 把通道统一到 256，则可以写 [256,256,256,256]
        self.pv_recon = PVReconVAE(
            ch_per_scale=[256, 256, 256, 256],
            latent_channels=64,
            lambda_rec=1.0,
            lambda_kl=1e-4,
        )

        # 用于在 forward_train 中取出 VAE loss
        self.vae_loss_dict = None

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth: bool = False, metas=None):
        """提取多尺度多视角特征（不在此处做 VAE 自监督）。"""
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)  # [B*V, 3, H, W]
        else:
            num_cams = 1

        if self.use_grid_mask:
            img = self.grid_mask(img)

        # 有些 backbone 会显式接收 metas
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)  # tuple

        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))  # list

        # 展回 [B, V, C, H, W]（如果原来就是 4D/其它维度则保持不动）
        fm_list = []
        for feat in feature_maps:
            if feat.dim() == 4:
                fm_list.append(
                    torch.reshape(
                        feat,
                        (bs, num_cams) + feat.shape[1:],
                    )
                )
            else:
                fm_list.append(feat)
        feature_maps = fm_list

        # 可选：深度分支
        if return_depth and self.depth_branch is not None:
            focal = None
            if isinstance(metas, dict):
                focal = metas.get("focal", None)
            if isinstance(metas, (list, tuple)) and len(metas) > 0 and isinstance(
                metas[0], dict
            ):
                focal = metas[0].get("focal", None)
            depths = self.depth_branch(feature_maps, focal)
        else:
            depths = None

        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)

        if return_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        """训练阶段：主任务 + 特征层 VAE 自监督（方案 B）。"""
        # img: [B, V, 3, H, W]

        # 1) 提取完整多视角特征（不对图像做 dropout）
        feature_maps, depths = self.extract_feat(
            img,
            return_depth=True,
            metas=data,
        )

        # 2) 使用 RandCamMask 仅采样 cam_mask（不改动特征 / 图像）
        #    期望 cam_mask: [B, V] bool，True 表示该视角被当作“缺失”
        B = feature_maps[0].shape[0]
        V = feature_maps[0].shape[1]
        dummy = torch.zeros(
            B,
            V,
            3,
            1,
            1,
            device=feature_maps[0].device,
            dtype=feature_maps[0].dtype,
        )
        _, cam_mask = self.cam_dropout(dummy, return_mask=True)

        # 3) VAE 自监督：只用来计算 loss，不改动用于 head 的特征
        if hasattr(self, "pv_recon") and self.pv_recon is not None:
            _, vae_loss = self.pv_recon(feature_maps, cam_mask, metas=data)
            self.vae_loss_dict = vae_loss
        else:
            self.vae_loss_dict = None

        # 4) 主任务 head：仍使用完整 feature_maps
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)

        # 5) 深度监督（如有）
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )

        # 6) 合并 VAE 自监督损失
        if self.vae_loss_dict is not None:
            output.update(self.vae_loss_dict)

        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(
            img,
            metas=data.get("img_metas", None),
        )
        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, data)
        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
