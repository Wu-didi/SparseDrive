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
except:
    DAF_VALID = False

__all__ = ["SparseDrive"]


# ================================
# 1) 随机相机失效（修正：可复现、返回缺失掩码）
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
        x :  [B, 6, 3, H, W]  float32  (0~1)
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
                n_drop = torch.randint(self.n_min, n_max + 1, (1,), generator=self._g, device=device).item()
                drop_ids = torch.randperm(N_cam, generator=self._g, device=device)[:n_drop]
                cam_mask[b, drop_ids] = True

        # 应用遮挡
        x = x * (~cam_mask).view(B, N_cam, 1, 1, 1).to(x.dtype)
        return (x, cam_mask) if return_mask else x


# ===========================================
# 2) 视角级特征补全（v0.1：其它相机均值 + 轻量历史）
#    - 先用同帧可见相机的特征均值补全缺失相机
#    - 再用上一帧同视角的历史先验做门控融合（极便宜）
#    - 可选一个极轻细化头（默认关闭）
# ===========================================
class PVReconMeanTemporal(nn.Module):
    """
    输入:
      feature_maps: List[Tensor]，每尺度 [B, V, C, H, W]
      cam_mask    : [B, V] bool，True=该视角缺失
    输出:
      feature_maps_same_shape（仅在 cam_mask=True 的视角进行了替换）
    """
    def __init__(self, use_refine: bool = False, ch_per_scale: list = None, hist_momentum: float = 1.0):
        super().__init__()
        self.use_refine = use_refine
        self.hist_momentum = hist_momentum
        if use_refine:
            assert ch_per_scale is not None, "use_refine=True 需要提供每个尺度的通道数"
            self.refiners = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
                    nn.Conv2d(c, c, 1, bias=False),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True)
                ) for c in ch_per_scale
            ])
        else:
            self.refiners = None

        # 历史缓冲：上一帧各尺度 PV 特征（不会参与梯度）
        self._hist = None  # List[Tensor] of [B,V,C,H,W]

    @torch.no_grad()
    def _mean_fill(self, F, cam_mask):
        # F: [B,V,C,H,W], cam_mask: [B,V] bool
        B, V, C, H, W = F.shape
        miss = cam_mask.view(B, V, 1, 1, 1).to(F.dtype)      # 1=缺失
        vis  = 1.0 - miss                                    # 1=可见
        vis_sum   = (F * vis).sum(dim=1, keepdim=True)       # [B,1,C,H,W]
        vis_count = vis.sum(dim=1, keepdim=True).clamp(min=1)
        mean_vis  = vis_sum / vis_count
        F_rec = F * vis + mean_vis * miss
        return F_rec

    def forward(self, feature_maps, cam_mask, metas=None):
        # 无缺失 -> 旁路
        if cam_mask is None or not cam_mask.any():
            # 更新历史
            if self._hist is None:
                self._hist = [f.detach() for f in feature_maps]
            else:
                # momentum 可选，这里默认1.0（直接替换）
                self._hist = [f.detach() * self.hist_momentum + h*(1-self.hist_momentum)
                              for f, h in zip(feature_maps, self._hist)]
            return feature_maps

        outs = []
        for i, F in enumerate(feature_maps):
            B, V, C, H, W = F.shape

            # 1) 同帧可见相机的均值回填
            F_rec = self._mean_fill(F, cam_mask)  # [B,V,C,H,W]

            # 2) 轻量历史先验（上一帧同视角）
            if self._hist is not None:
                F_hist = self._hist[i]  # [B,V,C,H,W]
                # 历史强度门控（上一帧可能也缺失）
                hist_w = (F_hist.abs().mean(dim=(2,3,4), keepdim=True) > 1e-6).to(F_rec.dtype)  # [B,V,1,1,1]
                # 软融合：缺失视角位置用 30% 历史 + 70% 同帧均值（仅在缺失处）
                miss = cam_mask.view(B, V, 1, 1, 1).to(F_rec.dtype)
                alpha = 0.3 * hist_w * miss
                F_rec = (1 - alpha) * F_rec + alpha * F_hist

            # 3) 细化头（只对缺失视角上跑）
            if self.refiners is not None:
                miss = cam_mask.view(B, V, 1, 1, 1).to(F_rec.dtype)
                F_ref = self.refiners[i](F_rec.view(B*V, C, H, W)).view(B, V, C, H, W)
                F_rec = F_rec * (1 - miss) + F_ref * miss

            outs.append(F_rec)

        # 更新历史
        if self._hist is None:
            self._hist = [f.detach() for f in outs]
        else:
            self._hist = [f.detach() * self.hist_momentum + h*(1-self.hist_momentum)
                          for f, h in zip(outs, self._hist)]
        return outs




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
    ):
        super(SparseDrive, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
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
            
            
        # ---------------- 随机相机失效 ----------------
        self.cam_dropout = RandCamMask(
            p_missing=0.6, n_min=1, n_max=2, train_only=True, seed=42
        )

        # ------------ 视角级特征补全模块（v0.1） ------------
        # 如果你的 neck 把各尺度通道统一到 256，这里就传 [256,256,256,256]
        # 若保持主干通道（如 256/512/1024/2048），按真实通道数组填写。
        self.pv_recon = PVReconMeanTemporal(
            use_refine=False,          # 先关闭细化头，省时；稳定后可 True
            ch_per_scale=None,         # use_refine=True 时需要填写，比如 [256,256,256,256]
            hist_momentum=1.0
        )

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)   # feature_maps是一个tuple
                                                    # feature_maps[0].shape     torch.Size([36, 256, 64, 176])
                                                    # feature_maps[1].shape     torch.Size([36, 512, 32, 88])
                                                    # feature_maps[2].shape     torch.Size([36, 1024, 16, 44])
                                                    # feature_maps[3].shape     torch.Size([36, 2048, 8, 22])
        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))    # feature_maps是一个list                                                     
                                                                # feature_maps[0].shape     torch.Size([36, 256, 64, 176])
                                                                # feature_maps[1].shape     torch.Size([36, 512, 32, 88])
                                                                # feature_maps[2].shape     torch.Size([36, 1024, 16, 44])
                                                                # feature_maps[3].shape     torch.Size([36, 2048, 8, 22])
        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(
                feat, (bs, num_cams) + feat.shape[1:]
            )
                # feature_maps[0].shape     torch.Size([6, 6, 256, 64, 176])
                # feature_maps[1].shape     torch.Size([6, 6, 256, 32, 88])
                # feature_maps[2].shape     torch.Size([6, 6, 1024, 16, 44])
                # feature_maps[3].shape     torch.Size([6, 6, 2048, 8, 22])
        

        # ====== 在这里做“视角级特征补全” ======
        # 仅当 cam_mask 有效且存在缺失时触发；否则旁路，不增加任何开销
        if (cam_mask is not None) and (cam_mask.dtype == torch.bool) and cam_mask.any():
            feature_maps = self.pv_recon(feature_maps, cam_mask, metas)
            
        
        
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
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
        # img shape torch.Size([8, 6, 3, 256, 704])
        #--------------------------start-------------------------------------------
        img = self.cam_dropout(img)
        #---------------------------end--------------------------------------------
        feature_maps, depths = self.extract_feat(img, True, data) # feature_maps是一len为3的list
                                                                  # feature_maps[0].shape torch.Size([6, 89760, 256])
                                                                  # feature_maps[1].shape torch.Size([6, 4, 2])
                                                                  # feature_maps[2].shape torch.Size([6, 4])
        
        model_outs = self.head(feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img)

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
