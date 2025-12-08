#!/usr/bin/env python3
"""
超快速测试脚本 - 只测试前10个batch，验证代码是否正常运行
"""
import sys
import torch
from mmcv import Config
from mmdet.apis import init_detector
from mmdet.datasets import build_dataloader, build_dataset
from mmcv.parallel import MMDataParallel

def quick_test():
    config_file = "projects/configs/sparsedrive_small_stage2.py"
    checkpoint_file = "work_dirs/sparsedrive_small_stage2/iter_70320.pth"

    print("=== 超快速测试：只测试前10个batch ===")
    print(f"配置文件: {config_file}")
    print(f"Checkpoint: {checkpoint_file}")

    # 加载配置
    cfg = Config.fromfile(config_file)

    # 构建模型
    print("\n1. 加载模型...")
    model = init_detector(cfg, checkpoint_file, device='cuda:0')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # 构建数据集
    print("2. 构建数据集...")
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # 测试前10个batch
    print("3. 测试前10个batch...")
    max_batches = 10
    success_count = 0

    for i, data in enumerate(data_loader):
        if i >= max_batches:
            break

        try:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            success_count += 1
            print(f"   Batch {i+1}/{max_batches}: ✓")
        except Exception as e:
            print(f"   Batch {i+1}/{max_batches}: ✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    print(f"\n✓ 成功测试 {success_count}/{max_batches} 个batch")
    print("修复成功！代码可以正常运行。")
    return True

if __name__ == "__main__":
    try:
        success = quick_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
