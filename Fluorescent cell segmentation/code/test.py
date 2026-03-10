import torch
import os, glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from tools4dataset import *
from model import *
from model_v2 import *
import matplotlib.pyplot as plt

def load_model(model_path, device):
    model = TemporalUNet(base = 32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, dataloader, device, save_root, dataset_mode):
    """
    运行推理并保存结果。
    特点：纯 FP32 最高精度推理、支持任意 Batch Size、适配 Image_XXXX 命名规则。
    """
    os.makedirs(save_root, exist_ok=True)

    for stack_tensor, stack_name_batch in tqdm(dataloader, desc="Inference"):
        # 将数据转入 GPU，保持默认的 FP32 高精度
        stack_tensor = stack_tensor.to(device, non_blocking=True)

        # 直接进行最高精度推理 (去除了 autocast)
        preds = model(stack_tensor)  # (B, 1, H, W)
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()

        # 批量转移至 CPU 并转换为 8-bit 图像矩阵 (纯数据搬迁，不损失精度)
        preds_np = preds.squeeze(1).cpu().numpy()
        preds_img_batch = (preds_np * 255).astype(np.uint8)

        current_batch_size = stack_tensor.size(0)

        # 遍历 Batch 中的每一张图
        for i in range(current_batch_size):
            if isinstance(stack_name_batch, (list, tuple)):
                current_full_name = stack_name_batch[i]
            else:
                current_full_name = stack_name_batch

            pred_img = preds_img_batch[i]

            if current_full_name.startswith("Image_"):
                # 截掉末尾的 4 位帧序号，提取文件夹名 (例如 "Image_0010000" -> "Image_001")
                stack_root_name = current_full_name[:-4]
            else:
                stack_root_name = "unknown_stack"

            filename_prefix = current_full_name

            stack_save_dir = os.path.join(save_root, stack_root_name)

            if not os.path.exists(stack_save_dir):
                os.makedirs(stack_save_dir, exist_ok=True)

            # 构造保存路径，例如: Image_001/Image_0010000_mask.tif
            save_path = os.path.join(stack_save_dir, f"{filename_prefix}_mask.tif")
            Image.fromarray(pred_img).save(save_path)


# -----------------------------
# 主函数 (需添加 mode 参数并传递给 run_inference)
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_root", type=str, required=True,
                        help="测试数据文件夹，包含 stack1, stack2, ...")
    parser.add_argument("--model_path", type=str, required=True,
                        help="训练好的模型权重路径 (.pth)")
    parser.add_argument("--outdir", type=str, default="./test_results",
                        help="输出结果保存路径")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="推理时batch大小（建议=1）")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", type=str, default="temporal", choices=["full", "temporal"],
                        help="数据集加载模式 (对应 TestStackDataset 的 mode)")
    parser.add_argument("--t_window", type=int, default=7)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.model_path, device)

    # **注意：将 mode 和 t_window 传递给 Dataset**
    test_dataset = TestStackDataset(
        test_root=args.test_root,
        mode=args.mode,  # 使用命令行参数的 mode
        t_window=args.t_window
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # **注意：将 mode 传递给 run_inference**
    run_inference(model, test_loader, device, args.outdir, args.mode)
    print(f"\n✅ 推理完成，结果已保存到: {args.outdir} 下的各自堆栈子文件夹中。")


if __name__ == "__main__":
    main()