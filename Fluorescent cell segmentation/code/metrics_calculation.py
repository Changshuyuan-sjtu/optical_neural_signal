import numpy as np
import os
import glob
from PIL import Image
from collections import defaultdict
import argparse


# -------------------------------------------------------------------
# 核心指标计算函数 (This part remains unchanged)
# -------------------------------------------------------------------
def calculate_dice(pred_mask, gt_mask, smooth=1e-6):
    pred_mask = pred_mask.astype(np.float64)
    gt_mask = gt_mask.astype(np.float64)
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice


def calculate_iou(pred_mask, gt_mask, smooth=1e-6):
    pred_mask = pred_mask.astype(np.float64)
    gt_mask = gt_mask.astype(np.float64)
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def calculate_precision_recall_f1(pred_mask, gt_mask, smooth=1e-6):
    pred_mask = pred_mask.astype(np.float64)
    gt_mask = gt_mask.astype(np.float64)
    tp = np.sum(pred_mask * gt_mask)
    fp = np.sum(pred_mask * (1 - gt_mask))
    fn = np.sum((1 - pred_mask) * gt_mask)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1_denominator = precision + recall
    f1 = (2 * precision * recall) / f1_denominator if f1_denominator > 0 else 0.0
    return precision, recall, f1


def load_mask(mask_path):
    try:
        mask = np.array(Image.open(mask_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件: {mask_path}")
    except Exception as e:
        raise IOError(f"加载图像出错 {mask_path}: {e}")

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(np.float32)
    return mask


def evaluate_single_pair(pred_path, gt_path):
    pred_mask = load_mask(pred_path)
    gt_mask = load_mask(gt_path)

    if pred_mask.shape != gt_mask.shape:
        print(
            f"[警告] 掩膜尺寸不匹配: 预测 {pred_mask.shape}, GT {gt_mask.shape} (文件: {os.path.basename(pred_path)}). 跳过此对。")
        return None

    precision, recall, f1 = calculate_precision_recall_f1(pred_mask, gt_mask)
    return {
        'dice': calculate_dice(pred_mask, gt_mask),
        'iou': calculate_iou(pred_mask, gt_mask),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# -------------------------------------------------------------------
# KEY CHANGE: New evaluation function matching by sorted order
# -------------------------------------------------------------------
def evaluate_folder_by_position(pred_dir, gt_dir):
    """
    Evaluates paired masks from two folders by matching them based on their
    alphabetically sorted file order.
    """
    print("=" * 50)
    print(f"开始按位置匹配评估文件夹...")
    print(f"预测 (Pred) 文件夹: {pred_dir}")
    print(f"真值 (GT)  文件夹: {gt_dir}")
    print("=" * 50)

    # Get a sorted list of image files from both directories
    # This filters for common image types and ignores other files
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
    pred_paths = []
    for ext in image_extensions:
        pred_paths.extend(glob.glob(os.path.join(pred_dir, ext)))

    gt_paths = []
    for ext in image_extensions:
        gt_paths.extend(glob.glob(os.path.join(gt_dir, ext)))

    # Sort the lists alphabetically to ensure consistent pairing
    pred_paths.sort()
    gt_paths.sort()

    # --- Crucial Check for Mismatched File Counts ---
    if not pred_paths or not gt_paths:
        print("错误: 一个或两个文件夹中没有找到任何图像文件。")
        return None

    if len(pred_paths) != len(gt_paths):
        print(f"[警告] 文件数量不匹配!")
        print(f"  - 预测文件夹找到: {len(pred_paths)} 个文件")
        print(f"  - GT 文件夹找到:   {len(gt_paths)} 个文件")
        print(f"将只评估前 {min(len(pred_paths), len(gt_paths))} 对匹配的文件。")

    all_results = defaultdict(list)
    successful_evaluations = 0

    # Use zip to pair the sorted lists. It will automatically stop at the end of the shorter list.
    print("按以下顺序匹配文件进行评估:")
    for pred_path, gt_path in zip(pred_paths, gt_paths):
        pred_filename = os.path.basename(pred_path)
        gt_filename = os.path.basename(gt_path)
        print(f"  - PRED: {pred_filename:<30} <--> GT: {gt_filename}")

        try:
            results = evaluate_single_pair(pred_path, gt_path)
            if results:
                for key, value in results.items():
                    all_results[key].append(value)
                successful_evaluations += 1
        except Exception as e:
            print(f"[错误] 处理文件对失败 {pred_filename} & {gt_filename}。错误: {e}")

    if successful_evaluations == 0:
        print("\n评估失败: 没有成功处理任何文件对。")
        return None

    # Calculate average metrics
    avg_results = {key: np.mean(values) for key, values in all_results.items()}

    print("-" * 50)
    print(f"成功评估了 {successful_evaluations} 对掩膜。")
    return avg_results


def print_average_results(avg_results, folder_name):
    """Prints the average results for a folder."""
    print("\n" + "=" * 50)
    print(f"文件夹 '{folder_name}' 的平均评估结果")
    print("=" * 50)
    print(f"平均Dice系数:    {avg_results['dice']:.4f}")
    print(f"平均IoU:         {avg_results['iou']:.4f}")
    print(f"平均精确度:      {avg_results['precision']:.4f}")
    print(f"平均召回率:      {avg_results['recall']:.4f}")
    print(f"平均F1分数:      {avg_results['f1']:.4f}")
    print("=" * 50)


# -------------------------------------------------------------------
# Main execution block
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='通过位置匹配计算文件夹内分割指标的平均值')
    parser.add_argument('--pred', type=str, required=True, help='包含预测掩膜的文件夹路径')
    parser.add_argument('--gt', type=str, required=True, help='包含真实掩膜的文件夹路径')
    args = parser.parse_args()

    if not os.path.isdir(args.pred) or not os.path.isdir(args.gt):
        print(f"错误: --pred '{args.pred}' 或 --gt '{args.gt}' 不是有效的文件夹路径。")
    else:
        try:
            average_results = evaluate_folder_by_position(args.pred, args.gt)
            if average_results:
                folder_name = os.path.basename(os.path.normpath(args.pred))
                print_average_results(average_results, folder_name)
        except Exception as e:
            print(f"评估过程中出现致命错误: {e}")