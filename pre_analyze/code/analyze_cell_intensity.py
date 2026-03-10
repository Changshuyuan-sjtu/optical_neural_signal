import os
import numpy as np
from scipy import ndimage
from skimage import io, measure
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def analyze_cell_intensity(original_dir, mask_dir, output_dir, rise_threshold=0.7):
    """
    根据190帧掩膜图确定全局细胞位置，计算每个细胞区域随时间的变化特征。

    参数：
    - original_dir: 原始图像文件夹
    - mask_dir: 掩码文件夹
    - output_dir: 输出文件夹
    - rise_threshold: 亮度阈值（默认0.6，即最大值的60%）
    """
    os.makedirs(output_dir, exist_ok=True)

    original_files = sorted(
        [f for f in os.listdir(original_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    mask_files = sorted(
        [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    if len(original_files) != 190 or len(mask_files) != 190:
        raise ValueError("文件数量不是190帧，请检查文件夹。")

    # 初始化全局细胞掩码（联合所有帧的掩码）
    global_mask = None
    cell_positions = {}  # {cell_id: mask}

    # 累积所有帧的掩码
    for frame_idx in range(190):
        mask_path = os.path.join(mask_dir, mask_files[frame_idx])
        mask_image = io.imread(mask_path, as_gray=True)
        mask_binary = mask_image > 0
        labeled_mask, num_labels = ndimage.label(mask_binary)

        if global_mask is None:
            global_mask = np.zeros_like(mask_image, dtype=bool)

        # 标记所有帧中出现过的像素
        for label in range(1, num_labels + 1):
            cell_mask = (labeled_mask == label)
            global_mask = np.logical_or(global_mask, cell_mask)
        Image.fromarray(global_mask.astype(np.uint8) * 255).save(os.path.join(output_dir, "global_mask_by_or.tif"))
        #io.imsave(os.path.join(output_dir, "global_mask_by_or.tif"), global_mask)

    # 重新标签全局掩码，确定唯一细胞ID
    labeled_global, num_cells = ndimage.label(global_mask)

    for cell_id in range(1, num_cells + 1):
        cell_positions[cell_id] = (labeled_global == cell_id)

    # 计算每帧每个细胞的亮度
    cell_intensities = {cell_id: [] for cell_id in range(1, num_cells + 1)}

    for frame_idx in range(190):
        orig_path = os.path.join(original_dir, original_files[frame_idx])
        original_image = io.imread(orig_path)

        if len(original_image.shape) == 3:
            intensity_image = np.sum(original_image, axis=2)  # 三通道平均
        else:
            intensity_image = original_image

        for cell_id, mask in cell_positions.items():
            if mask.any():  # 确保掩码非空
                mean_intensity = np.mean(intensity_image[mask])
                cell_intensities[cell_id].append(mean_intensity)

    # 填充缺失帧（如果某细胞未在某些帧检测到，填0或NaN）
    max_length = 190
    for cell_id in cell_intensities:
        while len(cell_intensities[cell_id]) < max_length:
            cell_intensities[cell_id].append(np.nan)  # 或0，根据需求

    # 计算持续亮度时长和统计特征
    bright_durations = []
    cell_stats = {}  # {cell_id: {'mean': float, 'max': float, 'min': float, 'duration': int}}
    for cell_id, signal in cell_intensities.items():
        signal = np.array(signal)
        max_intensity = np.max(signal)
        bright_threshold = max_intensity * rise_threshold
        bright_frames = np.sum(signal >= bright_threshold)
        bright_durations.append(bright_frames)

        cell_stats[cell_id] = {
            'mean': np.mean(signal),
            'max': max_intensity,
            'min': np.min(signal),
            'duration': bright_frames
        }

    if bright_durations:
        mean_duration = np.mean(bright_durations)
        median_duration = np.median(bright_durations)
        std_duration = np.std(bright_durations)
        print(f"mean_duration: {mean_duration:.2f} 帧")
        print(f"median_duration: {median_duration:.2f} 帧")
        print(f"std_duration: {std_duration:.2f} 帧")

        plt.figure()
        plt.hist(bright_durations, bins=20)
        plt.title('Distribution of cell sustained brightness duration')
        plt.xlabel('frames')
        plt.ylabel('cell counts')
        plt.savefig(os.path.join(output_dir, 'bright_duration_hist.png'))
    else:
        print("无有效亮度数据")

    # 保存CSV
    df = pd.DataFrame.from_dict(cell_stats, orient='index')
    df.to_csv(os.path.join(output_dir, 'cell_stats.csv'))

    # 保存亮度序列
    df_intensity = pd.DataFrame(cell_intensities)
    df_intensity.to_csv(os.path.join(output_dir, 'cell_intensities.csv'))

    return num_cells, bright_durations, mean_duration


if __name__ == "__main__":

    ids = [5,7,9,12,18,24,32]
    for id in ids:
        print(f"jiekang {id}:")
        original_folder = f"Intermediate experimental results/activity_map_jiekang/activity_map_jiekang_{id}/correct_images_rollingball"
        mask_folder = f"semantic_label/label_jiekang/Image {id} further_filtered"
        output_path = f"Intermediate experimental results/activity_map_jiekang/activity_map_jiekang_{id}"
        _, _, mean = analyze_cell_intensity(original_folder, mask_folder, output_path)

