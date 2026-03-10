import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import re
from typing import List, Dict, Tuple, Optional

# --- 新增导入 (用于裁剪) ---
import nibabel as nib
from skimage.measure import regionprops
from skimage import io
import tifffile

"""
本脚本实现基于多特征的时间序列聚类分析，并支持基于聚类结果的图像Patch裁剪。
主要功能包括：
1. 加载多个特征的时间序列数据。
2. 构建用于聚类的张量数据。
3. 使用K-Means算法进行聚类，支持不同的基线校正模式。
4. 绘制并保存聚类结果的原始数据曲线图。
5. 基于聚类结果裁剪图像Patch，支持全局坐标映射。
注意该脚本用于单个文件夹的聚类分析
"""


# --- 1. 全局配置 ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=1.1)


def check_and_create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"已创建输出目录: {path}")


# --- 2. 数据加载 ---
def load_feature_data(folder_path: str, feature_map: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    data_dict = {}
    base_shape = None
    path = Path(folder_path)

    print(f"\n--- [Step 1] 加载数据 ({folder_path}) ---")
    for feat_name, filename in feature_map.items():
        file_path = path / filename

        try:
            df = pd.read_csv(file_path, index_col=0)
        except Exception as e:
            raise ValueError(f"读取 {filename} 失败: {e}")

        if base_shape is None:
            base_shape = df.shape
        elif df.shape != base_shape:
            raise ValueError(f"文件 {filename} 维度 {df.shape} 不一致")

        data_dict[feat_name] = df
        print(f"  已加载 [{feat_name}]: {df.shape}")

    return data_dict


# --- 3. 张量构建 (提取原始数据) ---
def extract_tensor(
        data_dict: Dict[str, pd.DataFrame],
        stim_frames: List[int],
        cluster_window_size: int,
        view_window_count: int
) -> Tuple[np.ndarray, np.ndarray, List[tuple]]:
    """
    提取时间窗口。
    X_cluster, X_view 均包含【原始绝对值】。
    """
    features = list(data_dict.keys())
    cells = data_dict[features[0]].columns
    n_frames_total = data_dict[features[0]].shape[0]
    stim_indices = [x - 1 for x in stim_frames]

    long_window_size = cluster_window_size * view_window_count

    X_cluster_list = []
    X_view_list = []
    meta_info = []

    for cell in cells:
        for start_t in stim_indices:
            end_t_long = start_t + long_window_size

            if end_t_long <= n_frames_total:
                # 提取 (Time, Features)
                sample_cluster = np.stack([
                    data_dict[f].loc[start_t: start_t + cluster_window_size - 1, cell].values
                    for f in features
                ], axis=1)

                sample_view = np.stack([
                    data_dict[f].loc[start_t: end_t_long - 1, cell].values
                    for f in features
                ], axis=1)

                X_cluster_list.append(sample_cluster)
                X_view_list.append(sample_view)
                # 注意：这里存储的是 1-based 的刺激帧
                meta_info.append((cell, start_t + 1))

    return np.array(X_cluster_list), np.array(X_view_list), meta_info


# --- 4. 聚类算法 (内部处理：去基线+标准化) ---
def compute_clusters(
        X_cluster: np.ndarray,
        n_clusters: int,
        baseline_mode: str = 'window_min',
        meta_info: List[tuple] = None,
        feature_names: List[str] = None,
        data_dict: Dict[str, pd.DataFrame] = None
) -> np.ndarray:
    """
    仅负责计算聚类标签。
    内部会对数据进行【基线校正】和【标准化】，以保证分类是基于形态变化的。
    """
    print(f"\n--- [Step 2] 计算聚类 (模式: {baseline_mode}) ---")
    n_samples, _, n_features = X_cluster.shape

    # 1. 内部基线校正
    if baseline_mode == 'window_min':
        baselines = np.min(X_cluster, axis=1, keepdims=True)
        X_cluster_corr = X_cluster - baselines
        print("  > 已应用窗口最小值基线 (Window Min)")

    elif baseline_mode == 'global_min':
        print("  > 正在计算并应用全局最小值基线 (Global Min)...")
        if data_dict is None or meta_info is None or feature_names is None:
            raise ValueError("Global Min 模式需要提供 data_dict, meta_info 和 feature_names")

        global_mins_list = []
        for feat in feature_names:
            global_mins_list.append(data_dict[feat].min())

        baselines = np.zeros((n_samples, 1, n_features))

        for i, (cell_id, _) in enumerate(meta_info):
            for f_idx in range(n_features):
                val = global_mins_list[f_idx][cell_id]
                baselines[i, 0, f_idx] = val

        X_cluster_corr = X_cluster - baselines
    else:
        raise ValueError(f"未知的基线模式: {baseline_mode}")

    # 2. 内部标准化
    scaler = StandardScaler()
    X_fit = X_cluster_corr.reshape(-1, n_features)
    scaler.fit(X_fit)
    X_cluster_std = scaler.transform(X_cluster_corr.reshape(-1, n_features)).reshape(X_cluster.shape)

    # 3. K-Means
    X_input = X_cluster_std.reshape(n_samples, -1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_input)

    counts = pd.Series(labels).value_counts().sort_index()
    print(f"  聚类分布: \n{counts.to_string()}")

    return labels


# --- 5. 绘图与保存 (使用原始数据) ---
def save_and_plot_raw(
        X_view_raw: np.ndarray,
        labels: np.ndarray,
        meta_info: List[tuple],
        feature_names: List[str],
        output_dir: str,
        window_size: int,
        num_windows: int
):
    print(f"\n--- [Step 3] 绘制原始值曲线与保存 ({output_dir}) ---")
    out_path = Path(output_dir)
    n_clusters = len(np.unique(labels))
    n_features = len(feature_names)
    colors = plt.cm.tab10(np.arange(n_clusters))
    time_axis = np.arange(X_view_raw.shape[1])

    # A. 保存分类结果
    df_res = pd.DataFrame(meta_info, columns=['Cell_ID', 'Stim_Start'])
    df_res['Cluster'] = labels
    df_res.to_csv(out_path / 'cluster_results.csv', index=False)

    # B. 保存原始值的平均曲线
    center_data = {}
    for c in range(n_clusters):
        mask = labels == c
        mean_trace = X_view_raw[mask].mean(axis=0)
        for i, feat in enumerate(feature_names):
            center_data[f'Cluster{c}_{feat}'] = mean_trace[:, i]
    pd.DataFrame(center_data).to_csv(out_path / 'cluster_raw_traces.csv')
    print("  CSV数据已保存.")

    # C. 绘制独立子图
    print("  正在绘制独立特征子图...")
    for i, feat_name in enumerate(feature_names):
        plt.figure(figsize=(10, 7))

        for c in range(n_clusters):
            mask = labels == c
            data = X_view_raw[mask, :, i]
            mean_trace = data.mean(axis=0)
            sem = data.std(axis=0) / np.sqrt(data.shape[0])

            plt.plot(time_axis, mean_trace, label=f'Cluster {c} (n={mask.sum()})', color=colors[c], linewidth=3)
            plt.fill_between(time_axis, mean_trace - sem, mean_trace + sem, color=colors[c], alpha=0.15)

        plt.title(f"{feat_name} (Raw Values)", fontsize=20, fontweight='bold', pad=20)
        plt.xlabel("Frames (from Stimulus)", fontsize=16)
        plt.ylabel(f"Absolute Value", fontsize=16)

        plt.axvline(x=0, color='r', linestyle='--', label='Stim Start')
        plt.axvspan(0, 3, color='r', alpha=0.1)

        for w in range(1, num_windows):
            plt.axvline(x=w * window_size, color='gray', linestyle=':', alpha=0.6)
            plt.text(w * window_size, plt.ylim()[1], f' End W{w}', color='gray', ha='left', va='top', fontsize=10)

        plt.legend(loc='upper right', fontsize=12, frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        safe_name = feat_name.replace(" ", "_")
        plt.savefig(out_path / f'Plot_{safe_name}_Raw.png', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"    -> 已保存 Plot_{safe_name}_Raw.png")

    # D. 绘制组合大图
    print("  正在绘制组合概览图...")
    fig, axes = plt.subplots(nrows=n_features, ncols=1, figsize=(12, 4 * n_features), sharex=True)
    if n_features == 1: axes = [axes]

    for i, feat_name in enumerate(feature_names):
        ax = axes[i]
        for c in range(n_clusters):
            mask = labels == c
            mean_trace = X_view_raw[mask, :, i].mean(axis=0)
            ax.plot(time_axis, mean_trace, label=f'C{c}', color=colors[c], linewidth=2.5)

        ax.set_ylabel(f"{feat_name}\n(Raw)", fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.axvspan(0, 3, color='r', alpha=0.1)
        for w in range(1, num_windows):
            ax.axvline(x=w * window_size, color='gray', linestyle=':', alpha=0.5)

        if i == 0:
            ax.legend(loc='upper right', ncol=n_clusters, title="Cluster")
            ax.set_title(f"Multi-Feature Overview (Raw Values)", fontsize=18, pad=20)

    axes[-1].set_xlabel("Frames", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path / 'Summary_All_Features_Raw.png', bbox_inches='tight', dpi=150)
    plt.close()

    print(f"  绘图完成! 请查看文件夹: {output_dir}")


# =========================================================
# --- 新增: Patch 裁剪功能 (移植自 cluster_and_crop) ---
# =========================================================

def _load_sorted_image_files(folder: str, ext: str):
    """辅助函数：加载图像文件"""
    files = []
    pattern = re.compile(r'(\d+)')
    if not os.path.exists(folder):
        print(f"[Crop Error] 文件夹不存在 {folder}")
        return []

    for f in sorted(os.listdir(folder)):
        if f.endswith(ext) or f.endswith(ext.upper()):
            match = pattern.search(f)
            if match:
                frame_idx = int(re.findall(r'\d+', f)[-1])
                files.append((frame_idx, os.path.join(folder, f)))
    return sorted(files, key=lambda x: x[0])


def get_coordinates(mask_path: str, transpose_coords: bool) -> Dict[str, tuple]:
    """辅助函数：计算全局坐标"""
    print("\n--- [Step 4.1] 计算全局坐标 (Coordinate Mapping) ---")
    if not os.path.exists(mask_path):
        print(f"[Crop Error] 找不到全局掩码: {mask_path}")
        return {}

    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata().astype(int)

    if mask_data.ndim == 3:
        mask_2d = np.max(mask_data, axis=0)
    else:
        mask_2d = mask_data

    props = regionprops(mask_2d)
    coords = {}
    for prop in props:
        r, c = prop.centroid
        if transpose_coords:
            cx, cy = int(round(r)), int(round(c))
        else:
            cx, cy = int(round(c)), int(round(r))
        coords[str(prop.label)] = (cx, cy)

    print(f"  已计算 {len(coords)} 个细胞的质心坐标。")
    return coords


def debug_alignment(coords: Dict[str, tuple], image_files: list, output_dir: str, transpose_coords: bool):
    """辅助函数：生成对齐验证图"""
    if not image_files: return
    img = io.imread(image_files[0][1])
    plt.figure(figsize=(12, 12))
    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)

    xs = [pos[0] for pos in coords.values()]
    ys = [pos[1] for pos in coords.values()]

    plt.scatter(xs, ys, c='red', s=20, marker='x', label='Centroids')
    plt.title(f"Transpose={transpose_coords} (Red X should match cells)")
    plt.savefig(os.path.join(output_dir, '00_debug_alignment.png'))
    plt.close()
    print(f"  [重要] 对齐验证图已保存: {os.path.join(output_dir, '00_debug_alignment.png')}")


def crop_patches_simple(
        meta_info: List[tuple],
        labels: np.ndarray,
        coords: Dict[str, tuple],
        raw_image_dir: str,
        file_ext: str,
        output_dir: str,
        crop_window_size: int,
        patch_size: int,
        transpose_coords: bool
):
    """
    主裁剪函数
    """
    print("\n--- [Step 4.2] 图像裁剪 (Batch Cropping) ---")

    image_files = _load_sorted_image_files(raw_image_dir, file_ext)
    if not image_files:
        print("[Crop Error] 未找到图像文件，跳过裁剪。")
        return

    # 生成验证图
    debug_alignment(coords, image_files, output_dir, transpose_coords)

    image_map = {idx: path for idx, path in image_files}
    patch_r = patch_size // 2

    patches_dir = os.path.join(output_dir, 'image_patches')

    # 构建 DataFrame 方便处理
    df_clusters = pd.DataFrame(meta_info, columns=['cell', 'stim_frame'])
    df_clusters['cluster'] = labels

    n_clusters = len(np.unique(labels))
    for c in range(n_clusters):
        os.makedirs(os.path.join(patches_dir, f'Cluster_{c}'), exist_ok=True)

    total = len(df_clusters)
    print(f"  开始裁剪 {total} 个样本 (每个样本 {crop_window_size} 帧)...")

    count = 0
    for _, row in df_clusters.iterrows():
        cell_id = str(row['cell'])
        # 注意：extract_tensor 中存入的是 1-based frame，这里要小心使用
        stim_frame_1based = row['stim_frame']
        stim_frame_0based = stim_frame_1based - 1

        cluster = row['cluster']

        if cell_id not in coords: continue
        cx, cy = coords[cell_id]

        save_dir = os.path.join(patches_dir, f'Cluster_{cluster}')
        fname_base = f"cell{cell_id}_sti{stim_frame_1based}_cluster{cluster}"

        for t in range(crop_window_size):
            current_frame_0idx = stim_frame_0based + t

            if current_frame_0idx not in image_map: continue

            img = io.imread(image_map[current_frame_0idx])

            # 初始化 Patch
            if img.ndim == 3:
                patch = np.zeros((patch_size, patch_size, img.shape[2]), dtype=img.dtype)
            else:
                patch = np.zeros((patch_size, patch_size), dtype=img.dtype)

            src_y_start = cy - patch_r
            src_y_end = cy + patch_r
            src_x_start = cx - patch_r
            src_x_end = cx + patch_r
            dst_y_start, dst_y_end = 0, patch_size
            dst_x_start, dst_x_end = 0, patch_size

            h, w = img.shape[:2]
            if src_y_start < 0: dst_y_start = -src_y_start; src_y_start = 0
            if src_x_start < 0: dst_x_start = -src_x_start; src_x_start = 0
            if src_y_end > h: dst_y_end -= (src_y_end - h); src_y_end = h
            if src_x_end > w: dst_x_end -= (src_x_end - w); src_x_end = w

            if dst_y_end > dst_y_start and dst_x_end > dst_x_start:
                if img.ndim == 3:
                    patch[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = img[
                        src_y_start:src_y_end, src_x_start:src_x_end, :]
                else:
                    patch[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img[
                        src_y_start:src_y_end, src_x_start:src_x_end]

            # 文件名使用 1-based 索引
            fname_tif = f"{fname_base}_frame{current_frame_0idx + 1}.tif"
            tifffile.imwrite(os.path.join(save_dir, fname_tif), patch)

        count += 1
        if count % 50 == 0: print(f"    已处理 {count} 个样本...")

    print(f"  裁剪完成。Patch 保存在: {patches_dir}")


# ================= 运行配置 =================

def main():
    # 1. 特征数据路径
    DATA_FOLDER = r"pre_analyze/agonism_radiomics_features/Image 2/final_results_zero"
    FEATURE_FILES = {
        'Mean': 'Final_Mean.csv',
        'Area': 'Final_Area.csv',
        'Contrast': 'Final_Contrast.csv',
        'Entropy': 'Final_Entropy.csv',
        'Perimeter': 'Final_Perimeter.csv'
    }

    # 2. 输出设置
    OUTPUT_FOLDER = r"pre_analyze/results/jidong/Image 2 cluster_multifeatures_with_zero_padding_with_crop"

    # 3. 聚类核心参数
    STIMULI = [31, 42, 52, 63, 73, 84, 95, 105, 116, 126, 137, 148, 158, 169, 179]
    WINDOW_SIZE = 10
    CLUSTERS = 4
    VIEW_WINDOWS = 3
    BASELINE_MODE = 'global_min'

    # --- 新增: 裁剪相关配置 ---
    RAW_IMAGE_DIR = r"20251102_initial_code/data/data_20250817/agonism/Image 2"  # 请修改为真实路径
    FILE_EXTENSION = '.jpg'
    GLOBAL_STATIC_MASK = r"20251102_initial_code/data/data_20250817_tracked/agonism_mask/Image 2/global_static_mask/Global_Static_Mask.nii.gz"  # 请修改为真实路径
    CROP_WINDOW_SIZE = 32
    PATCH_SIZE = 32
    TRANSPOSE_COORDS = False  # Standard模式设为False
    # -------------------------

    check_and_create_dir(OUTPUT_FOLDER)

    try:
        # 1. 加载
        data_map = load_feature_data(DATA_FOLDER, FEATURE_FILES)

        # 2. 提取 (得到原始数据 X_cluster, X_view)
        X_cluster, X_view, meta = extract_tensor(data_map, STIMULI, WINDOW_SIZE, VIEW_WINDOWS)

        # 3. 聚类 (传入完整信息以支持 Global Min)
        labels = compute_clusters(
            X_cluster,
            CLUSTERS,
            baseline_mode=BASELINE_MODE,
            meta_info=meta,
            feature_names=list(FEATURE_FILES.keys()),
            data_dict=data_map
        )

        # 4. 绘图
        save_and_plot_raw(X_view, labels, meta, list(FEATURE_FILES.keys()), OUTPUT_FOLDER, WINDOW_SIZE, VIEW_WINDOWS)

        # --- [Step 4] 执行裁剪 ---
        coords = get_coordinates(GLOBAL_STATIC_MASK, TRANSPOSE_COORDS)
        crop_patches_simple(
            meta_info=meta,
            labels=labels,
            coords=coords,
            raw_image_dir=RAW_IMAGE_DIR,
            file_ext=FILE_EXTENSION,
            output_dir=OUTPUT_FOLDER,
            crop_window_size=CROP_WINDOW_SIZE,
            patch_size=PATCH_SIZE,
            transpose_coords=TRANSPOSE_COORDS
        )

    except Exception as e:
        print(f"\n[Error] 程序出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()