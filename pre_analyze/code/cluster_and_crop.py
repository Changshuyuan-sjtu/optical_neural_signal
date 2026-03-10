import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from skimage import io
import tifffile  # 用于保存原始数据 (需安装: pip install tifffile)

"""
此脚本实现了一个两步处理流程：
1. 重新聚类：基于之前计算的 Final_Mean.csv 文件，使用 K-Means 进行重新聚类，并生成聚类结果的可视化图。
2. 图像裁剪：根据聚类结果和全局静态掩码，裁剪出每个细胞在刺激时间窗口内的图像 Patch，并保存为 TIFF 文件。
该文件是基于单个图像文件夹进行处理的版本，适用于特定实验条件下的数据分析。
"""

# ================= 配置区域 (Configuration) =================

CONFIG = {
    # 1. 输入路径
    'RAW_IMAGE_DIR': r"20251102_initial_code/data/data_20250817/antagonism/Image 24",
    'FILE_EXTENSION': '.jpg',  # 图像文件扩展名

    # 之前生成的 Final_Mean.csv (用于聚类)
    'MEAN_CSV_PATH': r"pre_analyze/antagonism_radiomics_features/Image 24/final_results_zero/Final_Mean.csv",

    # 之前生成的全局静态掩码 (用于定位坐标)
    'GLOBAL_STATIC_MASK': r"20251102_initial_code/data/data_20250817_tracked/antagonism_mask/Image 24/global_static_mask/Global_Static_Mask.nii.gz",

    # 2. 输出路径
    'OUTPUT_DIR': r"pre_analyze/results/jiekang/Image 24 only intensity and crop",
    'TRANSPOSE_COORDS': False,
    # 3. 聚类参数
    'STIM_FRAMES': [31, 42, 52, 63, 73, 84, 95, 105, 116, 126, 137, 148, 158, 169, 179],  # 1-based
    'CLUSTER_WINDOW_SIZE': 11,  # 聚类用的窗口大小 (W1)
    'N_CLUSTERS': 4,  # 聚类簇数
    'NUM_WINDOWS_TO_VIEW': 3,  # 长视图中显示的窗口数量 (W1, W2, ...)
    'BASELINE_METHOD': 'global_min',  # 'min' (W1最小值) 或 't0'

    # 4. 裁剪参数
    'CROP_WINDOW_SIZE': 32,  # 裁剪的时间长度 (t0 ~ t0+32)
    'PATCH_SIZE': 32,  # 图像 Patch 大小 (32x32)
}


# ===========================================================

class ClusterAndCropPipeline:
    def __init__(self, config):
        self.cfg = config
        self.dirs = {
            'cluster_results': os.path.join(config['OUTPUT_DIR'], '01_clustering_results'),
            'patches': os.path.join(config['OUTPUT_DIR'], '02_image_patches')
        }
        for d in self.dirs.values(): os.makedirs(d, exist_ok=True)

    def _load_sorted_image_files(self):
        files = []
        pattern = re.compile(r'(\d+)')
        folder = self.cfg['RAW_IMAGE_DIR']
        ext = self.cfg['FILE_EXTENSION']
        for f in sorted(os.listdir(folder)):
            if f.endswith(ext) or f.endswith(ext.upper()):
                match = pattern.search(f)
                if match:
                    # 提取文件名中最后的数字串作为帧号
                    frame_idx = int(re.findall(r'\d+', f)[-1])
                    files.append((frame_idx, os.path.join(folder, f)))
        return sorted(files, key=lambda x: x[0])

    # ---------------------------------------------------------
    # 步骤 1: 重新聚类 (融合了 v2 的绘图逻辑)
    # ---------------------------------------------------------
    def step1_recluster(self):
        print("\n--- STEP 1: 重新聚类 (Re-Clustering & Visualization) ---")

        df = pd.read_csv(self.cfg['MEAN_CSV_PATH'], index_col=0)
        df.columns = df.columns.astype(str)
        print(f"  加载数据: {df.shape} (Frames x Cells)")

        cell_global_mins = df.min()

        stim_frames_0idx = [f - 1 for f in self.cfg['STIM_FRAMES']]
        w_size = self.cfg['CLUSTER_WINDOW_SIZE']
        long_w_size = w_size * self.cfg['NUM_WINDOWS_TO_VIEW']  # *** 计算长窗口 ***

        samples_w1 = []
        samples_long = []
        sample_info = []

        for cell_id in df.columns:
            for stim_start in stim_frames_0idx:
                end_w1 = stim_start + w_size
                end_long = stim_start + long_w_size

                # 必须保证长窗口也在数据范围内，否则不画图
                if end_long <= len(df):
                    win_w1 = df.loc[stim_start: end_w1 - 1, cell_id].values
                    win_long = df.loc[stim_start: end_long - 1, cell_id].values

                    if len(win_w1) == w_size and len(win_long) == long_w_size:
                        samples_w1.append(win_w1)
                        samples_long.append(win_long)
                        sample_info.append({'cell': cell_id, 'stim_frame': stim_start})

        X_w1 = np.array(samples_w1)
        X_long = np.array(samples_long)
        print(f"  提取到 {len(X_w1)} 个样本 (满足长窗口要求)。")

        # 基线校正
        if self.cfg['BASELINE_METHOD'] == 'global_min':
            baselines = np.array([cell_global_mins[str(info['cell'])] for info in sample_info]).reshape(-1, 1)
        elif self.cfg['BASELINE_METHOD'] == 'min':
            baselines = np.min(X_w1, axis=1, keepdims=True)
        elif self.cfg['BASELINE_METHOD'] == 't0':
            baselines = X_w1[:, 0:1]
        else:
            baselines = 0

        # 校正数据用于绘图和聚类
        X_w1_corr = X_w1 - baselines
        X_long_corr = X_long - baselines

        # K-Means (仅基于 W1)
        print(f"  正在执行 K-Means (k={self.cfg['N_CLUSTERS']})...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_w1_corr)

        kmeans = KMeans(n_clusters=self.cfg['N_CLUSTERS'], random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        results = pd.DataFrame(sample_info)
        results['cluster'] = labels
        results.to_csv(os.path.join(self.dirs['cluster_results'], 'new_clusters.csv'), index=False)

        # *** 调用高级绘图函数 ***
        self._plot_w1_view(X_w1_corr, labels)
        self._plot_long_view(X_long_corr, labels)

        return results

    # --- 绘图函数 (来自 v2) ---
    def _plot_w1_view(self, X, labels):
        n_clusters = self.cfg['N_CLUSTERS']
        w_size = self.cfg['CLUSTER_WINDOW_SIZE']
        colors = plt.cm.get_cmap('tab10', n_clusters)

        plt.figure(figsize=(12, 8))
        for i in range(n_clusters):
            cluster_data = X[labels == i]
            mean_trace = np.mean(cluster_data, axis=0)
            plt.plot(range(w_size), mean_trace,
                     label=f'Cluster {i} (n={len(cluster_data)})',
                     color=colors(i), linewidth=2)

        plt.axvline(x=0, color='red', linestyle='--', label='Stimulus Onset')
        plt.axvspan(0, 2, color='red', alpha=0.1, label='Stimulus Duration')
        plt.title(f'Cluster Average Response (W1: {w_size} frames)', fontsize=16)
        plt.xlabel('Time Relative to Stimulus Onset', fontsize=12)
        plt.ylabel(f'Intensity (relative to {self.cfg["BASELINE_METHOD"]})', fontsize=12)
        plt.legend(title='Cluster')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(self.dirs['cluster_results'], 'clusters_avg_w1.png'))
        plt.close()

    def _plot_long_view(self, X, labels):
        n_clusters = self.cfg['N_CLUSTERS']
        w_size = self.cfg['CLUSTER_WINDOW_SIZE']
        n_wins = self.cfg['NUM_WINDOWS_TO_VIEW']
        long_len = X.shape[1]
        colors = plt.cm.get_cmap('tab10', n_clusters)

        plt.figure(figsize=(10 + (n_wins - 1) * 4, 8))
        for i in range(n_clusters):
            cluster_data = X[labels == i]
            mean_trace = np.mean(cluster_data, axis=0)
            plt.plot(range(long_len), mean_trace,
                     label=f'Cluster {i} (n={len(cluster_data)})',
                     color=colors(i), linewidth=2)

        plt.axvline(x=0, color='red', linestyle='--', label='Stimulus Onset')
        plt.axvspan(0, 2, color='red', alpha=0.1, label='Stimulus Duration')

        # 标记窗口分割线
        for i in range(1, n_wins):
            x_pos = w_size * i
            plt.axvline(x=x_pos, color='blue', linestyle=':',
                        label=f'End of W{i}' if i == 1 else None)

        plt.title(f'Cluster Average Response (Long View: {n_wins} Windows)', fontsize=16)
        plt.xlabel('Time Relative to Stimulus Onset', fontsize=12)
        plt.ylabel(f'Intensity (relative to {self.cfg["BASELINE_METHOD"]})', fontsize=12)

        handles, lbs = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(lbs, handles))
        plt.legend(by_label.values(), by_label.keys(), title='Cluster', loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(self.dirs['cluster_results'], 'clusters_avg_long_view.png'))
        plt.close()

    # ---------------------------------------------------------
    # 步骤 2: 计算坐标 (含 v3 的转置逻辑)
    # ---------------------------------------------------------
    def step2_get_coordinates(self):
        print("\n--- STEP 2: 计算全局坐标 ---")
        mask_img = nib.load(self.cfg['GLOBAL_STATIC_MASK'])
        mask_data = mask_img.get_fdata().astype(int)

        if mask_data.ndim == 3:
            mask_2d = np.max(mask_data, axis=0)
        else:
            mask_2d = mask_data

        props = regionprops(mask_2d)
        coords = {}
        for prop in props:
            r, c = prop.centroid
            if self.cfg['TRANSPOSE_COORDS']:
                cx, cy = int(round(r)), int(round(c))
            else:
                cx, cy = int(round(c)), int(round(r))
            coords[str(prop.label)] = (cx, cy)

        print(f"  已计算 {len(coords)} 个细胞坐标。")
        return coords

    def step2_5_debug_alignment(self, coords, image_files):
        print("\n--- STEP 2.5: 生成对齐验证图 ---")
        if not image_files: return
        img = io.imread(image_files[0][1])
        plt.figure(figsize=(12, 12))
        plt.imshow(img, cmap='gray' if img.ndim == 2 else None)

        xs = [pos[0] for pos in coords.values()]
        ys = [pos[1] for pos in coords.values()]
        plt.scatter(xs, ys, c='red', s=20, marker='x', label='Centroids')

        plt.title(f"Transpose={self.cfg['TRANSPOSE_COORDS']} (Red X should match cells)")
        plt.savefig(os.path.join(self.cfg['OUTPUT_DIR'], '00_debug_alignment.png'))
        plt.close()
        print(f"  [重要] 请检查: 00_debug_alignment.png")

    # ---------------------------------------------------------
    # 步骤 3: 裁剪 (含 v3 的原始图像读取)
    # ---------------------------------------------------------
    def step3_crop_patches(self, cluster_df, coords):
        print("\n--- STEP 3: 图像裁剪 ---")
        image_files = self._load_sorted_image_files()
        if not image_files: return

        # 生成验证图
        self.step2_5_debug_alignment(coords, image_files)

        image_map = {idx: path for idx, path in image_files}
        patch_size = self.cfg['PATCH_SIZE']
        patch_r = patch_size // 2
        crop_len = self.cfg['CROP_WINDOW_SIZE']

        for c in range(self.cfg['N_CLUSTERS']):
            os.makedirs(os.path.join(self.dirs['patches'], f'Cluster_{c}'), exist_ok=True)

        print(f"  开始裁剪 {len(cluster_df)} 个样本...")
        count = 0

        for _, row in cluster_df.iterrows():
            cell_id = str(row['cell'])
            stim_frame = row['stim_frame']
            cluster = row['cluster']

            if cell_id not in coords: continue
            cx, cy = coords[cell_id]

            save_dir = os.path.join(self.dirs['patches'], f'Cluster_{cluster}')

            for t in range(crop_len):
                current_frame = stim_frame + t
                if current_frame not in image_map: continue

                img = io.imread(image_map[current_frame])

                # 处理通道
                if img.ndim == 3:
                    patch = np.zeros((patch_size, patch_size, img.shape[2]), dtype=img.dtype)
                else:
                    patch = np.zeros((patch_size, patch_size), dtype=img.dtype)

                # 坐标计算
                src_y_start = cy - patch_r;
                src_y_end = cy + patch_r
                src_x_start = cx - patch_r;
                src_x_end = cx + patch_r
                dst_y_start, dst_y_end = 0, patch_size
                dst_x_start, dst_x_end = 0, patch_size

                # Padding 逻辑
                h, w = img.shape[:2]
                if src_y_start < 0: dst_y_start = -src_y_start; src_y_start = 0
                if src_x_start < 0: dst_x_start = -src_x_start; src_x_start = 0
                if src_y_end > h: dst_y_end -= (src_y_end - h); src_y_end = h
                if src_x_end > w: dst_x_end -= (src_x_end - w); src_x_end = w

                if dst_y_end > dst_y_start and dst_x_end > dst_x_start:
                    if img.ndim == 3:
                        patch[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = \
                            img[src_y_start:src_y_end, src_x_start:src_x_end, :]
                    else:
                        patch[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                            img[src_y_start:src_y_end, src_x_start:src_x_end]

                fname = f"cell{cell_id}_sti{stim_frame + 1}_cluster{cluster}_frame{current_frame + 1}.tif"
                tifffile.imwrite(os.path.join(save_dir, fname), patch)

            count += 1
            if count % 50 == 0: print(f"    已处理 {count} 个样本...")

        print(f"  全部完成。")

    def run(self):
        cluster_results = self.step1_recluster()
        coords = self.step2_get_coordinates()
        self.step3_crop_patches(cluster_results, coords)


if __name__ == "__main__":
    try:
        import tifffile
    except ImportError:
        print("错误: 请先安装 tifffile (pip install tifffile)。")
        exit()
    pipeline = ClusterAndCropPipeline(CONFIG)
    pipeline.run()