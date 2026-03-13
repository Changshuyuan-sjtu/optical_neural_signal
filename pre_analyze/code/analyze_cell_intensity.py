import os
import re
import math
import numpy as np
import pandas as pd
import nibabel as nib
from skimage import io
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tifffile
from concurrent.futures import ThreadPoolExecutor
import warnings
import datetime

"""
作者: Changsy
本脚本实现了一个完整的细胞强度分析流程，包含以下功能模块：
1. 全时程均值曲线绘制 (模式 A)
2. 基于绝对时间映射的窗口聚类分析 (模式 B)
3. 基于聚类结果的多文件夹图像裁剪 (模式 C)
配置项详见 CONFIG 字典，用户可根据需要调整路径、分析参数和运行模式。
"""

warnings.filterwarnings('ignore')

# =====================================================================
#                          中央配置字典 (CONFIG)
# =====================================================================
CONFIG = {
    # ---------------- 1. 工作模式 ----------------
    # 可选:
    #   'full_time_course': 仅画每个文件夹的全时程平均曲线
    #   'cluster_only': 仅提取窗口聚类并画窗口平均曲线
    #   'cluster_and_crop': 执行聚类，并根据结果到原图裁剪 Patch (开启多线程)
    'RUN_MODE': 'cluster_only',

    # ---------------- 2. 路径配置 (全部采用绝对路径) ----------------
    # 数据集根目录
    'CSV_ROOT_DIR': r'/home/changsy/Optical_neural_signal/pre_analyze/new_data_20260309_features',                         # 存放 cell_intensities.csv 的根目录
    'IMG_ROOT_DIR': r'/home/data/changsy/optical_neural_signal/new_data_20260309/data_processed_relabel_jpg',              # 存放原始 jpg 序列的根目录
    'MASK_ROOT_DIR': r'/home/data/changsy/optical_neural_signal/new_data_20260309/instance_mask_tracked',                  # 存放掩码 .nii.gz 的根目录

    # 统一输出根目录 (建议单独建一个文件夹存放所有生成结果)
    'OUTPUT_DIR': r'/home/changsy/Optical_neural_signal/pre_analyze/results/new_data_20260309/unified_output_intensity',

    # 指定要处理的文件夹名称列表。如果设为 None，则自动处理所有找得到的 Image X 文件夹
    # 例如: ['Image 1', 'Image 19'] 或 None
    'TARGET_FOLDERS': None,

    'FILE_EXTENSION': '.jpg',

    # 子文件夹内的相对路径
    'TARGET_CSV_NAME': 'final_results_zero/Final_Mean.csv',
    'MASK_RELATIVE_PATH': 'global_static_mask/Global_Static_Mask.nii.gz',

    # ---------------- 3. 实验范式与时间参数 (绝对时间校准) ----------------
    'FPS_ACTUAL': 1.0 / 0.94556,    # 实际帧率 (约 1.057 fps)
    'TIME_PER_FRAME': 0.94556,      # 实际单帧耗时 (秒)

    'TIME_INIT_BLANK': 3.696,       # 初始灰屏时间 (秒)
    'TIME_STIM_ON': 3.696,          # 单次光栅刺激时间 (秒)
    'TIME_STIM_OFF': 7.392,         # 单次灰屏间隔时间 (秒)

    'NUM_DIRECTIONS': 8,            # 刺激方向数量 (0°, 45°, ..., 315°)
    'REPETITIONS_PER_DIR': 15,      # 每个方向的重复次数
    'SEQUENCE_TYPE': 'block',       # 刺激序列类型: 'block' (每个方向连续重复) 或 'interleaved' (交替)

    # ---------------- 4. 分析与聚类参数 ----------------
    'WINDOW_SIZE': 11,              # 统一固定的分析窗口长度 (帧)
    'N_CLUSTERS': 4,                # 聚类数量
    'NUM_WINDOWS_TO_VIEW': 2,       # 长视图查看的窗口数量 (用于画图)
    # 基线校正方法: 'none', 'min' (窗口最小值), 'global_min' (全局最小值, 已排除尾部空窗), 't0' (窗口第一帧), 'min_max', 'min_max_t0'
    'BASELINE_METHOD': 'min_max',
    'RANDOM_STATE': 42,

    # ---------------- 5. 图像裁剪与核验参数 (Mode C 专用) ----------------
    'CROP_WINDOW_SIZE': 11,         # 从刺激起始点向后裁剪的总帧数
    'PATCH_SIZE': 32,               # 裁剪的图像块大小 (32x32)
    'TRANSPOSE_COORDS': False,      # 行列反转开关 (用于修正 .nii.gz 坐标轴)
    'DEBUG_ALIGNMENT': True,        # 是否生成红叉对齐核验图
    'MAX_WORKERS': 8,               # 多线程裁剪时的最大工作线程数
}


# =====================================================================
#                          核心功能类
# =====================================================================

class CellAnalyzerPipeline:
    def __init__(self, config):
        self.cfg = config
        self.events = self._generate_stimulus_events()
        self.dirs = self._setup_directories()

    def _setup_directories(self):
        """设置规范的、带参数标识的防覆盖输出目录结构"""
        # 1. 生成基于时间戳和关键参数的动态文件夹名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = self.cfg['RUN_MODE']

        if mode == 'full_time_course':
            run_name = f"{timestamp}_{mode}_bsl-{self.cfg['BASELINE_METHOD']}"
        else:
            run_name = f"{timestamp}_{mode}_k{self.cfg['N_CLUSTERS']}_w{self.cfg['WINDOW_SIZE']}_bsl-{self.cfg['BASELINE_METHOD']}"

        # 2. 将动态文件夹拼接到统一的输出根目录下
        base_dir = os.path.join(self.cfg['OUTPUT_DIR'], run_name)

        dirs = {
            'base': base_dir,
            'csv': os.path.join(base_dir, '01_csv_reports'),
            'plots': os.path.join(base_dir, '02_visualizations'),
            'patches': os.path.join(base_dir, '03_image_patches'),
            'debug': os.path.join(base_dir, '00_debug_info')
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        print(f"[*] 已创建独立输出目录: {base_dir}")
        return dirs

    def _generate_stimulus_events(self):
        """基于绝对物理时间，计算向上取整的刺激帧和元数据"""
        events = []
        total_trials = self.cfg['NUM_DIRECTIONS'] * self.cfg['REPETITIONS_PER_DIR']
        cycle_time = self.cfg['TIME_STIM_ON'] + self.cfg['TIME_STIM_OFF']

        for i in range(total_trials):
            t_start_sec = self.cfg['TIME_INIT_BLANK'] + i * cycle_time
            frame_idx_0based = math.ceil(t_start_sec / self.cfg['TIME_PER_FRAME'])

            if self.cfg['SEQUENCE_TYPE'] == 'block':
                dir_idx = i // self.cfg['REPETITIONS_PER_DIR']
                rep_idx = i % self.cfg['REPETITIONS_PER_DIR']
            else:
                dir_idx = i % self.cfg['NUM_DIRECTIONS']
                rep_idx = i // self.cfg['NUM_DIRECTIONS']

            degree = dir_idx * 45

            events.append({
                'trial_id': i,
                'stim_frame_0idx': frame_idx_0based,
                'direction': degree,
                'repetition': rep_idx + 1
            })
        print(f"[*] 成功计算 {len(events)} 次刺激事件物理映射。")
        return events

    def _get_target_folders(self):
        """获取需要处理的文件夹列表"""
        if self.cfg['TARGET_FOLDERS'] is not None:
            return self.cfg['TARGET_FOLDERS']

        folders = []
        for d in os.listdir(self.cfg['CSV_ROOT_DIR']):
            if os.path.isdir(os.path.join(self.cfg['CSV_ROOT_DIR'], d)):
                folders.append(d)
        return sorted(folders)

    def _load_and_combine_csvs(self, folders):
        """加载多个文件夹的CSV并横向拼接，防止细胞重名，并统一对齐帧数"""
        all_dfs = []
        folder_cell_map = {}
        TARGET_FRAMES = 1500  # 统一对齐到 1500 帧

        for folder in folders:
            csv_path = os.path.join(self.cfg['CSV_ROOT_DIR'], folder, self.cfg['TARGET_CSV_NAME'])
            if not os.path.exists(csv_path):
                print(f"[!] 警告: 未找到 {csv_path}，跳过该文件夹。")
                continue
            try:
                df = pd.read_csv(csv_path, index_col=0)

                # === 新增核心修复：强制对齐到 1500 帧 ===
                # 重新构建 0 到 1499 的索引。
                # 超过 1500 的部分被直接截断；不足 1500 的部分用最后一帧的数据平铺填充。
                df = df.reindex(range(TARGET_FRAMES), method='ffill')
                # ========================================

                new_columns = []
                for col in df.columns:
                    new_id = f"{folder}__cell_{col}"
                    new_columns.append(new_id)
                    folder_cell_map[new_id] = {'folder': folder, 'original_cell_id': str(col)}
                df.columns = new_columns
                all_dfs.append(df)
            except Exception as e:
                print(f"[!] 加载 {folder} 失败: {e}")

        if not all_dfs:
            raise ValueError("未能加载任何有效的 CSV 数据。请检查 CSV_ROOT_DIR 和 TARGET_CSV_NAME。")

        combined_df = pd.concat(all_dfs, axis=1)
        combined_df = combined_df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        print(
            f"[*] 成功合并 {len(all_dfs)} 个文件夹数据，已统一对齐至 {TARGET_FRAMES} 帧，共计 {combined_df.shape[1]} 个细胞。")
        return combined_df, folder_cell_map

    # -----------------------------------------------------------------
    # 模式 A: 全时程均值曲线绘制
    # -----------------------------------------------------------------
    def run_full_time_course(self):
        print("\n--- [模式 A] 绘制文件夹级别全时程均值曲线 ---")
        folders = self._get_target_folders()

        for folder in folders:
            csv_path = os.path.join(self.cfg['CSV_ROOT_DIR'], folder, self.cfg['TARGET_CSV_NAME'])
            if not os.path.exists(csv_path): continue

            df = pd.read_csv(csv_path, index_col=0)

            if self.cfg['BASELINE_METHOD'] == 'global_min':
                baseline = df.min()
                df_corr = df - baseline
            elif self.cfg['BASELINE_METHOD'] == 't0':
                baseline = df.iloc[0, :]
                df_corr = df - baseline
            elif self.cfg['BASELINE_METHOD'] == 'none':
                df_corr = df
            else:
                df_corr = df

            mean_curve = df_corr.mean(axis=1)

            plt.figure(figsize=(15, 5))
            plt.plot(mean_curve.index, mean_curve.values, color='teal', linewidth=1.5)
            plt.title(f'Full Time-Course Average Intensity - {folder}', fontsize=16)
            plt.xlabel('Frame Number', fontsize=12)
            plt.ylabel(f'Intensity ({self.cfg["BASELINE_METHOD"]} corrected)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)

            for ev in self.events:
                plt.axvline(x=ev['stim_frame_0idx'], color='red', linestyle=':', alpha=0.3)

            out_png = os.path.join(self.dirs['plots'], f'full_time_course_{folder}.png')
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"  -> 已保存全时程曲线: {out_png}")

    # -----------------------------------------------------------------
    # 模式 B & C 的共用聚类逻辑
    # -----------------------------------------------------------------
    def run_clustering(self):
        print("\n--- [聚类分析阶段] 开始提取窗口特征并聚类 ---")
        folders = self._get_target_folders()
        df_combined, folder_cell_map = self._load_and_combine_csvs(folders)

        cell_ids = df_combined.columns
        w_size = self.cfg['WINDOW_SIZE']
        long_w_size = w_size * self.cfg['NUM_WINDOWS_TO_VIEW']

        all_windows_w1 = []
        all_windows_long = []
        sample_meta = []

        print("  正在根据绝对时间映射提取响应窗口...")
        for cell in cell_ids:
            for ev in self.events:
                start_frame = ev['stim_frame_0idx']
                end_frame_w1 = start_frame + w_size
                end_frame_long = start_frame + long_w_size

                if end_frame_long <= df_combined.index.max():
                    win_data_w1 = df_combined.loc[start_frame: end_frame_w1 - 1, cell].values
                    win_data_long = df_combined.loc[start_frame: end_frame_long - 1, cell].values

                    if len(win_data_w1) == w_size and len(win_data_long) == long_w_size:
                        all_windows_w1.append(win_data_w1)
                        all_windows_long.append(win_data_long)
                        sample_meta.append({
                            'cell_id_global': cell,
                            'folder': folder_cell_map[cell]['folder'],
                            'original_cell_id': folder_cell_map[cell]['original_cell_id'],
                            'stim_frame': start_frame,
                            'direction': ev['direction'],
                            'repetition': ev['repetition']
                        })

        X_w1 = np.array(all_windows_w1)
        X_long = np.array(all_windows_long)
        print(f"  成功提取有效窗口样本: {len(X_w1)} 个")

        # ---------------- 基线校正 (包含空窗期截断逻辑) ----------------
        print(f"  应用基线校正: {self.cfg['BASELINE_METHOD']}")

        # === 第一步：计算常规减法所需要的 baselines ===
        if self.cfg['BASELINE_METHOD'] == 'global_min':
            active_duration_sec = self.cfg['TIME_INIT_BLANK'] + (
                        self.cfg['TIME_STIM_ON'] + self.cfg['TIME_STIM_OFF']) * (
                                              self.cfg['NUM_DIRECTIONS'] * self.cfg['REPETITIONS_PER_DIR'])
            max_active_frame = math.ceil(active_duration_sec / self.cfg['TIME_PER_FRAME'])

            global_mins = df_combined.iloc[:max_active_frame].min().to_dict()
            baselines = np.array([global_mins[meta['cell_id_global']] for meta in sample_meta]).reshape(-1, 1)

        elif self.cfg['BASELINE_METHOD'] == 'min':
            baselines = np.min(X_w1, axis=1, keepdims=True)

        elif self.cfg['BASELINE_METHOD'] == 't0':
            baselines = X_w1[:, 0:1]

        else:
            baselines = 0

        # === 第二步：统一应用校正公式 ===
        if self.cfg['BASELINE_METHOD'] == 'min_max':
            # W1 独立归一化
            mins_w1 = np.min(X_w1, axis=1, keepdims=True)
            maxs_w1 = np.max(X_w1, axis=1, keepdims=True)
            ranges_w1 = maxs_w1 - mins_w1
            ranges_w1[ranges_w1 == 0] = 1e-6
            X_w1_corr = (X_w1 - mins_w1) / ranges_w1

            # Long 独立归一化
            mins_long = np.min(X_long, axis=1, keepdims=True)
            maxs_long = np.max(X_long, axis=1, keepdims=True)
            ranges_long = maxs_long - mins_long
            ranges_long[ranges_long == 0] = 1e-6
            X_long_corr = (X_long - mins_long) / ranges_long

        elif self.cfg['BASELINE_METHOD'] == 'min_max_t0':
            # W1 独立归一化 + t0对齐
            mins_w1 = np.min(X_w1, axis=1, keepdims=True)
            maxs_w1 = np.max(X_w1, axis=1, keepdims=True)
            ranges_w1 = maxs_w1 - mins_w1
            ranges_w1[ranges_w1 == 0] = 1e-6
            X_norm_w1 = (X_w1 - mins_w1) / ranges_w1
            t0_w1 = X_norm_w1[:, 0:1]
            X_w1_corr = X_norm_w1 - t0_w1

            # Long 独立归一化 + t0对齐
            mins_long = np.min(X_long, axis=1, keepdims=True)
            maxs_long = np.max(X_long, axis=1, keepdims=True)
            ranges_long = maxs_long - mins_long
            ranges_long[ranges_long == 0] = 1e-6
            X_norm_long = (X_long - mins_long) / ranges_long
            t0_long = X_norm_long[:, 0:1]
            X_long_corr = X_norm_long - t0_long

        else:
            # 其他所有常规减法方法 (global_min, min, t0, none)：直接减去 baseline
            X_w1_corr = X_w1 - baselines
            X_long_corr = X_long - baselines

        # ---------------- 执行 K-Means ----------------
        print(f"  正在执行 KMeans (k={self.cfg['N_CLUSTERS']})...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_w1_corr)
        kmeans = KMeans(n_clusters=self.cfg['N_CLUSTERS'], random_state=self.cfg['RANDOM_STATE'], n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        results_df = pd.DataFrame(sample_meta)
        results_df['cluster'] = labels
        out_csv = os.path.join(self.dirs['csv'], 'global_clustering_results.csv')
        results_df.to_csv(out_csv, index=False)
        print(f"  -> 聚类标签已保存: {out_csv}")

        self._plot_cluster_curves(X_w1_corr, labels, 'w1', w_size)
        self._plot_cluster_curves(X_long_corr, labels, 'long', long_w_size)

        return results_df

    def _plot_cluster_curves(self, X, labels, view_type, length):
        """共用绘图函数"""
        n_clusters = self.cfg['N_CLUSTERS']
        colors = plt.cm.get_cmap('tab10', n_clusters)

        plt.figure(figsize=(10 if view_type == 'w1' else 14, 8))
        for i in range(n_clusters):
            cluster_data = X[labels == i]
            mean_trace = np.mean(cluster_data, axis=0)
            plt.plot(range(length), mean_trace,
                     label=f'Cluster {i} (n={len(cluster_data)})',
                     color=colors(i), linewidth=2)

        plt.axvline(x=0, color='red', linestyle='--', label='Stimulus Onset')
        stim_dur_frames = math.ceil(self.cfg['TIME_STIM_ON'] / self.cfg['TIME_PER_FRAME'])
        plt.axvspan(0, stim_dur_frames, color='red', alpha=0.1, label='Stimulus Duration')

        if view_type == 'long':
            for i in range(1, self.cfg['NUM_WINDOWS_TO_VIEW']):
                plt.axvline(x=self.cfg['WINDOW_SIZE'] * i, color='blue', linestyle=':',
                            label='Window Boundary' if i == 1 else None)

        plt.title(f'Cluster Average Response ({view_type.upper()} View)', fontsize=16)
        plt.xlabel('Frames Relative to Stimulus Onset', fontsize=12)
        plt.ylabel(f'Intensity Corrected ({self.cfg["BASELINE_METHOD"]})', fontsize=12)

        handles, lbs = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(lbs, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.grid(True, linestyle='--', alpha=0.6)
        out_png = os.path.join(self.dirs['plots'], f'cluster_avg_{view_type}.png')
        plt.savefig(out_png, dpi=150)
        plt.close()

    # -----------------------------------------------------------------
    # 模式 C: 并行图像裁剪与存储树生成
    # -----------------------------------------------------------------
    def run_cropping(self, cluster_df):
        print("\n--- [模式 C] 基于聚类结果进行多文件夹图像裁剪 ---")

        grouped = cluster_df.groupby('folder')

        for folder, folder_df in grouped:
            print(f"\n[*] 正在处理视野/文件夹: {folder} (需裁剪 {len(folder_df)} 个 Patch)")

            img_dir = os.path.join(self.cfg['IMG_ROOT_DIR'], folder)
            mask_path = os.path.join(self.cfg['MASK_ROOT_DIR'], folder, self.cfg['MASK_RELATIVE_PATH'])

            if not os.path.exists(mask_path):
                print(f"[!] 跳过 {folder}: 找不到掩码文件 {mask_path}")
                continue

            coords = self._get_coordinates(mask_path, folder)

            image_files = self._index_image_files(img_dir)
            if not image_files:
                print(f"[!] 跳过 {folder}: 未找到匹配的 JPG 图像")
                continue

            def crop_single_patch(row):
                orig_cell_id = str(row['original_cell_id'])
                if orig_cell_id not in coords: return

                cx, cy = coords[orig_cell_id]
                cluster = row['cluster']
                direction = row['direction']
                rep = row['repetition']
                start_frame = row['stim_frame']

                save_dir = os.path.join(self.dirs['patches'],
                                        f"Cluster_{cluster}",
                                        folder,
                                        f"Cell_{orig_cell_id}",
                                        f"Direction_{direction:03d}",
                                        f"Repetition_{rep:02d}")
                os.makedirs(save_dir, exist_ok=True)

                patch_r = self.cfg['PATCH_SIZE'] // 2
                crop_len = self.cfg['CROP_WINDOW_SIZE']

                for t in range(crop_len):
                    curr_frame = start_frame + t
                    if curr_frame not in image_files: continue

                    try:
                        img = io.imread(image_files[curr_frame])

                        if img.ndim == 3:
                            patch = np.zeros((self.cfg['PATCH_SIZE'], self.cfg['PATCH_SIZE'], img.shape[2]),
                                             dtype=img.dtype)
                        else:
                            patch = np.zeros((self.cfg['PATCH_SIZE'], self.cfg['PATCH_SIZE']), dtype=img.dtype)

                        src_y1, src_y2 = cy - patch_r, cy + patch_r
                        src_x1, src_x2 = cx - patch_r, cx + patch_r
                        dst_y1, dst_y2 = 0, self.cfg['PATCH_SIZE']
                        dst_x1, dst_x2 = 0, self.cfg['PATCH_SIZE']

                        h, w = img.shape[:2]
                        if src_y1 < 0: dst_y1 = -src_y1; src_y1 = 0
                        if src_x1 < 0: dst_x1 = -src_x1; src_x1 = 0
                        if src_y2 > h: dst_y2 -= (src_y2 - h); src_y2 = h
                        if src_x2 > w: dst_x2 -= (src_x2 - w); src_x2 = w

                        if dst_y2 > dst_y1 and dst_x2 > dst_x1:
                            if img.ndim == 3:
                                patch[dst_y1:dst_y2, dst_x1:dst_x2, :] = img[src_y1:src_y2, src_x1:src_x2, :]
                            else:
                                patch[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

                        fname = f"frame_{curr_frame:04d}.tif"
                        tifffile.imwrite(os.path.join(save_dir, fname), patch)
                    except Exception as e:
                        pass

            print(f"    启动多线程裁剪 ({self.cfg['MAX_WORKERS']} workers)...")
            with ThreadPoolExecutor(max_workers=self.cfg['MAX_WORKERS']) as executor:
                list(executor.map(crop_single_patch, [row for _, row in folder_df.iterrows()]))

            print(f"    {folder} 裁剪完成。")

    def _get_coordinates(self, mask_path, folder_name):
        """解析掩码坐标并可选生成核验图"""
        mask_data = nib.load(mask_path).get_fdata().astype(int)
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

        if self.cfg['DEBUG_ALIGNMENT']:
            plt.figure(figsize=(10, 10))
            plt.imshow(mask_2d > 0, cmap='gray')
            plt.scatter([p[0] for p in coords.values()], [p[1] for p in coords.values()], c='red', marker='x', s=10)
            plt.title(f"{folder_name} (Transpose={self.cfg['TRANSPOSE_COORDS']})")
            plt.savefig(os.path.join(self.dirs['debug'], f'alignment_{folder_name}.png'))
            plt.close()

        return coords

    def _index_image_files(self, img_dir):
        """建立帧索引字典，完美兼容 Image 1_t0001.jpg 格式"""
        image_files = {}
        pattern = re.compile(r'(\d+)')
        ext = self.cfg['FILE_EXTENSION'].lower()
        if not os.path.exists(img_dir): return image_files

        for f in os.listdir(img_dir):
            if f.lower().endswith(ext):
                match = pattern.search(f)
                if match:
                    # 提取文件名最后一段数字 (例如 t0001 提取出 1)
                    # 减 1 是因为图片通常是 1-based (t0001), 我们需要转为 0-based 供代码处理
                    frame_idx = int(re.findall(r'\d+', f)[-1]) - 1
                    image_files[frame_idx] = os.path.join(img_dir, f)
        return image_files

    # -----------------------------------------------------------------
    # 主路由
    # -----------------------------------------------------------------
    def execute(self):
        print("====== Cell Intensity Analysis Pipeline 启动 ======")
        mode = self.cfg['RUN_MODE']

        if mode == 'full_time_course':
            self.run_full_time_course()
        elif mode == 'cluster_only':
            self.run_clustering()
        elif mode == 'cluster_and_crop':
            cluster_results = self.run_clustering()
            self.run_cropping(cluster_results)
        else:
            print(f"未知的运行模式: {mode}")

        print("\n====== Pipeline 执行完毕 ======")


if __name__ == '__main__':
    pipeline = CellAnalyzerPipeline(CONFIG)
    pipeline.execute()
