import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.measure import regionprops
from skimage import io
import tifffile
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.interpolate import interp1d

# 设置非交互模式
plt.ioff()

# ================= 配置区域 =================

CONFIG = {
    # --- 1. 路径配置 ---
    'DATA_ROOT_DIR': r"pre_analyze/antagonism_radiomics_features",
    'RAW_IMAGES_ROOT_DIR': r"20251102_initial_code/data/data_20250817/antagonism",

    'FEATURE_FILES': {
        'Area': 'Final_Area.csv',
        'Mean': 'Final_Mean.csv',
    },

    'GLOBAL_MASK_NAME': 'Global_Static_Mask.nii.gz',
    'RAW_IMAGE_EXT': '.jpg',

    # --- 2. 输出路径 ---
    'OUTPUT_DIR': r"pre_analyze/results/jiekang/late_responders_event_clustering",

    # --- 3. 实验参数 ---
    'STIM_FRAMES': [31, 42, 52, 63, 73, 84, 95, 105, 116, 126, 137, 148, 158, 169, 179],
    'TOTAL_FRAMES': 190,

    # --- 4. 筛选逻辑参数 ---
    'RISE_THRESHOLD': 10.0,  # 亮度回升阈值

    # --- 5. 聚类参数 ---
    'N_CLUSTERS': 4,  # 最终决定使用的类别数
    'USE_PCA': True,
    'PCA_COMPONENTS': 0.95,
    'TARGET_LEN': 11,  # 插值目标长度

    # --- 6. 裁剪与绘图参数 ---
    'DO_CROPPING': True,
    'PATCH_SIZE': 32,
    'PLOT_HEIGHT': 160,  # 迷你曲线图高度
}


# ===========================================

class EventClusteringPipeline:
    def __init__(self, config):
        self.cfg = config
        # 定义三个主要输出文件夹
        self.dirs = {
            'montages': os.path.join(config['OUTPUT_DIR'], '02_clustered_patches'),
            'plots': os.path.join(config['OUTPUT_DIR'], '01_cluster_trends'),  # 聚类曲线和K值评估都在这里
            'report': os.path.join(config['OUTPUT_DIR'], '00_reports')
        }
        for d in self.dirs.values(): os.makedirs(d, exist_ok=True)

    # ---------------------------------------------------------
    # 步骤 1: 加载数据
    # ---------------------------------------------------------
    def step1_load_data(self):
        print("\n--- STEP 1: 加载特征数据 ---")
        root = self.cfg['DATA_ROOT_DIR']
        subfolders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]

        mean_dfs = []
        area_dfs = []

        for folder in subfolders:
            folder_path = os.path.join(root, folder, 'final_results_zero')
            path_mean = os.path.join(folder_path, self.cfg['FEATURE_FILES']['Mean'])
            path_area = os.path.join(folder_path, self.cfg['FEATURE_FILES']['Area'])

            if not os.path.exists(path_mean): continue

            df_m = pd.read_csv(path_mean, index_col=0)
            df_m.columns = [f"{folder}___{col}" for col in df_m.columns]
            mean_dfs.append(df_m)

            if os.path.exists(path_area):
                df_a = pd.read_csv(path_area, index_col=0)
                df_a.columns = [f"{folder}___{col}" for col in df_a.columns]
                area_dfs.append(df_a)

        if not mean_dfs:
            raise ValueError("未找到特征文件！")

        big_mean_df = pd.concat(mean_dfs, axis=1).fillna(0)
        return big_mean_df

    # ---------------------------------------------------------
    # 步骤 2: 筛选事件
    # ---------------------------------------------------------
    def step2_filter_events(self, mean_df):
        print("\n--- STEP 2: 筛选迟发反应事件 (形态学) ---")

        stim_frames_0based = [x - 1 for x in self.cfg['STIM_FRAMES']]
        total_frames = self.cfg['TOTAL_FRAMES']
        rise_thresh = self.cfg['RISE_THRESHOLD']

        tasks = []

        for global_cell_id in mean_df.columns:
            trace = mean_df[global_cell_id].values
            folder, original_cell_id = global_cell_id.split('___')

            for i in range(len(stim_frames_0based)):
                start_f = stim_frames_0based[i]

                if i < len(stim_frames_0based) - 1:
                    end_f = stim_frames_0based[i + 1]
                else:
                    end_f = total_frames

                window_trace = trace[start_f: end_f]
                curr_win_len = len(window_trace)

                if curr_win_len < 3: continue

                found_valley = False
                for k in range(1, curr_win_len - 1):
                    val = window_trace[k]
                    prev = window_trace[k - 1]
                    next_val = window_trace[k + 1]

                    if val < prev and val < next_val:
                        if (window_trace[-1] - val) >= rise_thresh:
                            found_valley = True
                            break

                if found_valley:
                    tasks.append({
                        'Global_ID': global_cell_id,
                        'Folder': folder,
                        'Cell_ID': original_cell_id,
                        'Start_Frame': start_f,
                        'Window_Len': curr_win_len,
                        'Raw_Trace': window_trace
                    })

        tasks_df = pd.DataFrame(tasks)
        print(f"  筛选完成: 共发现 {len(tasks_df)} 个事件。")
        return tasks_df

    # ---------------------------------------------------------
    # [辅助] 评估最优 K 值 (保存到 01_cluster_trends)
    # ---------------------------------------------------------
    def evaluate_optimal_k(self, X, max_k=10):
        print(f"\n  正在评估最优 K 值 (范围 2-{max_k})...")
        inertias = []
        sil_scores = []
        K_range = range(2, max_k + 1)

        if X.shape[0] < max_k:
            print(f"  [提示] 样本数 ({X.shape[0]}) 过少，跳过 K 值评估。")
            return

        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            score = silhouette_score(X, labels)
            sil_scores.append(score)
            print(f"    k={k}: Inertia={km.inertia_:.1f}, Silhouette={score:.4f}")

        # 绘图
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (Elbow)', color=color)
        ax1.plot(K_range, inertias, 'o-', color=color, label='Inertia')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Silhouette Score', color=color)
        ax2.plot(K_range, sil_scores, 's--', color=color, label='Silhouette')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Optimal k Evaluation')
        plt.tight_layout()

        # [确认] 保存到 plots 文件夹，即 01_cluster_trends
        save_path = os.path.join(self.dirs['plots'], 'optimal_k_evaluation.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  -> 评估结果已保存: {save_path}")

    # ---------------------------------------------------------
    # 步骤 3: 事件向量构建与聚类
    # ---------------------------------------------------------
    def step3_cluster_events(self, tasks_df):
        if tasks_df.empty: return tasks_df

        print("\n--- STEP 3: 事件聚类 (Resample -> Normalize -> Cluster) ---")

        target_len = self.cfg['TARGET_LEN']
        X_list = []

        # 1. 构建特征矩阵 (插值对齐)
        for _, row in tasks_df.iterrows():
            raw_trace = row['Raw_Trace']
            curr_len = len(raw_trace)

            if curr_len == target_len:
                new_trace = raw_trace
            else:
                x_old = np.linspace(0, 1, curr_len)
                f = interp1d(x_old, raw_trace, kind='linear')
                x_new = np.linspace(0, 1, target_len)
                new_trace = f(x_new)

            X_list.append(new_trace)

        X = np.array(X_list)

        # 2. 数据预处理
        X_processed = X - X.min(axis=1, keepdims=True)
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_processed.T).T

        print(f"  聚类输入矩阵形状: {X_processed.shape}")

        # 3. PCA
        if self.cfg['USE_PCA']:
            pca = PCA(n_components=self.cfg['PCA_COMPONENTS'])
            X_input = pca.fit_transform(X_processed)
        else:
            X_input = X_processed

        # 4. 评估 K 值
        self.evaluate_optimal_k(X_input, max_k=8)

        # 5. 正式聚类
        print(f"  执行最终聚类 (k={self.cfg['N_CLUSTERS']})...")
        kmeans = KMeans(n_clusters=self.cfg['N_CLUSTERS'], random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_input)

        tasks_df['Cluster'] = labels

        # 6. 绘制聚类趋势图 (保存到 01_cluster_trends)
        self._plot_cluster_trends(X_processed, labels)

        tasks_df.drop(columns=['Raw_Trace']).to_csv(
            os.path.join(self.dirs['report'], 'clustered_events.csv'), index=False
        )

        return tasks_df

    def _plot_cluster_trends(self, X, labels):
        n_clusters = self.cfg['N_CLUSTERS']
        x_axis = range(self.cfg['TARGET_LEN'])

        plt.figure(figsize=(12, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

        for c in range(n_clusters):
            cluster_data = X[labels == c]
            if len(cluster_data) == 0: continue

            mean_trace = cluster_data.mean(axis=0)
            std_trace = cluster_data.std(axis=0)

            plt.plot(x_axis, mean_trace, label=f'Cluster {c} (n={len(cluster_data)})',
                     color=colors[c], linewidth=2)
            plt.fill_between(x_axis, mean_trace - std_trace, mean_trace + std_trace,
                             color=colors[c], alpha=0.1)

        plt.title('Average Event Trends (Standardized)')
        plt.legend()
        plt.tight_layout()

        # [确认] 保存到 plots 文件夹
        save_path = os.path.join(self.dirs['plots'], 'cluster_trends.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  -> 聚类趋势图已保存: {save_path}")

    # ---------------------------------------------------------
    # 辅助函数: 绘制 Sparkline
    # ---------------------------------------------------------
    def _draw_sparkline(self, data, width, height):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        if len(data) < 2: return img

        padding = 10
        d_min, d_max = np.min(data), np.max(data)

        if d_max == d_min:
            scale = 0
        else:
            scale = (height - 2 * padding) / (d_max - d_min)

        def to_pt(idx, val):
            patch_w = width / len(data)
            x = int(idx * patch_w + patch_w / 2)
            y = int(height - padding - (val - d_min) * scale)
            return (x, y)

        y_ref = int(height - padding - (data[0] - d_min) * scale)
        cv2.line(img, (0, y_ref), (width, y_ref), (100, 100, 100), 1)

        pts = [to_pt(i, v) for i, v in enumerate(data)]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (255, 255, 0), 2, cv2.LINE_AA)

        return img

    # ---------------------------------------------------------
    # 步骤 4: 分类保存拼图
    # ---------------------------------------------------------
    def step4_save_clustered_montages(self, tasks_df):
        if tasks_df.empty or not self.cfg['DO_CROPPING']: return
        print("\n--- STEP 4: 生成分类拼图 (Clustered Montages) ---")

        patch_size = self.cfg['PATCH_SIZE']
        plot_h = self.cfg['PLOT_HEIGHT']
        patch_r = patch_size // 2

        for c in range(self.cfg['N_CLUSTERS']):
            os.makedirs(os.path.join(self.dirs['montages'], f'Cluster_{c}'), exist_ok=True)

        for folder_name, group in tasks_df.groupby('Folder'):
            mask_path = os.path.join(self.cfg['DATA_ROOT_DIR'], folder_name, 'final_results_zero',
                                     self.cfg['GLOBAL_MASK_NAME'])
            raw_img_folder = os.path.join(self.cfg['RAW_IMAGES_ROOT_DIR'], folder_name)

            if not os.path.exists(mask_path) or not os.path.exists(raw_img_folder): continue

            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata().astype(int)
            if mask_data.ndim == 3: mask_data = np.max(mask_data, axis=-1)
            props = regionprops(mask_data)
            prop_map = {str(p.label): p.centroid for p in props}

            raw_files = [f for f in os.listdir(raw_img_folder) if f.endswith(self.cfg['RAW_IMAGE_EXT'])]

            def get_t_num(fname):
                match = re.search(r'_t(\d+)', fname)
                if match:
                    return int(match.group(1))
                else:
                    nums = re.findall(r'\d+', fname)
                    return int(nums[-1]) if nums else 0

            raw_files.sort(key=get_t_num)
            file_map = {i: f for i, f in enumerate(raw_files)}

            frame_requirements = {}
            montage_buffers = {}

            for _, row in group.iterrows():
                cid = str(row['Cell_ID'])
                start_f = row['Start_Frame']
                win_len = row['Window_Len']
                cluster_id = row['Cluster']

                if cid not in prop_map: continue

                key = (cid, start_f)
                img_strip = np.zeros((patch_size, patch_size * win_len, 3), dtype=np.uint8)
                raw_trace = row['Raw_Trace']

                montage_buffers[key] = {
                    'img': img_strip,
                    'curve_data': raw_trace,
                    'win_len': win_len,
                    'cluster': cluster_id
                }

                for t in range(win_len):
                    curr_f = start_f + t
                    if curr_f not in frame_requirements: frame_requirements[curr_f] = []
                    frame_requirements[curr_f].append({'key': key, 'offset': t, 'cid': cid})

            if not frame_requirements: continue

            needed_frames = sorted(frame_requirements.keys())

            for f_idx in needed_frames:
                if f_idx not in file_map: continue
                try:
                    img = io.imread(os.path.join(raw_img_folder, file_map[f_idx]))
                except:
                    continue

                if img.dtype != np.uint8:
                    img = img.astype(float)
                    p1, p99 = np.percentile(img, (1, 99))
                    img = (img - p1) / (p99 - p1 + 1e-6)
                    img = np.clip(img, 0, 1) * 255
                    img = img.astype(np.uint8)

                if img.ndim == 2:
                    img_rgb = np.stack([img] * 3, axis=-1)
                elif img.ndim == 3 and img.shape[2] == 4:
                    img_rgb = img[..., :3]
                else:
                    img_rgb = img

                h, w = img_rgb.shape[:2]

                for req in frame_requirements[f_idx]:
                    cid = req['cid']
                    cy, cx = prop_map[cid]
                    sy, ey = int(cy) - patch_r, int(cy) + patch_r
                    sx, ex = int(cx) - patch_r, int(cx) + patch_r

                    patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    src_sy, src_ey = max(0, sy), min(h, ey)
                    src_sx, src_ex = max(0, sx), min(w, ex)
                    dst_sy, dst_ey = src_sy - sy, src_sy - sy + (src_ey - src_sy)
                    dst_sx, dst_ex = src_sx - sx, src_sx - sx + (src_ex - src_sx)

                    if dst_ey > dst_sy and dst_ex > dst_sx:
                        patch[dst_sy:dst_ey, dst_sx:dst_ex] = img_rgb[src_sy:src_ey, src_sx:src_ex]

                    key = req['key']
                    off = req['offset']
                    x_start = off * patch_size
                    montage_buffers[key]['img'][:, x_start: x_start + patch_size, :] = patch

            for (cid, start_f), item in montage_buffers.items():
                img_strip = item['img']
                curve_data = item['curve_data']
                win_len = item['win_len']
                cluster_id = item['cluster']

                total_w = patch_size * win_len
                curve_img = self._draw_sparkline(curve_data, total_w, plot_h)
                final_output = np.vstack([img_strip, curve_img])

                actual_filename = file_map.get(start_f, f"Unknown_Idx{start_f}")
                t_num = get_t_num(actual_filename)

                try:
                    stim_idx = self.cfg['STIM_FRAMES'].index(start_f + 1) + 1
                except ValueError:
                    stim_idx = 0

                fname = f"{folder_name}_Cell{cid}_Stim{stim_idx}_t{t_num:03d}_Cluster{cluster_id}.tif"
                save_path = os.path.join(self.dirs['montages'], f'Cluster_{cluster_id}', fname)
                io.imsave(save_path, final_output, check_contrast=False)

            print(f"    -> 文件夹 {folder_name} 保存完毕。")

    def run(self):
        mean_df = self.step1_load_data()
        tasks_df = self.step2_filter_events(mean_df)
        tasks_df = self.step3_cluster_events(tasks_df)
        self.step4_save_clustered_montages(tasks_df)
        print("\n--- 全部处理完成 ---")


if __name__ == "__main__":
    pipeline = EventClusteringPipeline(CONFIG)
    pipeline.run()