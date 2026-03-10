import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import nibabel as nib
from skimage.measure import regionprops
from skimage import io
import tifffile
from sklearn.metrics import silhouette_score, calinski_harabasz_score
"""
本脚本实现了对多视野、多特征的全时程细胞响应数据进行聚类分析，并生成对应的拼图结果。用于对整个激动/拮抗类型的细胞聚类分析（多文件夹）
主要步骤包括：
1. 数据加载与合并：从多个子文件夹中读取不同特征的细胞响应数据，并进行对齐合并。
2. 聚类分析：基于用户选择的特征模式（仅亮度或多特征拼接），进行K-Means聚类，支持PCA降维。
3. 逆向解析与裁剪：根据聚类结果，生成每个细胞的全时程拼图，突出显示刺激帧。
用户可通过配置区域调整路径、聚类参数及裁剪选项。
"""


# 设置非交互模式
plt.ioff()

# ================= 配置区域 =================

CONFIG = {
    # --- 1. 路径配置 ---
    # 特征文件的根目录 (里面应包含 Image 1, Image 2 等子文件夹)
    'DATA_ROOT_DIR': r"pre_analyze/antagonism_radiomics_features",

    # 原始图像的根目录 (里面也应包含同名的 Image 1, Image 2 等子文件夹)
    'RAW_IMAGES_ROOT_DIR': r"20251102_initial_code/data/data_20250817/antagonism",

    # 特征文件名
    'FEATURE_FILES': {
        'Mean': 'Final_Mean.csv',
        # 以下仅在 MODE='multi_feature' 时需要
        'Area': 'Final_Area.csv',
        'Perimeter': 'Final_Perimeter.csv',
        'Contrast': 'Final_Contrast.csv',
        'Entropy': 'Final_Entropy.csv',
    },

    # 全局掩码的文件名 (假设它在每个子文件夹里都叫这个名字)
    'GLOBAL_MASK_NAME': 'Global_Static_Mask.nii.gz',
    'RAW_IMAGE_EXT': '.jpg',

    # --- 2. 输出路径 ---
    'OUTPUT_DIR': r"pre_analyze/results/jiekang/all_frames_clustering_with_multifeatures_patch_concat_Ktest",

    # --- 3. 聚类策略 ---
    # 'intensity': 仅使用亮度 (190维)
    # 'multi_feature': 使用所有特征拼接 (190 * 5 = 950维)
    'MODE': 'multi_feature',

    'N_CLUSTERS': 4,
    'USE_PCA': True,  # 是否启用 PCA 降维 (强烈建议 True)
    'PCA_COMPONENTS': 0.95,  # 保留 95% 的解释方差

    # 是否对亮度进行标准化？
    # True: 关注曲线形状 (忽略绝对强度差异)
    # False: 关注绝对强度 (强反应和弱反应会被分开)
    'NORMALIZE_INTENSITY': False,

    # --- 4. 实验参数 (用于绘图) ---
    # 所有的刺激帧 (1-based)
    'STIM_FRAMES': [31, 42, 52, 63, 73, 84, 95, 105, 116, 126, 137, 148, 158, 169, 179],
    'TOTAL_FRAMES': 190,

    # --- 5. 裁剪参数 ---
    'DO_CROPPING': True,  # 是否执行裁剪
    'CROP_WINDOW_SIZE': 32,
    'PATCH_SIZE': 32,
    'TRANSPOSE_COORDS': False, # 是否交换坐标系 (根据 Mask 的生成方式决定，部分nii.gz图像保存时xy轴被交换，这里没有交换)
}


# ===========================================

class WholeSeriesPipeline:
    def __init__(self, config):
        self.cfg = config
        self.dirs = {
            'results': os.path.join(config['OUTPUT_DIR'], '01_clustering_results'),
            'plots': os.path.join(config['OUTPUT_DIR'], '02_plots'),
            'patches': os.path.join(config['OUTPUT_DIR'], '03_patches')
        }
        for d in self.dirs.values(): os.makedirs(d, exist_ok=True)

    # ---------------------------------------------------------
    # 步骤 1: 数据加载与合并
    # ---------------------------------------------------------
    def step1_load_and_merge(self):
        print("\n--- STEP 1: 加载与合并多视野数据 ---")

        root = self.cfg['DATA_ROOT_DIR']
        subfolders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
        print(f"  发现 {len(subfolders)} 个子文件夹: {subfolders}")

        # 存储所有数据: { 'Mean': BigDataFrame, 'Area': BigDataFrame ... }
        merged_data = {k: [] for k in self.cfg['FEATURE_FILES'].keys()}

        # 遍历每个文件夹
        for folder in subfolders:
            folder_path = os.path.join(root, folder, 'final_results_zero')

            # 读取该文件夹下的所有特征
            for feat_name, filename in self.cfg['FEATURE_FILES'].items():
                file_path = os.path.join(folder_path, filename)
                if not os.path.exists(file_path):
                    print(f"    [警告] {folder} 中缺少 {filename}, 跳过此特征。")
                    continue

                df = pd.read_csv(file_path, index_col=0)

                # *** 核心：重命名列名，加上文件夹前缀 ***
                # 原列名 "1" -> 新列名 "Image 1_1"
                df.columns = [f"{folder}___{col}" for col in df.columns]

                merged_data[feat_name].append(df)

        # 合并 DataFrame
        final_dfs = {}
        common_columns = None

        for feat_name, dfs in merged_data.items():
            if not dfs: continue
            # 横向拼接 (axis=1)
            big_df = pd.concat(dfs, axis=1)

            # 简单清洗：填补 NaN (理论上 clean 脚本已处理，这里防万一)
            big_df = big_df.fillna(0)

            final_dfs[feat_name] = big_df
            print(f"  [{feat_name}] 合并完成。形状: {big_df.shape} (帧数 x 总细胞数)")

            if common_columns is None:
                common_columns = big_df.columns
            else:
                # 确保不同特征的细胞对齐 (取交集)
                common_columns = common_columns.intersection(big_df.columns)

        # 过滤掉未对齐的细胞
        for k in final_dfs:
            final_dfs[k] = final_dfs[k][common_columns]

        print(f"  最终对齐细胞数: {len(common_columns)}")
        return final_dfs, common_columns

    def evaluate_optimal_k(self, X, max_k=10):
        """
        计算并绘制 Elbow (Inertia) 和 Silhouette Score 曲线，帮助寻找最优 K 值。
        """
        print(f"\n  正在评估最优聚类数 (Range: 2 to {max_k})...")

        inertias = []
        silhouette_scores = []
        # ch_scores = [] # 可选：Calinski-Harabasz Score
        K_range = range(2, max_k + 1)

        for k in K_range:
            # 注意：为了速度，这里 n_init 可以设小一点，或者保持默认
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            # 1. Inertia (手肘法)
            inertias.append(kmeans.inertia_)

            # 2. Silhouette Score (轮廓系数)
            # 样本量非常大时，计算轮廓系数很慢，可以采样计算 (sample_size=10000)
            # 如果细胞数不多 (<5000)，直接算即可
            if X.shape[0] > 10000:
                score = silhouette_score(X, labels, sample_size=10000, random_state=42)
            else:
                score = silhouette_score(X, labels)
            silhouette_scores.append(score)

            # ch_scores.append(calinski_harabasz_score(X, labels))
            print(f"    k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={score:.4f}")

        # --- 绘图 ---
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:red'
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (Elbow Method)', color=color)
        ax1.plot(K_range, inertias, 'o-', color=color, label='Inertia')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        # 双坐标轴：右侧画轮廓系数
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Silhouette Score (Higher is better)', color=color)
        ax2.plot(K_range, silhouette_scores, 's--', color=color, label='Silhouette')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Optimal k Evaluation: Elbow Method & Silhouette Score')
        plt.tight_layout()

        save_path = os.path.join(self.dirs['plots'], 'optimal_k_evaluation.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  -> 评估曲线已保存: {save_path}")
        print("  请查看图片，根据拐点(Elbow)和最高轮廓系数选择合适的 N_CLUSTERS。")


    # ---------------------------------------------------------
    # 步骤 2: 聚类 (支持 PCA)
    # ---------------------------------------------------------
    def step2_clustering(self, data_dfs, cell_ids):
        print(f"\n--- STEP 2: 全时程聚类 (Mode: {self.cfg['MODE']}) ---")

        # 1. 构建特征矩阵 X (N_samples x N_features)
        #    我们需要转置：行是细胞，列是时间/特征

        if self.cfg['MODE'] == 'intensity':
            # 仅使用亮度
            df = data_dfs['Mean'].T  # 转置后: (Cells, Frames)
            X = df.values

            # 基线校正 (每个细胞减去自己的最小值)
            mins = X.min(axis=1, keepdims=True)
            X = X - mins

            # 可选标准化
            if self.cfg['NORMALIZE_INTENSITY']:
                scaler = StandardScaler()
                X = scaler.fit_transform(X.T).T  # 对每个细胞的时间序列做标准化

        else:  # multi_feature
            # 多特征拼接
            feature_blocks = []
            for feat_name in self.cfg['FEATURE_FILES']:
                df = data_dfs[feat_name].T  # (Cells, Frames)
                vals = df.values

                # 独立标准化每个特征矩阵 (关键！)
                # 我们要消除 Area(1000) 和 Contrast(0.1) 的量级差异
                scaler = StandardScaler()
                vals_scaled = scaler.fit_transform(vals)  # 对每一列(时间点)做标准化?
                # 不，应该是对整个矩阵做？或者对每个特征做？
                # 最稳健的做法：把整个 (N, 190) 矩阵拉平看作一种特征，整体标准化
                vals_scaled = scaler.fit_transform(vals)

                feature_blocks.append(vals_scaled)

            # 拼接: 变成 (Cells, 190*5)
            X = np.hstack(feature_blocks)

        print(f"  聚类输入矩阵形状: {X.shape}")

        # 2. PCA 降维 (可选)
        if self.cfg['USE_PCA']:
            print(f"  执行 PCA 降维 (目标方差: {self.cfg['PCA_COMPONENTS']})...")
            pca = PCA(n_components=self.cfg['PCA_COMPONENTS'])
            X_input = pca.fit_transform(X)
            print(f"  -> 降维后形状: {X_input.shape} (保留了 {pca.n_components_} 个主成分)")
        else:
            X_input = X

        # ================== 新增代码开始 ==================
        # 在这里插入评估函数，仅运行一次查看结果，确定 k 后可注释掉
        # 建议 max_k 设置为 8 或 10，生物学实验通常不会有太多类型的反应
        self.evaluate_optimal_k(X_input, max_k=10)

        # 如果你想让程序在这里暂停，等待你看完图后再输入 k，可以使用 input()
        new_k = input(f"当前配置 k={self.cfg['N_CLUSTERS']}。输入新 k 值继续，或直接回车保持默认: ")
        if new_k.strip():
            self.cfg['N_CLUSTERS'] = int(new_k)
        # ================== 新增代码结束 ==================

        # 3. K-Means
        print(f"  执行 K-Means (k={self.cfg['N_CLUSTERS']})...")
        kmeans = KMeans(n_clusters=self.cfg['N_CLUSTERS'], random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_input)

        # 保存结果
        results = pd.DataFrame({'Global_ID': cell_ids, 'Cluster': labels})
        # 解析 Global ID 回 Folder 和 Cell ID
        # 假设 ID 格式: "FolderName___CellID"
        results[['Folder', 'Original_Cell_ID']] = results['Global_ID'].str.split('___', expand=True)

        out_path = os.path.join(self.dirs['results'], 'whole_series_clusters.csv')
        results.to_csv(out_path, index=False)
        print(f"  -> 聚类结果已保存: {out_path}")

        # 4. 绘图
        print(f"  正在绘制 {len(data_dfs)} 个特征的平均曲线...")

        for feat_name, df in data_dfs.items():
            # 获取该特征的原始数据 (Cells x Frames)
            feature_data = df.T.values

            # 仅对 'Mean' 做基线校正 (减最小值)，其他特征保持原始值以便观察物理意义
            if feat_name == 'Mean':
                feature_data = feature_data - feature_data.min(axis=1, keepdims=True)

            # 调用绘图，传入特征名
            self._plot_clusters(feature_data, labels, feat_name)

        return results

    def _plot_clusters(self, X, labels, feature_name):
        print("  正在绘制全时程平均曲线...")
        n_clusters = self.cfg['N_CLUSTERS']
        colors = plt.cm.get_cmap('tab10', n_clusters)
        total_frames = self.cfg['TOTAL_FRAMES']

        # 设置宽幅画布 (20英寸宽)
        plt.figure(figsize=(20, 6))

        for c in range(n_clusters):
            cluster_data = X[labels == c]
            mean_trace = np.mean(cluster_data, axis=0)
            plt.plot(range(total_frames), mean_trace,
                     label=f'Cluster {c} (n={len(cluster_data)})',
                     color=colors(c), linewidth=2)

        # 标记所有刺激
        for i, sti in enumerate(self.cfg['STIM_FRAMES']):
            # 1-based 转 0-based
            x = sti - 1
            plt.axvline(x=x, color='red', linestyle=':', alpha=0.3)

        plt.title(f"Whole Time-Series: {feature_name}")
        plt.xlabel("Frames")
        plt.ylabel(f'{feature_name} Value')
        plt.legend()
        plt.xlim(0, total_frames)
        plt.grid(True, alpha=0.3)
        save_name = f'trend_{feature_name}.png'
        plt.savefig(os.path.join(self.dirs['plots'], save_name))
        plt.close()
        print(f"  -> {feature_name}曲线图已保存。")

    # ---------------------------------------------------------
    # 步骤 3: 逆向解析与裁剪 (Dynamic Cropping)
    # ---------------------------------------------------------
    def step3_dynamic_crop(self, results_df):
        if not self.cfg['DO_CROPPING']: return
        print("\n--- STEP 3: 生成全时程拼图 (Montage Generation) ---")

        # 参数准备
        patch_size = self.cfg['PATCH_SIZE']
        patch_r = patch_size // 2
        total_frames = self.cfg['TOTAL_FRAMES']
        stim_frames_set = set(self.cfg['STIM_FRAMES'])  # 转集合，加速查找

        # 布局配置: 190帧 -> 10行 x 19列
        GRID_ROWS = 10
        GRID_COLS = 19

        # 准备输出目录
        for c in range(self.cfg['N_CLUSTERS']):
            cluster_dir = os.path.join(self.dirs['patches'], f'Cluster_{c}')
            os.makedirs(cluster_dir, exist_ok=True)

        # 核心逻辑：按文件夹遍历
        # 既然 results_df 是全时程聚类结果，每个细胞在表中只会出现一次
        for folder_name, group in results_df.groupby('Folder'):
            print(f"    正在处理文件夹: {folder_name} (包含 {len(group)} 个细胞)...")

            # --- 1. 路径准备 ---
            mask_path = os.path.join(self.cfg['DATA_ROOT_DIR'], folder_name, 'final_results_zero',
                                     self.cfg['GLOBAL_MASK_NAME'])
            raw_img_folder = os.path.join(self.cfg['RAW_IMAGES_ROOT_DIR'], folder_name)

            if not os.path.exists(mask_path) or not os.path.exists(raw_img_folder):
                print(f"      [警告] 缺少 Mask 或图像文件夹，跳过: {folder_name}")
                continue

            # --- 2. 加载 Mask 并建立坐标索引 ---
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata().astype(int)
            # 处理可能存在的 3D Mask (如 (X,Y,1))
            if mask_data.ndim == 3:
                mask_data = np.max(mask_data, axis=-1)

            props = regionprops(mask_data)
            # 建立映射: Cell_ID (str) -> Centroid (row, col)
            # 注意：regionprops 返回的是 (row/y, col/x)
            prop_map = {str(p.label): p.centroid for p in props}

            # --- 3. 初始化细胞画布 ---
            # target_cells 结构: { 'Original_Cell_ID': {'cluster': int, 'cy': int, 'cx': int, 'canvas': ndarray} }
            target_cells = {}

            for _, row in group.iterrows():
                cid = str(row['Original_Cell_ID'])

                # 校验：CSV里的细胞ID必须在Mask里存在
                if cid in prop_map:
                    r, c = prop_map[cid]  # r=y, c=x

                    # 初始化全黑画布
                    canvas_h = GRID_ROWS * patch_size
                    canvas_w = GRID_COLS * patch_size
                    montage = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

                    target_cells[cid] = {
                        'cluster': row['Cluster'],
                        'cy': int(r),  # Row / Y
                        'cx': int(c),  # Col / X
                        'canvas': montage
                    }

            if not target_cells:
                print(f"      [提示] 该文件夹下没有匹配的细胞，跳过。")
                continue

            # --- 4. 遍历所有时间帧 (0 ~ 189) 并拼图 ---
            # 建立文件名映射 (Frame_Index -> File_Path)
            img_files_map = {}
            raw_files = sorted([f for f in os.listdir(raw_img_folder) if f.endswith(self.cfg['RAW_IMAGE_EXT'])])

            # 简单假设：按文件名排序后，第0个文件对应第1帧(index 0)
            # 如果文件名数字不连续，需要用正则精确匹配，这里沿用之前的逻辑
            pattern = re.compile(r'(\d+)')
            for f in raw_files:
                match = pattern.search(f)
                if match:
                    # 获取文件名里的数字
                    f_num = int(match.group(1))
                    # 这里的映射取决于文件名是从1开始还是0开始
                    # 假设文件名如 "1.jpg", "2.jpg"，则 file_idx = f_num - 1
                    # 假设文件名如 "000.jpg"，则 file_idx = f_num
                    # 为了最稳健，我们直接按 sorted 列表的顺序索引 (假设没有缺帧)
                    pass

                    # 采用按列表顺序索引更安全 (对应 total_frames)
            for frame_idx in range(total_frames):
                if frame_idx >= len(raw_files):
                    break

                f_name = raw_files[frame_idx]
                img_path = os.path.join(raw_img_folder, f_name)

                # 4.1 读取大图 (一次IO)
                try:
                    img = io.imread(img_path)
                except Exception as e:
                    print(f"Read Error: {e}")
                    continue

                # 4.2 图像标准化与转 RGB
                if img.dtype != np.uint8:
                    img = img.astype(float)
                    # 动态归一化 (防止全黑或全白)
                    vmin, vmax = np.percentile(img, 1), np.percentile(img, 99)
                    img = (img - vmin) / (vmax - vmin + 1e-6)
                    img = np.clip(img, 0, 1) * 255
                    img = img.astype(np.uint8)

                if img.ndim == 2:
                    img_rgb = np.stack([img, img, img], axis=-1)
                elif img.ndim == 3 and img.shape[2] == 4:
                    img_rgb = img[..., :3]
                else:
                    img_rgb = img

                img_h, img_w = img_rgb.shape[:2]

                # 4.3 计算在 Montage 中的位置
                m_row = frame_idx // GRID_COLS  # 行号 (0-9)
                m_col = frame_idx % GRID_COLS  # 列号 (0-18)

                dst_y = m_row * patch_size
                dst_x = m_col * patch_size

                # 4.4 检查是否需要红框 (Stim Frames 是 1-based)
                is_stim = (frame_idx + 1) in stim_frames_set
                border_color = [255, 0, 0]  # Red

                # 4.5 对该 Image 下的所有目标细胞进行裁剪
                for cid, info in target_cells.items():
                    cy, cx = info['cy'], info['cx']  # Centroids

                    # 源图裁剪坐标
                    sy, ey = cy - patch_r, cy + patch_r
                    sx, ex = cx - patch_r, cx + patch_r

                    # 目标 Patch 初始化
                    patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

                    # 计算重叠区域 (Padding Logic)
                    src_sy = max(0, sy);
                    src_ey = min(img_h, ey)
                    src_sx = max(0, sx);
                    src_ex = min(img_w, ex)

                    # 对应的 patch 内部坐标
                    p_sy = src_sy - sy
                    p_ey = p_sy + (src_ey - src_sy)
                    p_sx = src_sx - sx
                    p_ex = p_sx + (src_ex - src_sx)

                    if p_ey > p_sy and p_ex > p_sx:
                        patch[p_sy:p_ey, p_sx:p_ex] = img_rgb[src_sy:src_ey, src_sx:src_ex]

                    # 4.6 画红框
                    if is_stim:
                        bw = 2  # 边框宽度
                        patch[:bw, :, :] = border_color
                        patch[-bw:, :, :] = border_color
                        patch[:, :bw, :] = border_color
                        patch[:, -bw:, :] = border_color

                    # 4.7 填入画布
                    info['canvas'][dst_y: dst_y + patch_size,
                    dst_x: dst_x + patch_size] = patch

            # --- 5. 保存结果 ---
            print(f"      保存 {len(target_cells)} 张拼图...")
            for cid, info in target_cells.items():
                cluster_id = info['cluster']
                montage_img = info['canvas']

                # 文件名: Folder_CellID_Cluster.tif
                # 这样即使不同文件夹有相同ID的细胞，文件名也不同
                fname = f"{folder_name}_Cell{cid}_Cluster{cluster_id}_Montage.tif"
                save_path = os.path.join(self.dirs['patches'], f'Cluster_{cluster_id}', fname)

                io.imsave(save_path, montage_img, check_contrast=False)

        print("  所有拼图生成完成。")

    def run(self):
        # 1. 加载
        data_dfs, cell_ids = self.step1_load_and_merge()

        # 2. 聚类
        results = self.step2_clustering(data_dfs, cell_ids)

        # 3. 裁剪
        self.step3_dynamic_crop(results)


if __name__ == "__main__":
    pipeline = WholeSeriesPipeline(CONFIG)
    pipeline.run()