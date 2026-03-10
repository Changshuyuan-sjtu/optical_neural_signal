import os
import re
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.measure import label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries
from skimage.morphology import binary_erosion, disk
from skimage.color import label2rgb
import matplotlib.pyplot as plt


"""
当前脚本用于实验对不同帧相同细胞的追踪，即对于不同帧的同一个细胞，用同一个全局ID进行标记。
主要思路：
1. 生成全局静态掩码：
   - 对所有帧的掩码进行距离变换累加，得到一个全局距离场。
   - 在该距离场上寻找局部峰值，作为细胞中心的种子点。
   - 使用分水岭算法基于这些种子点生成全局静态掩码。
2. 逐帧重标记：
   - 对每一帧的掩码，识别出各个连通区域（细胞）。
   - 对每个连通区域，查看其在全局静态掩码中的对应位置，采用投票法确定该区域的全局ID。
3. 可视化验证：
   - 生成每一帧的彩色可视化图像，使用与全局静态掩码一致的颜色映射，便于验证重标记效果。
"""

class CellRelabelingPipeline:
    def __init__(self, config):
        self.cfg = config
        id = self.cfg['ID']
        self.dirs = {
            'static_mask': os.path.join(config['OUTPUT_DIR'], 'global_static_mask'),
            'relabeled_masks': os.path.join(config['OUTPUT_DIR'], f'Image {id} relabeled_masks'),
            'visualization': os.path.join(config['OUTPUT_DIR'], 'visualization_check')
        }
        for d in self.dirs.values(): os.makedirs(d, exist_ok=True)
        self.global_num_cells = 0

    def _load_sorted_files(self, folder):
        files = []
        pattern = re.compile(r'_t(\d+)')
        for f in sorted(os.listdir(folder)):
            if f.endswith('.nii.gz'):
                match = pattern.search(f)
                if match: files.append((int(match.group(1)), os.path.join(folder, f)))
        return sorted(files, key=lambda x: x[0])

    # ---------------------------------------------------------
    # 步骤 1: 生成全局静态掩码 (距离变换法 - 已修复维度问题)
    # ---------------------------------------------------------
    def step1_generate_static_mask(self):
        print(f"\n--- STEP 1: 生成全局静态掩码 (距离变换法) ---")
        files = self._load_sorted_files(self.cfg['RAW_MASK_DIR'])
        if not files: raise FileNotFoundError("未找到原始掩码。")

        first_img = nib.load(files[0][1])
        shape = first_img.shape
        affine = first_img.affine
        header = first_img.header

        # 1.1 初始化距离场累加器
        distance_accumulator = np.zeros(shape, dtype=np.float32)
        binary_accumulator = np.zeros(shape, dtype=np.float32)

        print(f"  正在处理 {len(files)} 帧 (计算每帧 EDT)...")

        for _, path in files:
            img = nib.load(path)
            # 确保是整数类型
            labels = img.get_fdata().astype(int)

            if np.max(labels) == 0: continue

            binary_accumulator += (labels > 0).astype(np.float32)

            # A. 找出边界
            bounds = find_boundaries(labels, mode='thick')

            # B. 准备距离变换输入 (前景且非边界)
            core_mask = (labels > 0) & (~bounds)

            # C. 计算距离变换
            edt_frame = ndimage.distance_transform_edt(core_mask)

            # D. 累加
            distance_accumulator += edt_frame

        # 1.2 处理累加图
        smooth_dist = ndimage.gaussian_filter(distance_accumulator, sigma=self.cfg['SIGMA'])

        # 1.3 寻找峰值
        print(f"  正在寻找峰值 (Min Dist={self.cfg['MIN_DISTANCE']}, Thresh={self.cfg['ABSOLUTE_THRESHOLD']})...")

        local_maxi = peak_local_max(smooth_dist,
                                    min_distance=self.cfg['MIN_DISTANCE'],
                                    threshold_abs=self.cfg['ABSOLUTE_THRESHOLD'],
                                    labels=(binary_accumulator > 0))

        seeds = np.zeros_like(distance_accumulator, dtype=int)
        for idx, point in enumerate(local_maxi):
            if len(point) == 2:
                seeds[point[0], point[1]] = idx + 1
            else:
                seeds[point[0], point[1], point[2]] = idx + 1

        # 1.4 分水岭分割
        print("  执行分水岭分割...")
        watershed_labels = watershed(-smooth_dist, seeds, mask=(binary_accumulator > 0))

        self.global_num_cells = len(local_maxi)

        # 保存
        out_path = os.path.join(self.dirs['static_mask'], 'Global_Static_Mask.nii.gz')
        nib.save(nib.Nifti1Image(watershed_labels.astype(np.uint16), affine, header), out_path)

        # --- 调试图 (关键修复：智能降维) ---
        vis_path = os.path.join(self.dirs['static_mask'], 'Global_Static_Mask_Debug.png')

        # 智能判断维度
        if watershed_labels.ndim == 3:
            vis_background = np.max(smooth_dist, axis=0)
            vis_labels = np.max(watershed_labels, axis=0)
        else:
            vis_background = smooth_dist
            vis_labels = watershed_labels

        plt.figure(figsize=(12, 12))
        plt.imshow(vis_background, cmap='magma')
        masked_vis = np.ma.masked_where(vis_labels == 0, vis_labels)
        plt.imshow(masked_vis, cmap='spring', alpha=0.4, interpolation='none')
        plt.title(f"Distance Accumulation + Watershed\nCells Found: {self.global_num_cells}")
        plt.axis('off')
        plt.savefig(vis_path)
        plt.close()

        print(f"  -> 全局掩码已保存。共找到 {self.global_num_cells} 个细胞。")
        print(f"  -> 请查看调试图: {vis_path}")
        return watershed_labels, files, affine, header

    # ---------------------------------------------------------
    # 步骤 2: 逐帧重标记 (保持投票法)
    # ---------------------------------------------------------
    def step2_relabel_frames(self, static_mask_data, raw_files, affine, header):
        print(f"\n--- STEP 2: 逐帧重标记 ---")
        for i, (frame_num, path) in enumerate(raw_files):
            if i % 20 == 0: print(f"  Processing frame {i}...")
            img = nib.load(path)
            curr_data = img.get_fdata().astype(np.uint16)
            new_mask = np.zeros_like(curr_data)

            blobs, num_blobs = label(curr_data > 0, return_num=True, connectivity=2)
            for blob_id in range(1, num_blobs + 1):
                blob_locs = (blobs == blob_id)
                global_ids = static_mask_data[blob_locs]
                valid_ids = global_ids[global_ids > 0]
                if valid_ids.size > 0:
                    new_mask[blob_locs] = np.argmax(np.bincount(valid_ids))

            nib.save(nib.Nifti1Image(new_mask, affine, header),
                     os.path.join(self.dirs['relabeled_masks'], os.path.basename(path)))
        print("  -> 重标记完成。")

    # ---------------------------------------------------------
    # 步骤 3: 可视化 (保持同步渲染 + 维度修复)
    # ---------------------------------------------------------
    def step3_visualize(self):
        print(f"\n--- STEP 3: 生成可视化验证图 ---")
        relabel_dir = self.dirs['relabeled_masks']
        vis_dir = self.dirs['visualization']
        files = sorted([f for f in os.listdir(relabel_dir) if f.endswith('.nii.gz')])

        max_id = self.global_num_cells + 10
        cmap = plt.get_cmap('tab20')
        base_colors = np.array(cmap.colors)
        rng = np.random.RandomState(42)

        color_lut = np.zeros((max_id + 1, 3), dtype=np.float32)
        for idx in range(1, max_id + 1):
            color_idx = idx % len(base_colors)
            shuffled_idx = rng.permutation(len(base_colors))[color_idx]
            color_lut[idx] = base_colors[shuffled_idx]

        # 静态参考
        static_path = os.path.join(self.dirs['static_mask'], 'Global_Static_Mask.nii.gz')
        if os.path.exists(static_path):
            s_data = nib.load(static_path).get_fdata().astype(int)
            # 修复：智能降维
            s_vis = np.max(s_data, axis=0) if s_data.ndim == 3 else s_data
            s_vis[s_vis > max_id] = 0
            plt.imsave(os.path.join(vis_dir, '000_Global_Reference.png'), color_lut[s_vis])

        # 动态帧
        for i, filename in enumerate(files):
            if i % 20 == 0: print(f"  Rendering frame {i}...")
            path = os.path.join(relabel_dir, filename)
            data = nib.load(path).get_fdata().astype(int)
            # 修复：智能降维
            d_vis = np.max(data, axis=0) if data.ndim == 3 else data
            d_vis[d_vis > max_id] = 0
            plt.imsave(os.path.join(vis_dir, filename.replace('.nii.gz', '.png')), color_lut[d_vis])

        print(f"  -> 可视化完成。")

    def run(self):
        static_mask, files, affine, header = self.step1_generate_static_mask()
        self.step2_relabel_frames(static_mask, files, affine, header)
        self.step3_visualize()
        print("\n=== 完成 ===")


if __name__ == "__main__":
    # ================= 配置区域 (Configuration) =================
    ids = [2,4,6,8,11,18,24,30,39]
    for id in ids:
        CONFIG = {
            # 1. 输入路径：存放原始掩码的文件夹
            'RAW_MASK_DIR': f"20251102_initial_code/data/data_20250817/agonism_mask/Image {id}",

            # 2. 输出路径：存放所有结果的根目录
            'OUTPUT_DIR': f"20251102_initial_code/data/data_20250817_tracked/agonism_mask/Image {id}",

            # 1. 最小峰值距离
            #    决定了两个细胞中心最少要隔多远。
            #    既然有粘连，这个值不要设太大，以免漏掉挤在一起的小细胞。
            #    建议：5 ~ 8
            'MIN_DISTANCE': 3,

            # 2. 绝对阈值 (Filter)
            #    累加后的距离场中，只有高度超过此值的峰才算有效细胞。
            #    数值越大，过滤越狠。
            #    如果发现丢失小细胞 -> 调小 (例如 5.0)
            #    如果发现太多噪点 -> 调大 (例如 20.0)
            'ABSOLUTE_THRESHOLD': 1.0,

            # 平滑系数 (保持地形连贯)
            'SIGMA': 1.0,
            'ID': id
        }

        # ===========================================================
        pipeline = CellRelabelingPipeline(CONFIG)
        pipeline.run()