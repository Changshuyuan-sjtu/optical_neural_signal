import os
import re
import logging
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from radiomics import featureextractor

"""
本脚本实现了一个完整的细胞影像放射组学特征提取与清洗流水线，
包括静态亮度特征和动态形态/纹理特征的计算与处理。
具体流程为：
1. 静态流 (Static Stream):
   - 使用全局静态掩码，基于 Numpy 直接计算每个细胞的平均亮度。
   - 优点：忽略头文件空间差异，绝对准确，无空值。 
2. 动态流 (Dynamic Stream):
   - 使用跟踪掩码，基于 PyRadiomics 计算每个细胞的形态和纹理特征。
   - 优点：标准化的形态与纹理计算。
3. 数据清洗与对齐:
    - 对动态特征进行线性插值，处理短间隙。
    - 根据配置选择长间隙处理策略（填0或前向填充）。
    - 输出最终清洗后的特征表格。
"""

# ===========================================

class FeatureExtractionPipeline:
    def __init__(self, config):
        self.cfg = config
        self.logger = self._setup_logger()

        self.dirs = {
            'final': os.path.join(config['OUTPUT_DIR'], f'final_results_{self.cfg["LONG_GAP_STRATEGY"]}')
        }
        for d in self.dirs.values(): os.makedirs(d, exist_ok=True)

    def _setup_logger(self):
        l = logging.getLogger('RadiomicsPipeline')
        l.setLevel(logging.INFO)
        if not l.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            l.addHandler(h)
        return l

    def _load_sorted_files(self, folder):
        files = []
        import re
        pattern = re.compile(r'_t(\d+)')
        for f in sorted(os.listdir(folder)):
            if f.endswith('.nii.gz'):
                match = pattern.search(f)
                if match:
                    files.append((int(match.group(1)), os.path.join(folder, f)))
        return sorted(files, key=lambda x: x[0])

    # ---------------------------------------------------------
    # 步骤 1: 提取静态特征 (亮度 + 纹理)
    # ---------------------------------------------------------
    def step1_extract_static_all(self):
        self.logger.info("--- STEP 1: 提取静态特征 (Intensity & Texture) ---")
        self.logger.info("  (基于 Global Mask，保证数据连续无缺失)")

        if not os.path.exists(self.cfg['GLOBAL_STATIC_MASK']):
            raise FileNotFoundError("找不到全局掩码。")

        # 1. 准备 PyRadiomics (用于纹理)
        #    force2D=True 确保按层计算
        settings = {'binWidth': 25, 'force2D': True}
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()

        # 启用 GLCM
        if self.cfg['STATIC_FEATURES_GLCM']:
            extractor.enableFeaturesByName(glcm=self.cfg['STATIC_FEATURES_GLCM'])

        # 2. 准备 Numpy (用于亮度，比 PyRadiomics 快)
        static_img_nib = nib.load(self.cfg['GLOBAL_STATIC_MASK'])
        static_data = static_img_nib.get_fdata().astype(int)
        global_ids = np.unique(static_data)
        global_ids = global_ids[global_ids > 0]

        roi_indices = {gid: np.where(static_data == gid) for gid in global_ids}

        # 3. 加载 SimpleITK 版本的 Mask (用于 PyRadiomics)
        static_mask_sitk = sitk.ReadImage(self.cfg['GLOBAL_STATIC_MASK'])

        image_files = self._load_sorted_files(self.cfg['RAW_IMAGE_DIR'])

        # 存储器
        results = {
            'Mean': [],
            'Contrast': [], 'JointEntropy': []
        }

        for i, (_, img_path) in enumerate(image_files):
            if i % 10 == 0: self.logger.info(f"  Processing frame {i}/{len(image_files)}...")

            # A. Numpy 读取 (算亮度)
            img_nib = nib.load(img_path)
            img_data_np = img_nib.get_fdata()

            # B. SimpleITK 读取 (算纹理)
            img_sitk = sitk.ReadImage(img_path)

            for gid in global_ids:
                # --- 算亮度 (Numpy) ---
                vals = img_data_np[roi_indices[gid]]
                mean_val = np.mean(vals) if vals.size > 0 else 0
                med_val = np.median(vals) if vals.size > 0 else 0

                results['Mean'].append({'frame': i, 'cell': gid, 'val': mean_val})

                # --- 算纹理 (PyRadiomics) ---
                # 只有在配置了 GLCM 时才算
                if self.cfg['STATIC_FEATURES_GLCM']:
                    try:
                        # execute(image, mask, label_id)
                        feats = extractor.execute(img_sitk, static_mask_sitk, label=int(gid))

                        # 提取 Contrast
                        if 'Contrast' in self.cfg['STATIC_FEATURES_GLCM']:
                            # 模糊匹配 'original_glcm_Contrast'
                            val = next((v for k, v in feats.items() if 'Contrast' in k), 0)
                            results['Contrast'].append({'frame': i, 'cell': gid, 'val': float(val)})

                        # 提取 Entropy
                        if 'JointEntropy' in self.cfg['STATIC_FEATURES_GLCM']:
                            val = next((v for k, v in feats.items() if 'JointEntropy' in k), 0)
                            results['JointEntropy'].append({'frame': i, 'cell': gid, 'val': float(val)})

                    except Exception:
                        # 计算失败填 0
                        if 'Contrast' in self.cfg['STATIC_FEATURES_GLCM']:
                            results['Contrast'].append({'frame': i, 'cell': gid, 'val': 0})
                        if 'JointEntropy' in self.cfg['STATIC_FEATURES_GLCM']:
                            results['JointEntropy'].append({'frame': i, 'cell': gid, 'val': 0})

        # 4. 保存静态特征 (直接 Pivot，无须清洗)
        self.logger.info("  正在保存静态特征 CSV...")
        for feat_name, data_list in results.items():
            if not data_list: continue

            df = pd.DataFrame(data_list)
            pivot = df.pivot(index='frame', columns='cell', values='val')
            pivot = pivot.reindex(columns=global_ids).fillna(0)
            pivot.columns = pivot.columns.astype(str)

            out_name = 'Entropy' if feat_name == 'JointEntropy' else feat_name
            pivot.to_csv(os.path.join(self.dirs['final'], f"Final_{out_name}.csv"))

        return global_ids

    # ---------------------------------------------------------
    # 步骤 2: 提取动态特征 (仅形态)
    # ---------------------------------------------------------
    def step2_extract_dynamic_shape(self, global_ids):
        self.logger.info("--- STEP 2: 提取动态形态 (Dynamic Shape) ---")

        if not self.cfg['DYNAMIC_FEATURES_SHAPE']:
            self.logger.info("  未配置动态形态特征，跳过。")
            return

        image_files = self._load_sorted_files(self.cfg['RAW_IMAGE_DIR'])
        mask_files = self._load_sorted_files(self.cfg['TRACKED_MASK_DIR'])
        total = min(len(image_files), len(mask_files))

        settings = {'binWidth': 25, 'force2D': True}
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        extractor.disableAllFeatures()
        extractor.enableFeaturesByName(shape2D=self.cfg['DYNAMIC_FEATURES_SHAPE'])

        all_data = []

        # 1. 提取
        for i in range(total):
            if i % 20 == 0: self.logger.info(f"  Processing frame {i}/{total}...")

            img_path = image_files[i][1]
            msk_path = mask_files[i][1]

            try:
                im_sitk = sitk.ReadImage(img_path)
                ma_sitk = sitk.ReadImage(msk_path)

                # Track ID 和 Global ID 在 v10 中是一致的
                labels = np.unique(sitk.GetArrayFromImage(ma_sitk))
                labels = labels[labels > 0]

                for label_id in labels:
                    label_id = int(label_id)
                    # 只计算属于 Global ID 列表里的 (过滤噪点)
                    if label_id not in global_ids: continue

                    try:
                        feats = extractor.execute(im_sitk, ma_sitk, label=label_id)
                        row = {'frame': i, 'cell': label_id}

                        for tf in self.cfg['DYNAMIC_FEATURES_SHAPE']:
                            val = np.nan
                            for k, v in feats.items():
                                if tf in k:
                                    val = float(v)
                                    break
                            row[tf] = val
                        all_data.append(row)
                    except Exception:
                        pass
            except Exception as e:
                self.logger.error(f"Frame {i} Error: {e}")

        # 2. 清洗与保存
        self.logger.info("  正在清洗动态特征...")
        df_raw = pd.DataFrame(all_data)

        frames = range(total)

        for feat in self.cfg['DYNAMIC_FEATURES_SHAPE']:
            # Pivot: 自动对齐 Track ID 到列
            pivot = df_raw.pivot(index='frame', columns='cell', values=feat)

            # 强制对齐到 Global ID (缺少的列填 NaN)
            pivot = pivot.reindex(index=frames, columns=global_ids)

            # --- 清洗 ---
            pivot = pivot.astype(float)
            # 插值
            # Fill all missing values with 0 (no interpolation or forward/back-fill)
            pivot = pivot.astype(float).fillna(0)

            out_name = 'Area' if feat == 'PixelSurface' else feat
            pivot.columns = pivot.columns.astype(str)
            pivot.to_csv(os.path.join(self.dirs['final'], f"Final_{out_name}.csv"))

        self.logger.info("动态特征处理完成。")

    def run(self):
        # 1. 静态流 (亮度+纹理) -> 产出无空值的 CSV
        global_ids = self.step1_extract_static_all()

        # 2. 动态流 (形态) -> 产出清洗后的 CSV
        self.step2_extract_dynamic_shape(global_ids)

        self.logger.info(f"所有结果已保存至: {self.dirs['final']}")


if __name__ == "__main__":

    ids = [5,7,9,12,18,24,32]
    for id in ids:
        print(f"Processing ID: {id}")
        # ================= 配置区域 =================

        CONFIG = {
            # 1. 输入路径
            'RAW_IMAGE_DIR': f"20251102_initial_code/data/data_20250817/antagonism/Image {id}",

            # track_cell 步骤生成的全局静态掩码
            'GLOBAL_STATIC_MASK': f"20251102_initial_code/data/data_20250817_tracked/antagonism_mask/Image {id}/global_static_mask/Global_Static_Mask.nii.gz",

            # track_cell 步骤生成的逐帧追踪掩码
            'TRACKED_MASK_DIR': f"20251102_initial_code/data/data_20250817_tracked/antagonism_mask/Image {id}/Image {id} relabeled_masks",

            # 2. 输出路径
            'OUTPUT_DIR': f"pre_analyze/antagonism_radiomics_features/Image {id}",

            # 3. 数据清洗策略 (仅用于动态形态特征)
            'LONG_GAP_STRATEGY': 'zero',

            # 4. 特征选择 (新策略)

            # [静态流] 亮度和纹理 -> 基于 Global Mask 计算 (无空值)
            'STATIC_FEATURES_INTENSITY': ['Mean'],
            #'STATIC_FEATURES_INTENSITY': [],
            'STATIC_FEATURES_GLCM': ['Contrast', 'JointEntropy'],  # 纹理移到这里
            #'STATIC_FEATURES_GLCM': [],
            # [动态流] 仅保留形态 -> 基于 Tracked Mask 计算 (有空值)
            'DYNAMIC_FEATURES_SHAPE': ['PixelSurface', 'Perimeter']
        }
        pipeline = FeatureExtractionPipeline(CONFIG)
        pipeline.run()