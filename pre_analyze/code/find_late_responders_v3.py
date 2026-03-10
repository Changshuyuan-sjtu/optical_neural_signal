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

# 设置非交互模式
plt.ioff()

"""
本脚本用于基于 Mask 的细胞响应筛选与可视化，包含以下步骤：
1. 加载 Area 和 Mean 特征数据。
2. 根据 Area 和 Mean 的时间序列进行筛选，找出符合条件的细胞事件。
3. 生成每个事件的拼图 (Montage)。
4. 生成包含标注的全视野视频 (Full FOV Video)。
注意v3版本的筛选逻辑和之前不同。
筛选条件一：窗口的最后两帧存在对应的mask(确保是激活细胞)
筛选条件二：窗口的亮度曲线存在一个下去又起来的趋势即可，且“起来”的幅度需大于设定阈值 (RISE_THRESHOLD)，以防止噪点影响。
"""


# ================= 配置区域 =================

CONFIG = {
    # --- 1. 路径配置 ---
    'DATA_ROOT_DIR': r"pre_analyze/agonism_radiomics_features",
    'RAW_IMAGES_ROOT_DIR': r"20251102_initial_code/data/data_20250817/agonism",

    'FEATURE_FILES': {
        'Area': 'Final_Area.csv',
        'Mean': 'Final_Mean.csv',
    },

    'GLOBAL_MASK_NAME': 'Global_Static_Mask.nii.gz',
    'RAW_IMAGE_EXT': '.jpg',

    # --- 2. 输出路径 ---
    'OUTPUT_DIR': r"pre_analyze/results/jidong/find_late_responders_v3_valley_stats",

    # --- 3. 实验参数 ---
    'STIM_FRAMES': [31, 42, 52, 63, 73, 84, 95, 105, 116, 126, 137, 148, 158, 169, 179],
    'TOTAL_FRAMES': 190,

    # --- 4. 筛选逻辑参数 (Version 2) ---
    # 只要倒数第1帧或倒数第2帧有Mask即可 (Loose Mask)
    'MASK_LOOKBACK': 2,

    # [新] 亮度回升阈值:
    # 也就是“下去又起来”中，“起来”的那一段高度必须大于这个值，防止噪点
    # 根据您的图片位深调整 (8bit图建议5-10, 16bit图建议更大)
    'RISE_THRESHOLD': 5.0,

    # --- 5. 裁剪参数 ---
    'DO_CROPPING': True,
    'PATCH_SIZE': 32,
    'TRANSPOSE_COORDS': False,

    # --- 6. 视频参数 ---
    'DO_VIDEO': True,
    'VIDEO_FPS': 2,  # 0.5秒一帧 = 2 FPS
    'CIRCLE_RADIUS': 15,
}


# ===========================================

class MaskBasedResponderPipeline:
    def __init__(self, config):
        self.cfg = config
        self.dirs = {
            'montages': os.path.join(config['OUTPUT_DIR'], 'montages'),
            'videos': os.path.join(config['OUTPUT_DIR'], 'videos'),
            'report': os.path.join(config['OUTPUT_DIR'], 'report')
        }
        for d in self.dirs.values(): os.makedirs(d, exist_ok=True)

    # ---------------------------------------------------------
    # 步骤 1: 加载数据
    # ---------------------------------------------------------
    def step1_load_data(self):
        print("\n--- STEP 1: 加载特征数据 (Area & Mean) ---")
        root = self.cfg['DATA_ROOT_DIR']
        subfolders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]

        area_dfs = []
        mean_dfs = []

        for folder in subfolders:
            folder_path = os.path.join(root, folder, 'final_results_zero')
            path_area = os.path.join(folder_path, self.cfg['FEATURE_FILES']['Area'])
            path_mean = os.path.join(folder_path, self.cfg['FEATURE_FILES']['Mean'])

            if not os.path.exists(path_area) or not os.path.exists(path_mean):
                continue

            # Pandas 读取时自动处理 index (Frame 0-189)
            df_a = pd.read_csv(path_area, index_col=0)
            df_a.columns = [f"{folder}___{col}" for col in df_a.columns]
            area_dfs.append(df_a)

            df_m = pd.read_csv(path_mean, index_col=0)
            df_m.columns = [f"{folder}___{col}" for col in df_m.columns]
            mean_dfs.append(df_m)

        if not area_dfs or not mean_dfs:
            raise ValueError("未找到特征文件！")

        big_area_df = pd.concat(area_dfs, axis=1).fillna(0)
        big_mean_df = pd.concat(mean_dfs, axis=1).fillna(0)

        common_cols = big_area_df.columns.intersection(big_mean_df.columns)
        return big_area_df[common_cols], big_mean_df[common_cols]

    # ---------------------------------------------------------
    # 步骤 2: 筛选逻辑 (Version 2: Valley Detection + Stats)
    # ---------------------------------------------------------
    def step2_filter_responders(self, area_df, mean_df):
        print("\n--- STEP 2: 执行筛选 (V2: Mask OR + Down-Up Trend) ---")

        stim_frames_0based = [x - 1 for x in self.cfg['STIM_FRAMES']]
        total_frames = self.cfg['TOTAL_FRAMES']
        rise_thresh = self.cfg['RISE_THRESHOLD']  # 亮度回升阈值

        tasks = []
        stats_rows = []

        for global_cell_id in area_df.columns:
            trace_area = area_df[global_cell_id].values
            trace_mean = mean_df[global_cell_id].values
            folder, original_cell_id = global_cell_id.split('___')

            for i in range(len(stim_frames_0based)):
                start_f = stim_frames_0based[i]

                if i < len(stim_frames_0based) - 1:
                    end_f = stim_frames_0based[i + 1]
                else:
                    end_f = total_frames

                window_area = trace_area[start_f: end_f]
                window_mean = trace_mean[start_f: end_f]
                curr_win_len = len(window_area)

                if curr_win_len < 3: continue  # 窗口太短无法判断 V 型

                # --- 条件一：Mask 存在性 (OR Logic) ---
                # 检查最后两帧是否存在任意 Area > 0
                has_mask = False
                # 检查范围: [-1] 和 [-2] (如果窗口够长)
                check_range = min(2, curr_win_len)
                for k in range(1, check_range + 1):
                    if window_area[-k] > 0:
                        has_mask = True
                        break

                if not has_mask:
                    continue

                # --- 条件二：形态趋势 ("下去又起来") ---
                # 寻找局部极小值 (Local Minimum / Valley)
                # 定义：存在索引 k (1 <= k <= Len-2)，使得 x[k] < x[k-1] 且 x[k] < x[k+1]
                # 且为了排除微小抖动，要求最终亮度回升幅度 > 阈值

                found_valley = False
                valley_idx = -1

                # 遍历中间部分，寻找谷底
                for k in range(1, curr_win_len - 1):
                    val = window_mean[k]
                    prev = window_mean[k - 1]
                    next_val = window_mean[k + 1]

                    # 判断是否是局部低点 (Down ... Up)
                    if val < prev and val < next_val:
                        # 找到了谷底，现在检查回升幅度
                        # 这里的回升指：窗口结束时的亮度 - 谷底亮度
                        rise_amplitude = window_mean[-1] - val

                        if rise_amplitude >= rise_thresh:
                            found_valley = True
                            valley_idx = k
                            break  # 只要找到一个符合的深谷即可

                if found_valley:
                    tasks.append({
                        'Folder': folder,
                        'Cell_ID': original_cell_id,
                        'Start_Frame': start_f,
                        'Window_Len': curr_win_len
                    })
                    stats_rows.append({
                        'Folder': folder,
                        'Cell_ID': original_cell_id,
                        'Start_Frame_0based': start_f,
                        'Valley_Index': valley_idx,
                        'Rise_Amplitude': window_mean[-1] - window_mean[valley_idx]
                    })

        print(f"  筛选完成: 共发现 {len(tasks)} 个事件。")
        if stats_rows:
            pd.DataFrame(stats_rows).to_csv(os.path.join(self.cfg['OUTPUT_DIR'], 'filtered_stats_debug.csv'),
                                            index=False)
        return pd.DataFrame(tasks)

    # ---------------------------------------------------------
    # 步骤 2.5: 统计分析 (Statistics)
    # ---------------------------------------------------------
    def step2_5_statistics(self, tasks_df):
        if tasks_df.empty: return
        print("\n--- STEP 2.5: 生成统计报告 ---")

        # 1. 映射 Stim_Index
        # 建立映射: Start_Frame(0-based) -> Stim_Index(1-based)
        stim_map = {(f - 1): i + 1 for i, f in enumerate(self.cfg['STIM_FRAMES'])}

        # 使用 .map 映射，创建一个新列。如果 Start_Frame 不在 map 里 (理论不可能)，会是 NaN
        tasks_df['Stim_Index'] = tasks_df['Start_Frame'].map(stim_map).fillna(0).astype(int)

        # --- 统计 1: 每种刺激的出现频次 ---
        # 统计每个 Stim_Index 出现了多少次
        stim_counts = tasks_df['Stim_Index'].value_counts().sort_index().reset_index()
        stim_counts.columns = ['Stim_Index', 'Event_Count']

        # 为了完整性，最好把 1-15 都列出来，即使是 0
        all_stims = pd.DataFrame({'Stim_Index': range(1, 16)})
        stim_counts = pd.merge(all_stims, stim_counts, on='Stim_Index', how='left').fillna(0)
        stim_counts['Event_Count'] = stim_counts['Event_Count'].astype(int)

        path_stim = os.path.join(self.dirs['report'], 'statistics_stim_counts.csv')
        stim_counts.to_csv(path_stim, index=False)
        print(f"  -> 刺激频次统计已保存: {path_stim}")

        # --- 统计 2: 多次响应的细胞 (Count >= 2) ---
        # 按 Folder, Cell_ID 分组
        cell_groups = tasks_df.groupby(['Folder', 'Cell_ID'])

        recurring_data = []
        for (folder, cell), group in cell_groups:
            count = len(group)
            if count >= 2:
                # 获取该细胞响应的所有 Stim Index，排序并转字符串
                stim_indices = sorted(group['Stim_Index'].unique())
                stim_str = ",".join(map(str, stim_indices))

                recurring_data.append({
                    'Folder': folder,
                    'Cell_ID': cell,
                    'Response_Count': count,
                    'Responded_Stim_Indices': stim_str
                })

        if recurring_data:
            recurring_df = pd.DataFrame(recurring_data)
            # 按次数降序排列
            recurring_df = recurring_df.sort_values('Response_Count', ascending=False)
            path_rec = os.path.join(self.dirs['report'], 'statistics_recurring_cells.csv')
            recurring_df.to_csv(path_rec, index=False)
            print(f"  -> 多次响应细胞统计 (>=2次) 已保存: {path_rec}")
            print(f"     发现 {len(recurring_df)} 个“惯犯”细胞。")
        else:
            print("  -> 未发现响应 >= 2 次的细胞。")

    # ---------------------------------------------------------
    # 步骤 3: 拼图生成
    # ---------------------------------------------------------
    def step3_generate_montages(self, tasks_df):
        if tasks_df.empty or not self.cfg['DO_CROPPING']: return
        print("\n--- STEP 3: 生成拼图 (Montage) ---")

        patch_size = self.cfg['PATCH_SIZE']
        patch_r = patch_size // 2

        for folder_name, group in tasks_df.groupby('Folder'):
            mask_path = os.path.join(self.cfg['DATA_ROOT_DIR'], folder_name, 'final_results_zero',
                                     self.cfg['GLOBAL_MASK_NAME'])
            raw_img_folder = os.path.join(self.cfg['RAW_IMAGES_ROOT_DIR'], folder_name)
            save_folder_dir = os.path.join(self.dirs['montages'], folder_name)
            os.makedirs(save_folder_dir, exist_ok=True)

            if not os.path.exists(mask_path) or not os.path.exists(raw_img_folder): continue

            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata().astype(int)
            if mask_data.ndim == 3: mask_data = np.max(mask_data, axis=-1)
            props = regionprops(mask_data)
            prop_map = {str(p.label): p.centroid for p in props}

            raw_files = [f for f in os.listdir(raw_img_folder) if f.endswith(self.cfg['RAW_IMAGE_EXT'])]

            # 排序逻辑: _t(\d+)
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
                if cid not in prop_map: continue

                key = (cid, start_f)
                montage_buffers[key] = np.zeros((patch_size, patch_size * win_len, 3), dtype=np.uint8)

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
                    montage_buffers[key][:, x_start: x_start + patch_size, :] = patch

            for (cid, start_f), canvas in montage_buffers.items():
                w_len = canvas.shape[1] // patch_size
                actual_filename = file_map.get(start_f, f"Unknown_Idx{start_f}")
                t_num = get_t_num(actual_filename)
                fname = f"{folder_name}_Cell{cid}_t{t_num:03d}_Len{w_len}.tif"
                io.imsave(os.path.join(save_folder_dir, fname), canvas, check_contrast=False)

            print(f"    -> 文件夹 {folder_name} 拼图生成完毕 ({len(montage_buffers)} 张)。")

    # ---------------------------------------------------------
    # 步骤 4: 视频生成
    # ---------------------------------------------------------
    def step4_generate_videos(self, tasks_df):
        if tasks_df.empty or not self.cfg['DO_VIDEO']: return
        print("\n--- STEP 4: 生成视频 (Video) ---")

        radius = self.cfg['CIRCLE_RADIUS']
        fps = self.cfg['VIDEO_FPS']

        for folder_name, group in tasks_df.groupby('Folder'):
            print(f"    正在为文件夹 {folder_name} 生成视频...")

            mask_path = os.path.join(self.cfg['DATA_ROOT_DIR'], folder_name, 'final_results_zero',
                                     self.cfg['GLOBAL_MASK_NAME'])
            raw_img_folder = os.path.join(self.cfg['RAW_IMAGES_ROOT_DIR'], folder_name)
            save_video_dir = os.path.join(self.dirs['videos'], folder_name)
            os.makedirs(save_video_dir, exist_ok=True)

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

            for _, row in group.iterrows():
                cid = str(row['Cell_ID'])
                start_f = row['Start_Frame']
                win_len = row['Window_Len']

                if cid not in prop_map: continue

                t_num_start = get_t_num(raw_files[start_f])
                video_name = f"{folder_name}_Cell{cid}_t{t_num_start:03d}_Video.avi"
                video_path = os.path.join(save_video_dir, video_name)

                cy, cx = prop_map[cid]
                center_point = (int(cx), int(cy))

                first_img_path = os.path.join(raw_img_folder, raw_files[start_f])
                temp_img = io.imread(first_img_path)
                h, w = temp_img.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h), isColor=True)

                for t in range(win_len):
                    curr_f_idx = start_f + t
                    if curr_f_idx >= len(raw_files): break

                    img_path = os.path.join(raw_img_folder, raw_files[curr_f_idx])
                    try:
                        img = io.imread(img_path)
                        if img.dtype != np.uint8:
                            img = img.astype(float)
                            p1, p99 = np.percentile(img, (1, 99))
                            img = (img - p1) / (p99 - p1 + 1e-6)
                            img = np.clip(img, 0, 1) * 255
                            img = img.astype(np.uint8)

                        if img.ndim == 2:
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        else:
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        cv2.circle(img_bgr, center_point, radius, (255, 255, 255), 2)
                        video_writer.write(img_bgr)
                    except Exception as e:
                        print(f"      [Error] Video write: {e}")
                        continue

                video_writer.release()

            print(f"    -> 视频生成完毕。")

    def run(self):
        area_df, mean_df = self.step1_load_data()
        tasks_df = self.step2_filter_responders(area_df, mean_df)
        self.step2_5_statistics(tasks_df)  # [新增] 统计步骤
        self.step3_generate_montages(tasks_df)
        self.step4_generate_videos(tasks_df)
        print("\n--- 全部处理完成 ---")


if __name__ == "__main__":
    pipeline = MaskBasedResponderPipeline(CONFIG)
    pipeline.run()