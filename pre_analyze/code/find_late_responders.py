import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.measure import regionprops
from skimage import io
import tifffile
import cv2  # [新增] 引入 OpenCV 用于生成视频和绘制圆圈

# 设置非交互模式
plt.ioff()

"""
本脚本用于基于 Mask 的细胞响应筛选与可视化，包含以下步骤：
1. 加载 Area 和 Mean 特征数据。
2. 根据预设逻辑筛选出可能的响应细胞事件，即在窗口末尾突然亮起来的事件。
3. 生成每个事件的拼图 (Montage)。
4. 生成全视野验证视频 (Full FOV Video)。

"""


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
    'OUTPUT_DIR': r"pre_analyze/results/jiekang/find_late_responders",

    # --- 3. 实验参数 ---
    'STIM_FRAMES': [31, 42, 52, 63, 73, 84, 95, 105, 116, 126, 137, 148, 158, 169, 179],
    'TOTAL_FRAMES': 190,

    # --- 4. 筛选逻辑参数 ---
    'GLOBAL_PERCENTILE': 70,
    'PEAK_LOOKBACK': 2,
    'CONTINUOUS_MISSING_LEN': 4,

    # --- 5. 裁剪参数 ---
    'DO_CROPPING': True,
    'PATCH_SIZE': 32,
    'TRANSPOSE_COORDS': False,

    # --- 6. 视频参数 ---
    'DO_VIDEO': True,
    'VIDEO_FPS': 1,  # 1秒一帧 = 1 FPS
    'CIRCLE_RADIUS': 15,  # 固定半径 15
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
    # 步骤 2: 筛选逻辑
    # ---------------------------------------------------------
    def step2_filter_responders(self, area_df, mean_df):
        print("\n--- STEP 2: 执行筛选 (Area存在 + 亮度前5% + 突现) ---")

        stim_frames_0based = [x - 1 for x in self.cfg['STIM_FRAMES']]
        total_frames = self.cfg['TOTAL_FRAMES']
        cont_missing = self.cfg['CONTINUOUS_MISSING_LEN']
        lookback = self.cfg['PEAK_LOOKBACK']
        global_pct = self.cfg['GLOBAL_PERCENTILE']

        tasks = []
        stats_rows = []

        for global_cell_id in area_df.columns:
            trace_area = area_df[global_cell_id].values
            trace_mean = mean_df[global_cell_id].values
            folder, original_cell_id = global_cell_id.split('___')

            intensity_threshold = np.percentile(trace_mean, global_pct)

            for i in range(len(stim_frames_0based)):
                start_f = stim_frames_0based[i]

                if i < len(stim_frames_0based) - 1:
                    end_f = stim_frames_0based[i + 1]
                else:
                    end_f = total_frames

                window_area = trace_area[start_f: end_f]
                window_mean = trace_mean[start_f: end_f]
                curr_win_len = len(window_area)

                if curr_win_len <= cont_missing + 1: continue

                found_anchor = False
                anchor_idx = -1

                for test_idx in range(curr_win_len - 1, curr_win_len - 1 - lookback, -1):
                    if test_idx < 0: break

                    if window_area[test_idx] > 0 and window_mean[test_idx] >= intensity_threshold:
                        start_check = test_idx - cont_missing
                        if start_check < 0: continue

                        preceding_areas = window_area[start_check: test_idx]
                        if np.all(preceding_areas == 0):
                            found_anchor = True
                            anchor_idx = test_idx
                            break

                if found_anchor:
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
                        'Anchor_Frame_In_Window': anchor_idx
                    })

        print(f"  筛选完成: 共发现 {len(tasks)} 个事件。")
        if stats_rows:
            pd.DataFrame(stats_rows).to_csv(os.path.join(self.cfg['OUTPUT_DIR'], 'filtered_stats.csv'), index=False)
        return pd.DataFrame(tasks)

    # ---------------------------------------------------------
    # [新增] 步骤 2.5: 统计分析
    # ---------------------------------------------------------
    def step2_5_statistics(self, tasks_df):
        if tasks_df.empty: return
        print("\n--- STEP 2.5: 生成统计报告 ---")

        # 1. 为了不修改 step2 的返回值结构，我们在这里根据 Start_Frame 反推 Stim_Index
        # 映射表: Start_Frame(0-based) -> Stim_Index(1-based)
        stim_map = {(f - 1): i + 1 for i, f in enumerate(self.cfg['STIM_FRAMES'])}

        # 给 tasks_df 增加两列用于统计
        # map 如果找不到会填 NaN，但理论上 Start_Frame 一定在配置里
        tasks_df['Stim_Index'] = tasks_df['Start_Frame'].map(stim_map)
        tasks_df['Start_Frame_1based'] = tasks_df['Start_Frame'] + 1

        # --- 需求 1: 统计每种刺激的出现次数 ---
        stim_counts = tasks_df.groupby(['Stim_Index', 'Start_Frame_1based']).size().reset_index(name='Event_Count')
        stim_counts = stim_counts.sort_values('Stim_Index')

        path_stim = os.path.join(self.dirs['report'], 'statistics_stim_frequency.csv')
        stim_counts.to_csv(path_stim, index=False)
        print(f"  -> 刺激频率统计已保存: {path_stim}")

        # --- 需求 2: 统计多次响应的细胞 (Count >= 2) ---
        # 按 Folder 和 Cell_ID 分组聚合
        cell_stats = tasks_df.groupby(['Folder', 'Cell_ID']).agg(
            Response_Count=('Stim_Index', 'count'),
            Responded_Stim_Indices=('Stim_Index', lambda x: list(sorted(x)))
        ).reset_index()

        # 筛选 >= 2 的细胞
        recurring_cells = cell_stats[cell_stats['Response_Count'] >= 2].copy()
        recurring_cells = recurring_cells.sort_values('Response_Count', ascending=False)

        path_cells = os.path.join(self.dirs['report'], 'statistics_recurring_cells.csv')
        recurring_cells.to_csv(path_cells, index=False)
        print(f"  -> 多次响应细胞统计 (>=2次) 已保存: {path_cells}")
        print(f"     共发现 {len(recurring_cells)} 个多次响应细胞。")

    # ---------------------------------------------------------
    # 步骤 3: 拼图生成 (Left-to-Right)
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
    # 步骤 4: 视频生成 (Full FOV + Annotation)
    # ---------------------------------------------------------
    def step4_generate_videos(self, tasks_df):
        if tasks_df.empty or not self.cfg['DO_VIDEO']: return
        print("\n--- STEP 4: 生成全视野验证视频 (Full FOV Video) ---")

        radius = self.cfg['CIRCLE_RADIUS']
        fps = self.cfg['VIDEO_FPS']

        for folder_name, group in tasks_df.groupby('Folder'):
            print(f"    正在为文件夹 {folder_name} 生成视频 ({len(group)} 个事件)...")

            # 1. 准备路径与坐标 (复用 Step 3 逻辑)
            mask_path = os.path.join(self.cfg['DATA_ROOT_DIR'], folder_name, 'final_results_zero',
                                     self.cfg['GLOBAL_MASK_NAME'])
            raw_img_folder = os.path.join(self.cfg['RAW_IMAGES_ROOT_DIR'], folder_name)
            save_video_dir = os.path.join(self.dirs['videos'], folder_name)
            os.makedirs(save_video_dir, exist_ok=True)

            if not os.path.exists(mask_path) or not os.path.exists(raw_img_folder): continue

            # 加载 Mask 获取坐标
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata().astype(int)
            if mask_data.ndim == 3: mask_data = np.max(mask_data, axis=-1)
            props = regionprops(mask_data)
            prop_map = {str(p.label): p.centroid for p in props}  # ID -> (row, col)

            # 准备文件列表
            raw_files = [f for f in os.listdir(raw_img_folder) if f.endswith(self.cfg['RAW_IMAGE_EXT'])]

            def get_t_num(fname):
                match = re.search(r'_t(\d+)', fname)
                if match:
                    return int(match.group(1))
                else:
                    nums = re.findall(r'\d+', fname)
                    return int(nums[-1]) if nums else 0

            raw_files.sort(key=get_t_num)

            # 2. 遍历该文件夹下的每个事件
            for _, row in group.iterrows():
                cid = str(row['Cell_ID'])
                start_f = row['Start_Frame']
                win_len = row['Window_Len']

                if cid not in prop_map: continue

                # 获取该事件的起始文件名标号 (用于命名)
                t_num_start = get_t_num(raw_files[start_f])
                video_name = f"{folder_name}_Cell{cid}_t{t_num_start:03d}_Video.avi"
                video_path = os.path.join(save_video_dir, video_name)

                # 获取细胞中心坐标 (注意: prop返回 row,col; cv2需要 x,y 即 col,row)
                cy, cx = prop_map[cid]
                center_point = (int(cx), int(cy))

                # 初始化 VideoWriter
                # 我们先读取第一帧来确定视频尺寸
                first_img_path = os.path.join(raw_img_folder, raw_files[start_f])
                temp_img = io.imread(first_img_path)
                h, w = temp_img.shape[:2]

                # MJPG 编码
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h), isColor=True)

                # 逐帧写入
                for t in range(win_len):
                    curr_f_idx = start_f + t
                    if curr_f_idx >= len(raw_files): break

                    img_path = os.path.join(raw_img_folder, raw_files[curr_f_idx])
                    try:
                        # 1. 读取
                        img = io.imread(img_path)

                        # 2. 归一化 (Step3 同款逻辑，保证亮度一致)
                        if img.dtype != np.uint8:
                            img = img.astype(float)
                            p1, p99 = np.percentile(img, (1, 99))
                            img = (img - p1) / (p99 - p1 + 1e-6)
                            img = np.clip(img, 0, 1) * 255
                            img = img.astype(np.uint8)

                        # 3. 转为 BGR (OpenCV 默认格式)
                        if img.ndim == 2:
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        else:
                            # 假设 skimage 读入的是 RGB
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        # 4. 绘制白色圆圈
                        # (img, center, radius, color, thickness)
                        cv2.circle(img_bgr, center_point, radius, (255, 255, 255), 2)

                        # 5. 写入视频
                        video_writer.write(img_bgr)

                    except Exception as e:
                        print(f"      [Error] Video frame write failed: {e}")
                        continue

                video_writer.release()

            print(f"    -> 文件夹 {folder_name} 视频生成完毕。")

    def run(self):
        area_df, mean_df = self.step1_load_data()
        tasks_df = self.step2_filter_responders(area_df, mean_df)

        # [新增] 插入统计分析
        self.step2_5_statistics(tasks_df)

        self.step3_generate_montages(tasks_df)
        self.step4_generate_videos(tasks_df)
        print("\n--- 全部处理完成 ---")


if __name__ == "__main__":
    pipeline = MaskBasedResponderPipeline(CONFIG)
    pipeline.run()