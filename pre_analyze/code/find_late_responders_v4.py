import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.measure import regionprops
from skimage import io
import tifffile
import cv2
import datetime
import warnings

"""
作者：Changsy
本脚本旨在提供一个统一的框架，用于筛查细胞荧光数据中的异常响应事件，特别是针对晚响应者的捕捉。通过形态学特征和全局静态Mask的结合，能够更精准地识别出真正的细胞响应，避免背景噪点的干扰。
主要功能包括：
1. 数据加载：从指定目录批量加载细胞的 Mean 和 Area 特征数据。
2. 异常筛查：基于设定的阈值和严格度，对每个细胞在每次刺激事件的响应窗口内进行分析，捕捉到明显的 V 型趋势且满足面积条件的异常响应事件。
3. 统计分析：对捕捉到的异常事件进行统计，生成各刺激事件的响应频率报告，以及发生过多次异常的细胞统计。
4. 可视化：为每个异常事件生成高精度的拼图，展示响应窗口内的原始图像和 Mean 曲线，并可选地生成带有响应标记的视频。
使用说明：
- 配置项：通过 CONFIG 字典中的参数，可以灵活控制模块的开启与关闭、路径设置、时间参数、筛查严格度等。
- 输出结果：所有结果将保存在 OUTPUT_DIR 指定的目录下，按照时间戳和筛查等级进行组织，便于后续查看和分析。
"""



warnings.filterwarnings('ignore')
plt.ioff()

# =====================================================================
#                          中央配置字典 (CONFIG)
# =====================================================================
CONFIG = {
    # ---------------- 1. 模块自由开关 ----------------
    'DO_FILTERING': True,
    'DO_STATISTICS': True,
    'DO_MONTAGE': False,
    'DO_VIDEO': False,

    # ---------------- 2. 路径配置 ----------------
    'CSV_ROOT_DIR': r'/home/changsy/Optical_neural_signal/pre_analyze/new_data_20260309_features',
    'IMG_ROOT_DIR': r'/home/data/changsy/optical_neural_signal/new_data_20260309/data_processed_relabel_jpg',
    'MASK_ROOT_DIR': r'/home/data/changsy/optical_neural_signal/new_data_20260309/instance_mask_tracked',
    'OUTPUT_DIR': r'/home/changsy/Optical_neural_signal/pre_analyze/results/new_data_20260309/unified_output_late_responders',
    # 支持局部排查，如 ['Image 1']；设为 None 则遍历所有 Image 文件夹
    'TARGET_FOLDERS': None,

    'FILE_EXTENSION': '.jpg',
    'TARGET_MEAN_NAME': 'final_results_zero/Final_Mean.csv',
    'TARGET_AREA_NAME': 'final_results_zero/Final_Area.csv',  # 用于严格模式的面积校验
    'MASK_RELATIVE_PATH': 'global_static_mask/Global_Static_Mask.nii.gz',

    # ---------------- 3. 实验时间参数 ----------------
    'FPS_ACTUAL': 1.0 / 0.94556,
    'TIME_PER_FRAME': 0.94556,
    'TIME_INIT_BLANK': 3.696,
    'TIME_STIM_ON': 3.696,
    'TIME_STIM_OFF': 7.392,
    'NUM_DIRECTIONS': 8,
    'REPETITIONS_PER_DIR': 15,
    'SEQUENCE_TYPE': 'block',
    'TOTAL_FRAMES': 1500,

    # ---------------- 4. 异常筛查参数 (形态学 + Mask) ----------------
    # 筛查严格度
    # 1: 宽松 (仅依靠亮度V型趋势，可能抓到背景噪点)
    # 2: 严格 (亮度V型趋势 + 窗口内必须有细胞的 Area > 0)
    'FILTER_STRICTNESS': 2,

    'RISE_THRESHOLD': 5.0,

    # ---------------- 5. 拼图与视频参数 ----------------
    'PATCH_SIZE': 32,
    'PLOT_HEIGHT': 160,
    'TRANSPOSE_COORDS': False,
    'VIDEO_FPS': 2,
    'CIRCLE_RADIUS': 15,
}


# =====================================================================
#                          核心功能类
# =====================================================================

class AbnormalResponderTracker:
    def __init__(self, config):
        self.cfg = config
        self.events = self._generate_stimulus_events()
        self.dirs = self._setup_directories()

    def _setup_directories(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"responders_{timestamp}_Lv{self.cfg['FILTER_STRICTNESS']}_thresh{self.cfg['RISE_THRESHOLD']}"
        base_dir = os.path.join(self.cfg['OUTPUT_DIR'], run_name)

        dirs = {
            'base': base_dir,
            'report': os.path.join(base_dir, '01_reports'),
            'montages': os.path.join(base_dir, '02_montages'),
            'videos': os.path.join(base_dir, '03_videos')
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        print(f"[*] 初始化完毕。筛查等级: Level {self.cfg['FILTER_STRICTNESS']}。输出目录: {base_dir}")
        return dirs

    def _generate_stimulus_events(self):
        events = []
        total_trials = self.cfg['NUM_DIRECTIONS'] * self.cfg['REPETITIONS_PER_DIR']
        cycle_time = self.cfg['TIME_STIM_ON'] + self.cfg['TIME_STIM_OFF']

        for i in range(total_trials):
            t_start_sec = self.cfg['TIME_INIT_BLANK'] + i * cycle_time
            frame_idx = math.ceil(t_start_sec / self.cfg['TIME_PER_FRAME'])

            if self.cfg['SEQUENCE_TYPE'] == 'block':
                dir_idx = i // self.cfg['REPETITIONS_PER_DIR']
                rep_idx = i % self.cfg['REPETITIONS_PER_DIR']
            else:
                dir_idx = i % self.cfg['NUM_DIRECTIONS']
                rep_idx = i // self.cfg['NUM_DIRECTIONS']

            events.append({
                'trial_id': i,
                'stim_frame': frame_idx,
                'direction': dir_idx * 45,
                'repetition': rep_idx + 1
            })
        return events

    def _get_target_folders(self):
        if self.cfg['TARGET_FOLDERS'] is not None:
            return self.cfg['TARGET_FOLDERS']
        folders = []
        for d in os.listdir(self.cfg['CSV_ROOT_DIR']):
            if os.path.isdir(os.path.join(self.cfg['CSV_ROOT_DIR'], d)):
                folders.append(d)
        return sorted(folders)

    def _load_features_data(self):
        """根据严格度加载所需数据"""
        print(f"\n--- [STEP 1] 加载细胞特征数据 (Level {self.cfg['FILTER_STRICTNESS']}) ---")
        folders = self._get_target_folders()
        all_mean_dfs = []
        all_area_dfs = []

        for folder in folders:
            mean_path = os.path.join(self.cfg['CSV_ROOT_DIR'], folder, self.cfg['TARGET_MEAN_NAME'])
            if not os.path.exists(mean_path): continue

            try:
                # 1. 永远加载 Mean
                df_m = pd.read_csv(mean_path, index_col=0)
                df_m = df_m.reindex(range(self.cfg['TOTAL_FRAMES']), method='ffill')
                df_m.columns = [f"{folder}___{col}" for col in df_m.columns]
                all_mean_dfs.append(df_m)

                # 2. 如果是 Level 2，同步加载 Area
                if self.cfg['FILTER_STRICTNESS'] == 2:
                    area_path = os.path.join(self.cfg['CSV_ROOT_DIR'], folder, self.cfg['TARGET_AREA_NAME'])
                    if os.path.exists(area_path):
                        df_a = pd.read_csv(area_path, index_col=0)
                        df_a = df_a.reindex(range(self.cfg['TOTAL_FRAMES']), method='ffill')
                        df_a.columns = [f"{folder}___{col}" for col in df_a.columns]
                        all_area_dfs.append(df_a)

            except Exception as e:
                print(f"[!] 加载 {folder} 失败: {e}")

        big_mean_df = pd.concat(all_mean_dfs, axis=1)
        big_mean_df = big_mean_df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        big_area_df = None
        if self.cfg['FILTER_STRICTNESS'] == 2 and all_area_dfs:
            big_area_df = pd.concat(all_area_dfs, axis=1)
            big_area_df = big_area_df.fillna(0)  # 面积缺失直接补0

            # 确保列名完全对齐
            common_cols = big_mean_df.columns.intersection(big_area_df.columns)
            big_mean_df = big_mean_df[common_cols]
            big_area_df = big_area_df[common_cols]

        print(f"[*] 成功加载特征，共计 {big_mean_df.shape[1]} 个细胞。")
        return big_mean_df, big_area_df

    # ---------------------------------------------------------
    # 核心步骤: 形态学 + Mask 筛选
    # ---------------------------------------------------------
    def run_filtering(self, mean_df, area_df):
        print("\n--- [STEP 2] 执行异常响应筛查 ---")
        thresh = self.cfg['RISE_THRESHOLD']
        strictness = self.cfg['FILTER_STRICTNESS']
        tasks = []

        for global_cell_id in mean_df.columns:
            trace_mean = mean_df[global_cell_id].values

            # 如果是等级2，提取面积轨迹
            trace_area = area_df[global_cell_id].values if strictness == 2 else None

            folder, original_cell_id = global_cell_id.split('___')

            for i in range(len(self.events)):
                ev = self.events[i]
                start_f = ev['stim_frame']

                if i < len(self.events) - 1:
                    end_f = self.events[i + 1]['stim_frame']
                else:
                    end_f = min(start_f + 12, self.cfg['TOTAL_FRAMES'])

                window_mean = trace_mean[start_f: end_f]
                win_len = len(window_mean)
                if win_len < 3: continue

                # === [新增校验]: 严格模式下，窗口内必须有细胞 ===
                if strictness == 2:
                    window_area = trace_area[start_f: end_f]
                    if np.max(window_area) <= 0:
                        continue  # 整个窗口内连一个像素的细胞都没识别到，直接视为背景噪点，抛弃
                # ===============================================

                found_valley = False
                valley_idx = -1

                for k in range(1, win_len - 1):
                    val = window_mean[k]
                    prev = window_mean[k - 1]
                    next_val = window_mean[k + 1]

                    if val < prev and val < next_val:
                        rise_amplitude = window_mean[-1] - val
                        if rise_amplitude >= thresh:
                            found_valley = True
                            valley_idx = k
                            break

                if found_valley:
                    tasks.append({
                        'Folder': folder,
                        'Cell_ID': original_cell_id,
                        'Global_ID': global_cell_id,
                        'Start_Frame': start_f,
                        'Window_Len': win_len,
                        'Direction': ev['direction'],
                        'Repetition': ev['repetition'],
                        'Trial_ID': ev['trial_id'],
                        'Rise_Amplitude': window_mean[-1] - window_mean[valley_idx]
                    })

        tasks_df = pd.DataFrame(tasks)
        print(f"  -> 筛查完成！共捕捉到 {len(tasks_df)} 次有效异常响应事件。")
        if not tasks_df.empty:
            tasks_df.to_csv(os.path.join(self.dirs['report'], 'all_detected_events.csv'), index=False)
        return tasks_df

    # ---------------------------------------------------------
    # 统计生成
    # ---------------------------------------------------------
    def run_statistics(self, tasks_df):
        if tasks_df.empty or not self.cfg['DO_STATISTICS']: return
        print("\n--- [STEP 3] 生成统计报告 ---")

        stim_counts = tasks_df['Trial_ID'].value_counts().sort_index().reset_index()
        stim_counts.columns = ['Trial_ID', 'Event_Count']
        all_trials = pd.DataFrame({'Trial_ID': range(len(self.events))})
        stim_counts = pd.merge(all_trials, stim_counts, on='Trial_ID', how='left').fillna(0).astype(int)

        meta = pd.DataFrame(self.events)[['trial_id', 'direction', 'repetition']]
        stim_counts = pd.merge(stim_counts, meta, left_on='Trial_ID', right_on='trial_id').drop(columns=['trial_id'])
        stim_counts.to_csv(os.path.join(self.dirs['report'], 'stats_trial_frequencies.csv'), index=False)

        cell_groups = tasks_df.groupby(['Folder', 'Cell_ID'])
        recurring_data = []
        for (folder, cell), group in cell_groups:
            if len(group) >= 2:
                recurring_data.append({
                    'Folder': folder,
                    'Cell_ID': cell,
                    'Response_Count': len(group),
                    'Directions_Triggered': ",".join(map(str, sorted(group['Direction'].unique()))),
                    'Max_Rise_Amplitude': group['Rise_Amplitude'].max()
                })

        if recurring_data:
            df = pd.DataFrame(recurring_data).sort_values('Response_Count', ascending=False)
            df.to_csv(os.path.join(self.dirs['report'], 'stats_recurring_cells.csv'), index=False)
            print(f"  -> 已保存惯犯细胞统计，共有 {len(df)} 个细胞发生过多次异常。")

    # ---------------------------------------------------------
    # OpenCV Sparkline 绘图助手
    # ---------------------------------------------------------
    def _draw_sparkline(self, data, width, height):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        if len(data) < 2: return img

        padding = 10
        d_min, d_max = np.min(data), np.max(data)
        scale = 0 if d_max == d_min else (height - 2 * padding) / (d_max - d_min)

        def to_pt(idx, val):
            patch_w = width / len(data)
            x = int(idx * patch_w + patch_w / 2)
            y = int(height - padding - (val - d_min) * scale)
            return (x, y)

        ref_val = data[0]
        y_ref = int(height - padding - (ref_val - d_min) * scale)
        cv2.line(img, (0, y_ref), (width, y_ref), (100, 100, 100), 1)

        pts = [to_pt(i, v) for i, v in enumerate(data)]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (255, 255, 0), 2, cv2.LINE_AA)
        return img

    def _index_image_files(self, img_dir):
        image_files = {}
        pattern = re.compile(r'(\d+)')
        ext = self.cfg['FILE_EXTENSION'].lower()
        if not os.path.exists(img_dir): return image_files

        for f in os.listdir(img_dir):
            if f.lower().endswith(ext):
                match = pattern.search(f)
                if match:
                    frame_idx = int(re.findall(r'\d+', f)[-1]) - 1
                    image_files[frame_idx] = os.path.join(img_dir, f)
        return image_files

    def _get_coordinates(self, mask_path):
        mask_data = nib.load(mask_path).get_fdata().astype(int)
        if mask_data.ndim == 3: mask_data = np.max(mask_data, axis=-1)
        props = regionprops(mask_data)
        coords = {}
        for prop in props:
            r, c = prop.centroid
            if self.cfg['TRANSPOSE_COORDS']:
                cx, cy = int(round(r)), int(round(c))
            else:
                cx, cy = int(round(c)), int(round(r))
            coords[str(prop.label)] = (cx, cy)
        return coords

    # ---------------------------------------------------------
    # 生成带层级的拼图
    # ---------------------------------------------------------
    def run_montages(self, tasks_df, mean_df):
        if tasks_df.empty or not self.cfg['DO_MONTAGE']: return
        print("\n--- [STEP 4] 生成高精度异常核验拼图 ---")

        patch_size = self.cfg['PATCH_SIZE']
        plot_h = self.cfg['PLOT_HEIGHT']
        patch_r = patch_size // 2

        for folder_name, group in tasks_df.groupby('Folder'):
            mask_path = os.path.join(self.cfg['MASK_ROOT_DIR'], folder_name, self.cfg['MASK_RELATIVE_PATH'])
            raw_img_folder = os.path.join(self.cfg['IMG_ROOT_DIR'], folder_name)

            if not os.path.exists(mask_path) or not os.path.exists(raw_img_folder): continue

            coords = self._get_coordinates(mask_path)
            file_map = self._index_image_files(raw_img_folder)

            for _, row in group.iterrows():
                cid = str(row['Cell_ID'])
                if cid not in coords: continue

                start_f = row['Start_Frame']
                win_len = row['Window_Len']
                direction = row['Direction']
                rep = row['Repetition']

                save_dir = os.path.join(self.dirs['montages'], folder_name, f"Cell_{cid}", f"Direction_{direction:03d}",
                                        f"Repetition_{rep:02d}")
                os.makedirs(save_dir, exist_ok=True)

                img_strip = np.zeros((patch_size, patch_size * win_len, 3), dtype=np.uint8)
                trace_win = mean_df[row['Global_ID']].values[start_f: start_f + win_len]
                cx, cy = coords[cid]

                for t in range(win_len):
                    curr_f = start_f + t
                    if curr_f not in file_map: continue
                    try:
                        img = io.imread(file_map[curr_f])

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
                        sy, ey = int(cy) - patch_r, int(cy) + patch_r
                        sx, ex = int(cx) - patch_r, int(cx) + patch_r

                        patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                        src_sy, src_ey = max(0, sy), min(h, ey)
                        src_sx, src_ex = max(0, sx), min(w, ex)
                        dst_sy, dst_ey = src_sy - sy, src_sy - sy + (src_ey - src_sy)
                        dst_sx, dst_ex = src_sx - sx, src_sx - sx + (src_ex - src_sx)

                        if dst_ey > dst_sy and dst_ex > dst_sx:
                            patch[dst_sy:dst_ey, dst_sx:dst_ex] = img_rgb[src_sy:src_ey, src_sx:src_ex]

                        x_start = t * patch_size
                        img_strip[:, x_start: x_start + patch_size, :] = patch
                    except:
                        pass

                curve_img = self._draw_sparkline(trace_win, patch_size * win_len, plot_h)
                final_output = np.vstack([img_strip, curve_img])

                fname = f"event_frame_{start_f:04d}_len{win_len}.tif"
                io.imsave(os.path.join(save_dir, fname), final_output, check_contrast=False)

            print(f"    -> 文件夹 {folder_name} 的异常拼图生成完毕。")

    # ---------------------------------------------------------
    # 生成带圈视频
    # ---------------------------------------------------------
    def run_videos(self, tasks_df):
        if tasks_df.empty or not self.cfg['DO_VIDEO']: return
        print("\n--- [STEP 5] 生成异常追踪视频 ---")

        radius = self.cfg['CIRCLE_RADIUS']
        fps = self.cfg['VIDEO_FPS']

        for folder_name, group in tasks_df.groupby('Folder'):
            mask_path = os.path.join(self.cfg['MASK_ROOT_DIR'], folder_name, self.cfg['MASK_RELATIVE_PATH'])
            raw_img_folder = os.path.join(self.cfg['IMG_ROOT_DIR'], folder_name)

            if not os.path.exists(mask_path) or not os.path.exists(raw_img_folder): continue

            coords = self._get_coordinates(mask_path)
            file_map = self._index_image_files(raw_img_folder)

            for _, row in group.iterrows():
                cid = str(row['Cell_ID'])
                if cid not in coords: continue

                start_f = row['Start_Frame']
                win_len = row['Window_Len']
                direction = row['Direction']
                rep = row['Repetition']

                save_dir = os.path.join(self.dirs['videos'], folder_name, f"Cell_{cid}", f"Direction_{direction:03d}",
                                        f"Repetition_{rep:02d}")
                os.makedirs(save_dir, exist_ok=True)

                video_name = f"tracker_frame_{start_f:04d}.avi"
                video_path = os.path.join(save_dir, video_name)

                cy, cx = coords[cid]
                center_point = (int(cx), int(cy))

                if start_f not in file_map: continue
                temp_img = io.imread(file_map[start_f])
                h, w = temp_img.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h), isColor=True)

                for t in range(win_len):
                    curr_f = start_f + t
                    if curr_f not in file_map: break

                    try:
                        img = io.imread(file_map[curr_f])
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
                    except:
                        continue
                video_writer.release()
            print(f"    -> 文件夹 {folder_name} 的异常视频生成完毕。")

    def execute(self):
        print("====== 细胞荧光异常响应筛查系统 启动 ======")
        if not self.cfg['DO_FILTERING']:
            print("当前未开启过滤模式(DO_FILTERING=False)，程序退出。")
            return

        mean_df, area_df = self._load_features_data()
        tasks_df = self.run_filtering(mean_df, area_df)

        self.run_statistics(tasks_df)
        self.run_montages(tasks_df, mean_df)
        self.run_videos(tasks_df)
        print("\n====== 全部任务执行完毕 ======")


if __name__ == "__main__":
    pipeline = AbnormalResponderTracker(CONFIG)
    pipeline.execute()
