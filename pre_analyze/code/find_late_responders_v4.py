import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.measure import regionprops
from skimage import io
import tifffile
import cv2  # 用于生成视频和绘制曲线

# 设置非交互模式
plt.ioff()

"""
该脚本用于处理细胞影像数据，识别在刺激后出现亮度回升的细胞事件。
与之前版本相比，主要改动包括：
1. 移除对 Area 特征的强制筛选，仅基于 Mean 特征进行形态学筛选。即只要窗口内出现了“下去又起来”的形态，即可被识别为响应事件。
2. 调整亮度回升阈值，以适应仅基于 Mean 特征的筛选逻辑。
3. 保持拼图和视频生成逻辑不变，但确保能处理新的筛选结果。

"""

# ================= 配置区域 =================

CONFIG = {
    # --- 1. 路径配置 ---
    'DATA_ROOT_DIR': r"pre_analyze/agonism_radiomics_features",
    'RAW_IMAGES_ROOT_DIR': r"20251102_initial_code/data/data_20250817/agonism",

    'FEATURE_FILES': {
        'Area': 'Final_Area.csv',  # 依然加载，但筛选逻辑中不再强制要求
        'Mean': 'Final_Mean.csv',
    },

    'GLOBAL_MASK_NAME': 'Global_Static_Mask.nii.gz',
    'RAW_IMAGE_EXT': '.jpg',

    # --- 2. 输出路径 ---
    'OUTPUT_DIR': r"pre_analyze/results/jidong/find_late_responders_v4",

    # --- 3. 实验参数 ---
    'STIM_FRAMES': [31, 42, 52, 63, 73, 84, 95, 105, 116, 126, 137, 148, 158, 169, 179],
    'TOTAL_FRAMES': 190,

    # --- 4. 筛选逻辑参数 (形态学筛选) ---
    # [关键参数] 亮度回升阈值
    # 既然移除了Mask限制，这个值必须足够大以排除纯噪音
    'RISE_THRESHOLD': 5.0,

    # --- 5. 裁剪与绘图参数 ---
    'DO_CROPPING': True,
    'PATCH_SIZE': 32,

    # [新] 底部曲线图的高度
    'PLOT_HEIGHT': 160,

    # --- 6. 视频参数 ---
    'DO_VIDEO': True,
    'VIDEO_FPS': 2,
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
        print("\n--- STEP 1: 加载特征数据 ---")
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
    # 步骤 2: 筛选逻辑 (Morphology Only: Down-Up)
    # ---------------------------------------------------------
    def step2_filter_responders(self, area_df, mean_df):
        print("\n--- STEP 2: 执行筛选 (仅形态学: 下去又起来) ---")

        stim_frames_0based = [x - 1 for x in self.cfg['STIM_FRAMES']]
        total_frames = self.cfg['TOTAL_FRAMES']
        rise_thresh = self.cfg['RISE_THRESHOLD']

        tasks = []
        stats_rows = []

        for global_cell_id in mean_df.columns:
            # trace_area = area_df[global_cell_id].values # 这一版不再校验 Area
            trace_mean = mean_df[global_cell_id].values
            folder, original_cell_id = global_cell_id.split('___')

            for i in range(len(stim_frames_0based)):
                start_f = stim_frames_0based[i]

                if i < len(stim_frames_0based) - 1:
                    end_f = stim_frames_0based[i + 1]
                else:
                    end_f = total_frames

                window_mean = trace_mean[start_f: end_f]
                curr_win_len = len(window_mean)

                if curr_win_len < 3: continue

                # --- 核心筛选逻辑: 寻找局部深谷 (Valley) ---
                # 定义：存在索引 k，使得 Mean[k] < Mean[k-1] 且 Mean[k] < Mean[k+1]
                # 且 Mean[-1] - Mean[k] >= 阈值 (确保回升幅度)

                found_valley = False
                valley_idx = -1

                # 遍历窗口内部 (排除首尾)
                for k in range(1, curr_win_len - 1):
                    val = window_mean[k]
                    prev = window_mean[k - 1]
                    next_val = window_mean[k + 1]

                    # 1. 判断形态: Down -> Up
                    if val < prev and val < next_val:
                        # 2. 判断幅度: 回升幅度足够大
                        rise_amplitude = window_mean[-1] - val
                        if rise_amplitude >= rise_thresh:
                            found_valley = True
                            valley_idx = k
                            break

                if found_valley:
                    tasks.append({
                        'Folder': folder,
                        'Cell_ID': original_cell_id,
                        'Start_Frame': start_f,
                        'Window_Len': curr_win_len,
                        'Global_ID': global_cell_id  # 记录这个以便绘图时取数据
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
    # 步骤 2.5: 统计分析
    # ---------------------------------------------------------
    def step2_5_statistics(self, tasks_df):
        if tasks_df.empty: return
        print("\n--- STEP 2.5: 生成统计报告 ---")

        stim_map = {(f - 1): i + 1 for i, f in enumerate(self.cfg['STIM_FRAMES'])}
        tasks_df['Stim_Index'] = tasks_df['Start_Frame'].map(stim_map).fillna(0).astype(int)

        # 1. 刺激频次
        stim_counts = tasks_df['Stim_Index'].value_counts().sort_index().reset_index()
        stim_counts.columns = ['Stim_Index', 'Event_Count']
        all_stims = pd.DataFrame({'Stim_Index': range(1, 16)})
        stim_counts = pd.merge(all_stims, stim_counts, on='Stim_Index', how='left').fillna(0).astype(int)
        stim_counts.to_csv(os.path.join(self.dirs['report'], 'statistics_stim_counts.csv'), index=False)

        # 2. 惯犯细胞
        cell_groups = tasks_df.groupby(['Folder', 'Cell_ID'])
        recurring_data = []
        for (folder, cell), group in cell_groups:
            if len(group) >= 2:
                stim_indices = sorted(group['Stim_Index'].unique())
                recurring_data.append({
                    'Folder': folder,
                    'Cell_ID': cell,
                    'Response_Count': len(group),
                    'Responded_Stim_Indices': ",".join(map(str, stim_indices))
                })

        if recurring_data:
            df = pd.DataFrame(recurring_data).sort_values('Response_Count', ascending=False)
            df.to_csv(os.path.join(self.dirs['report'], 'statistics_recurring_cells.csv'), index=False)
            print(f"  -> 多次响应细胞统计已保存 (n={len(df)})。")

    # ---------------------------------------------------------
    # 辅助函数: 绘制迷你曲线 (Sparkline)
    # ---------------------------------------------------------
    def _draw_sparkline(self, data, width, height):
        """
        使用 OpenCV 绘制简单的趋势线
        data: list or array of values
        width: output image width
        height: output image height
        """
        # 创建黑色背景 (H, W, 3)
        img = np.zeros((height, width, 3), dtype=np.uint8)

        if len(data) < 2: return img

        # 数据归一化到高度范围 (留一点 Padding)
        padding = 10
        d_min, d_max = np.min(data), np.max(data)

        # 防止除零
        if d_max == d_min:
            scale = 0
        else:
            scale = (height - 2 * padding) / (d_max - d_min)

        # 坐标转换函数
        def to_pt(idx, val):
            # X轴：均匀分布在 Patch 中心
            # 第 i 个点对应第 i 个 Patch 的中心
            # PatchWidth = width / len(data)
            patch_w = width / len(data)
            x = int(idx * patch_w + patch_w / 2)

            # Y轴：翻转 (图像坐标原点在左上)
            # val越高，y越小
            y = int(height - padding - (val - d_min) * scale)
            return (x, y)

        # 1. 绘制参考线 (起始帧亮度) - 白色虚线
        # OpenCV 画虚线比较麻烦，这里画一条细灰线代替
        ref_val = data[0]
        y_ref = int(height - padding - (ref_val - d_min) * scale)
        cv2.line(img, (0, y_ref), (width, y_ref), (100, 100, 100), 1)

        # 2. 绘制曲线 - 青色 (Cyan: B=255, G=255, R=0)
        pts = [to_pt(i, v) for i, v in enumerate(data)]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (255, 255, 0), 2, cv2.LINE_AA)

        return img

    # ---------------------------------------------------------
    # 步骤 3: 拼图生成 (含曲线图)
    # ---------------------------------------------------------
    def step3_generate_montages(self, tasks_df, mean_df):
        if tasks_df.empty or not self.cfg['DO_CROPPING']: return
        print("\n--- STEP 3: 生成拼图 (Image + Curve) ---")

        patch_size = self.cfg['PATCH_SIZE']
        plot_h = self.cfg['PLOT_HEIGHT']
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
            # Buffer: key -> [Image_Montage, Curve_Data]
            montage_buffers = {}

            for _, row in group.iterrows():
                cid = str(row['Cell_ID'])
                start_f = row['Start_Frame']
                win_len = row['Window_Len']
                gid = row['Global_ID']

                if cid not in prop_map: continue

                key = (cid, start_f)

                # 初始化图片条
                img_strip = np.zeros((patch_size, patch_size * win_len, 3), dtype=np.uint8)

                # 获取曲线数据
                trace_full = mean_df[gid].values
                trace_win = trace_full[start_f: start_f + win_len]

                montage_buffers[key] = {
                    'img': img_strip,
                    'curve_data': trace_win,
                    'win_len': win_len
                }

                for t in range(win_len):
                    curr_f = start_f + t
                    if curr_f not in frame_requirements: frame_requirements[curr_f] = []
                    frame_requirements[curr_f].append({'key': key, 'offset': t, 'cid': cid})

            if not frame_requirements: continue

            needed_frames = sorted(frame_requirements.keys())

            # 填充图像部分
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

            # 合成与保存
            for (cid, start_f), item in montage_buffers.items():
                img_strip = item['img']
                curve_data = item['curve_data']
                win_len = item['win_len']

                # 绘制下方的曲线图
                total_w = patch_size * win_len
                curve_img = self._draw_sparkline(curve_data, total_w, plot_h)

                # 垂直拼接
                final_output = np.vstack([img_strip, curve_img])

                actual_filename = file_map.get(start_f, f"Unknown_Idx{start_f}")
                t_num = get_t_num(actual_filename)
                fname = f"{folder_name}_Cell{cid}_t{t_num:03d}_Len{win_len}.tif"

                io.imsave(os.path.join(save_folder_dir, fname), final_output, check_contrast=False)

            print(f"    -> 文件夹 {folder_name} 拼图生成完毕 ({len(montage_buffers)} 张)。")

    # ---------------------------------------------------------
    # 步骤 4: 视频生成 (保持不变)
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
                        continue
                video_writer.release()
            print(f"    -> 视频生成完毕。")

    def run(self):
        area_df, mean_df = self.step1_load_data()
        tasks_df = self.step2_filter_responders(area_df, mean_df)
        self.step2_5_statistics(tasks_df)
        self.step3_generate_montages(tasks_df, mean_df)  # [修改] 传入 mean_df
        self.step4_generate_videos(tasks_df)
        print("\n--- 全部处理完成 ---")


if __name__ == "__main__":
    pipeline = MaskBasedResponderPipeline(CONFIG)
    pipeline.run()