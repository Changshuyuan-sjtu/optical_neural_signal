import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

"""
本脚本用于分析细胞强度数据，执行以下操作:（多文件夹）
1. 从指定根目录的所有子文件夹中加载 'cell_intensities.csv' 文件。
2. 合并所有数据，确保每个细胞ID唯一（通过添加子文件夹前缀）。
3. 基于指定的刺激帧提取窗口数据。
4. 对提取的窗口数据进行基线校正。
5. 使用 K-Means 对基线校正后的数据进行聚类。
6. 可视化每个聚类的平均响应曲线，支持多窗口长视图。
7. 将聚类结果保存为 CSV 文件 (W1 和 Long)。
"""


def analyze_cell_clusters_multi_window_view(
        raw_dataframe,
        stim_frames_1idx,
        window_size=11,
        n_clusters=4,
        num_windows_to_view=3,
        random_state=42,
        baseline_method='min',
        output_csv_path_w1='clusters_w1_data.csv',  # W1 数据保存路径
        output_csv_path_long='clusters_long_data.csv',  # *** 新增：长窗口数据保存路径 ***
        output_png_path_w1='clusters_avg_w1.png',
        output_png_path_long='clusters_avg_long_view.png'
):
    print(f"开始分析 (长视图): {n_clusters} 聚类, 聚类窗口 {window_size} 帧...")
    print(f"长视图将显示 {num_windows_to_view} 个窗口 (共 {window_size * num_windows_to_view} 帧)。")

    # 1. 定义窗口参数
    stim_frames_0idx = [f - 1 for f in stim_frames_1idx]
    long_window_size = window_size * num_windows_to_view

    # 2. 提取特征窗口 (W1 和 W1+...+Wn)
    all_windows_list_w1 = []
    all_windows_list_long = []
    all_windows_index = []
    cell_ids = raw_dataframe.columns

    for cell in cell_ids:
        for start_frame in stim_frames_0idx:
            end_frame_w1 = start_frame + window_size - 1
            end_frame_long = start_frame + long_window_size - 1

            # 仅当 *长窗口* 也在数据范围内时，才使用此样本
            if end_frame_long <= raw_dataframe.index.max():
                window_data_w1 = raw_dataframe.loc[start_frame:end_frame_w1, cell].values
                window_data_long = raw_dataframe.loc[start_frame:end_frame_long, cell].values

                if len(window_data_w1) == window_size and len(window_data_long) == long_window_size:
                    all_windows_list_w1.append(window_data_w1)
                    all_windows_list_long.append(window_data_long)
                    all_windows_index.append((cell, start_frame))

    if not all_windows_index:
        print(f"错误：未能提取任何有效的 *长窗口* (大小 {long_window_size})。")
        print("请检查刺激帧是否离数据末尾太近，或减小 'num_windows_to_view'。")
        return None, None, None, None

    # --- 创建 DataFrame ---
    feature_df_w1 = pd.DataFrame(
        all_windows_list_w1,
        columns=[f'T_{i}' for i in range(window_size)],
        index=pd.MultiIndex.from_tuples(all_windows_index, names=['cell_id', 'stim_frame_0idx'])
    )

    feature_df_long = pd.DataFrame(
        all_windows_list_long,
        columns=[f'T_{i}' for i in range(long_window_size)],
        index=pd.MultiIndex.from_tuples(all_windows_index, names=['cell_id', 'stim_frame_0idx'])
    )

    print(f"总共提取了 {len(feature_df_w1)} 个有效样本 (用于 W1 和长视图)。")

    # 3. 基线校正 (基于 W1)
    if baseline_method == 'min':
        window_baseline = feature_df_w1.min(axis=1)
        print("使用 'W1 窗口最小值' 作为基线。")
    elif baseline_method == 'global_min':
        global_min_per_cell = {}
        for cell in cell_ids:
            global_min_per_cell[cell] = raw_dataframe[cell].min()
        window_baseline = feature_df_w1.index.get_level_values('cell_id').map(global_min_per_cell)
        print("使用 'raw_dataframe全局最小值' 作为聚类基线。")
    else:
        window_baseline = feature_df_w1['T_0']
        print("使用 'W1 T_0' 作为基线。")

    baseline_corrected_df_w1 = feature_df_w1.subtract(window_baseline, axis=0)
    baseline_corrected_df_long = feature_df_long.subtract(window_baseline, axis=0)

    # 4. 聚类 (仅基于 W1)
    scaler = StandardScaler()
    scaled_data_w1 = scaler.fit_transform(baseline_corrected_df_w1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data_w1)

    print("\n每个聚类中的样本数量:")
    print(pd.Series(cluster_labels).value_counts().sort_index())

    # --- 5. 保存 CSV (这是主要修改部分) ---

    # 保存 W1
    output_df_w1 = feature_df_w1.copy()
    output_df_w1['cluster'] = cluster_labels
    output_df_w1.to_csv(output_csv_path_w1)
    print(f"\n已保存 W1 标准窗口数据: {output_csv_path_w1}")

    # 保存 Long (为了后续对比画图)
    output_df_long = feature_df_long.copy()
    output_df_long['cluster'] = cluster_labels
    output_df_long.to_csv(output_csv_path_long)
    print(f"已保存 Long 长窗口数据: {output_csv_path_long}")

    # 6. 可视化 (完全保留原始画图逻辑)
    colors = plt.cm.get_cmap('tab10', n_clusters)

    # --- 图 1: 标准 W1 视图 ---
    analysis_df_w1 = baseline_corrected_df_w1.copy()
    analysis_df_w1['cluster'] = cluster_labels
    cluster_averages_w1 = analysis_df_w1.groupby('cluster').mean()

    plt.figure(figsize=(12, 8))
    for i in range(n_clusters):
        cluster_data = cluster_averages_w1.loc[i]
        cluster_size = pd.Series(cluster_labels).value_counts().get(i, 0)
        plt.plot(
            cluster_data.index,
            cluster_data.values,
            label=f'Cluster {i} (n={cluster_size})',
            color=colors(i),
            linewidth=2
        )

    plt.axvline(x='T_0', color='red', linestyle='--', label='Stimulus Onset (T_0)')
    stim_end_col_w1 = f'T_{min(2, window_size - 1)}'
    plt.axvspan('T_0', stim_end_col_w1, color='red', alpha=0.1, label='Stimulus Duration')

    plt.title(f'Cluster Average Response (W1: T_0 to T_{window_size - 1})', fontsize=16)
    plt.xlabel('Time Relative to Stimulus Onset', fontsize=12)
    plt.ylabel(f'Intensity (relative to W1 {baseline_method})', fontsize=12)
    plt.legend(title='Cluster')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_png_path_w1)
    print(f"已将标准窗口 (W1) 响应图保存到: {output_png_path_w1}")
    plt.close()  # 加上 close 防止不显示

    # --- 图 2: 长视图 (W1...Wn) ---
    analysis_df_long = baseline_corrected_df_long.copy()
    analysis_df_long['cluster'] = cluster_labels
    cluster_averages_long = analysis_df_long.groupby('cluster').mean()

    long_view_width = 10 + (num_windows_to_view - 1) * 4
    plt.figure(figsize=(long_view_width, 8))

    for i in range(n_clusters):
        cluster_data = cluster_averages_long.loc[i]
        cluster_size = pd.Series(cluster_labels).value_counts().get(i, 0)
        plt.plot(
            cluster_data.index,
            cluster_data.values,
            label=f'Cluster {i} (n={cluster_size})',
            color=colors(i),
            linewidth=2
        )

    plt.axvline(x='T_0', color='red', linestyle='--', label='Stimulus Onset (T_0)')
    stim_end_col_long = f'T_{min(2, long_window_size - 1)}'
    plt.axvspan('T_0', stim_end_col_long, color='red', alpha=0.1, label='Stimulus Duration')

    # 标记 W1, W2... 的结束
    for i in range(1, num_windows_to_view):
        end_frame_label = f'T_{window_size * i - 1}'
        if end_frame_label in cluster_averages_long.columns:
            plt.axvline(x=end_frame_label, color='blue', linestyle=':',
                        label=f'End of W{i} (T_{window_size * i - 1})')

    plt.title(f'Cluster Average Response (Long View: {num_windows_to_view} Windows)', fontsize=16)
    plt.xlabel('Time Relative to Stimulus Onset', fontsize=12)
    plt.ylabel(f'Intensity (relative to W1 {baseline_method})', fontsize=12)

    # 处理图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Cluster', loc='upper right')

    plt.grid(True, linestyle='--', alpha=0.6)

    # 优化 X 轴
    long_tick_labels = cluster_averages_long.columns
    tick_interval = max(1, long_window_size // 15)
    ticks_to_show_indices = list(range(0, long_window_size, tick_interval))
    if long_window_size - 1 not in ticks_to_show_indices:
        ticks_to_show_indices.append(long_window_size - 1)
    ticks_to_show = [long_tick_labels[i] for i in ticks_to_show_indices]

    plt.xticks(ticks=ticks_to_show, rotation=45)
    plt.tight_layout()
    plt.savefig(output_png_path_long)
    print(f"已将长窗口 ({num_windows_to_view} windows) 响应图保存到: {output_png_path_long}")
    plt.close()

    return output_csv_path_w1, output_csv_path_long


if __name__ == '__main__':

    # ================= 配置区域 =================

    # 1. 请在这里修改路径 (分别运行两次，一次激动，一次拮抗)
    # ROOT_DIR = 'cell_segmentation/Intermediate experimental results/activity_map_jidong'
    ROOT_DIR = 'Fluorescent cell segmentation/Intermediate experimental results/activity_map_jidong'

    # 2. 设置输出前缀
    OUTPUT_PREFIX = 'jidong'
    # OUTPUT_PREFIX = 'jiekang'

    # ===========================================

    TARGET_CSV_NAME = 'cell_intensities.csv'
    STIMULI = [31, 42, 52, 63, 73, 84, 95, 105, 116, 126, 137, 148, 158, 169, 179]

    # ... (数据加载部分保持不变) ...
    all_dataframes = []
    print(f"开始扫描 {ROOT_DIR} ...")
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        if TARGET_CSV_NAME in filenames:
            if dirpath == ROOT_DIR: continue
            csv_path = os.path.join(dirpath, TARGET_CSV_NAME)
            subfolder_name = os.path.basename(dirpath)
            try:
                df = pd.read_csv(csv_path, index_col=0)
                df.columns = [f"{subfolder_name}_cell_{col}" for col in df.columns]
                all_dataframes.append(df)
            except Exception as e:
                print(f"加载失败: {e}")

    if all_dataframes:
        df_raw_combined = pd.concat(all_dataframes, axis=1)

        # 参数
        N_CLUSTERS = 4
        CLUSTER_WINDOW_SIZE = 11
        N_WINDOWS_TO_VIEW = 3
        BASELINE_METHOD = 'global_min'

        # 确保输出目录存在
        os.makedirs(f'pre_analyze/results/{OUTPUT_PREFIX}/average', exist_ok=True)

        # 定义输出路径
        out_csv_w1 = f'pre_analyze/results/{OUTPUT_PREFIX}/average/clusters_w1.csv'

        # *** 重点：这个文件就是你要发给我的 ***
        out_csv_long = f'pre_analyze/results/{OUTPUT_PREFIX}/average/clusters_long.csv'

        out_png_w1 = f'pre_analyze/results/{OUTPUT_PREFIX}/average/w1_view.png'
        out_png_long = f'pre_analyze/results/{OUTPUT_PREFIX}/average/long_view.png'

        analyze_cell_clusters_multi_window_view(
            raw_dataframe=df_raw_combined,
            stim_frames_1idx=STIMULI,
            window_size=CLUSTER_WINDOW_SIZE,
            n_clusters=N_CLUSTERS,
            num_windows_to_view=N_WINDOWS_TO_VIEW,
            baseline_method=BASELINE_METHOD,
            output_csv_path_w1=out_csv_w1,
            output_csv_path_long=out_csv_long,  # 传入新参数
            output_png_path_w1=out_png_w1,
            output_png_path_long=out_png_long
        )