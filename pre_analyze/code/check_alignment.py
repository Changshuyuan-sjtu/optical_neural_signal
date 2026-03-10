import os
import re
import numpy as np
import nibabel as nib
from skimage.measure import regionprops
from skimage import io
import matplotlib.pyplot as plt

# ================= 配置区域 =================

# 1. 原始 JPG 文件夹
RAW_IMAGE_DIR = r"20251102_initial_code/data/data_20250817/agonism/Image 2"
FILE_EXTENSION = '.jpg'

# 2. 全局静态掩码文件
GLOBAL_STATIC_MASK = r"20251102_initial_code/data/data_20250817_tracked/agonism_mask/Image 2/global_static_mask/Global_Static_Mask.nii.gz"


# ===========================================

def check_alignment():
    # 1. 加载第一张图片
    files = sorted(
        [f for f in os.listdir(RAW_IMAGE_DIR) if f.endswith(FILE_EXTENSION) or f.endswith(FILE_EXTENSION.upper())])
    if not files:
        print("未找到图像文件！")
        return

    img_path = os.path.join(RAW_IMAGE_DIR, files[0])
    print(f"正在读取参考图像: {os.path.basename(img_path)}")
    img = io.imread(img_path)
    h, w = img.shape[:2]

    # 2. 加载 Mask 并计算原始质心
    print(f"正在读取掩码: {GLOBAL_STATIC_MASK}")
    mask_img = nib.load(GLOBAL_STATIC_MASK)
    mask_data = mask_img.get_fdata().astype(int)

    if mask_data.ndim == 3:
        mask_2d = np.max(mask_data, axis=0)
    else:
        mask_2d = mask_data

    props = regionprops(mask_2d)
    # 原始坐标列表 [(row, col), ...] 即 [(y, x)]
    raw_centroids = [p.centroid for p in props]

    # 3. 定义 4 种变换模式
    # r = row (y-axis in numpy), c = col (x-axis in numpy)
    transformations = {
        "1. Standard (No Change)\nx=col, y=row":
            lambda r, c: (c, r),

        "2. Transpose (Swap X/Y)\nx=row, y=col":
            lambda r, c: (r, c),

        "3. Flip Vertical (上下颠倒)\nx=col, y=H-row":
            lambda r, c: (c, h - r),

        "4. Transpose + Flip (转置+颠倒)\nx=row, y=W-col":
            lambda r, c: (r, w - c)
    }

    # 4. 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.ravel()

    print("正在生成 4 种变换的对比图...")

    for i, (title, func) in enumerate(transformations.items()):
        ax = axes[i]
        # 显示背景图
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)

        # 计算当前变换下的坐标
        xs = []
        ys = []
        for (r, c) in raw_centroids:
            nx, ny = func(r, c)
            xs.append(nx)
            ys.append(ny)

        # 画红叉
        ax.scatter(xs, ys, c='red', s=40, marker='x', linewidths=1.5)
        ax.set_title(title, fontsize=14, color='blue', fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    save_path = "pre_analyze/alignment_diagnostic_4in1.png"
    plt.savefig(save_path)
    print(f"\n诊断图已保存至: {os.path.abspath(save_path)}")
    print("请打开图片，查看哪一个小图里的红叉是对得最准的。")


if __name__ == "__main__":
    check_alignment()