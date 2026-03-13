import matplotlib.pyplot as plt
import numpy as np
import cv2
import os, re
import glob
import warnings
import tifffile
from skimage import io, measure, morphology, segmentation, feature, color
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Union
from skimage.morphology import disk
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from scipy import ndimage as ndi


#把cellpose-GUI标注的.npy格式标签转换为.tif格式的binary mask
def transform_npy_to_tif(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('_seg.npy'):
            npy_path = os.path.join(input_dir, filename)

            data = np.load(npy_path, allow_pickle=True).item()

            if "masks" not in data:
                print(f"⚠️ 跳过文件（无 masks 字段）: {filename}")
                continue

            masks = data["masks"]

            binary_mask = np.where(masks > 0, 255, 0).astype(np.uint8)

            base_name = os.path.splitext(filename)[0].replace("_seg", "")
            save_path = os.path.join(output_dir, f"{base_name}_mask.tif")

            # 保存为tif格式
            Image.fromarray(binary_mask).save(save_path)
            print(f"✅ 已保存: {save_path}")
    print("全部处理完成！")



def plot_RGB_channels():
    # 读取图像（保持彩色）
    img = cv2.imread("data_20250817_only_RGB/jidong/Image 8/Image 8_t033.jpg", cv2.IMREAD_COLOR)  # OpenCV 默认是 BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB顺序

    # 分离三个通道
    R, G, B = cv2.split(img_rgb)

    # 可视化
    plt.figure(figsize=(12, 6))

    # 原图
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("Original (RGB)")
    plt.axis("off")

    # R通道
    plt.subplot(1, 4, 2)
    plt.imshow(R, cmap="Reds")
    plt.title("Red Channel")
    plt.axis("off")

    # G通道
    plt.subplot(1, 4, 3)
    plt.imshow(G, cmap="Greens")
    plt.title("Green Channel")
    plt.axis("off")

    # B通道
    plt.subplot(1, 4, 4)
    plt.imshow(B, cmap="Blues")
    plt.title("Blue Channel")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def morphological_opening(img, kernel_size=(3, 3), kernel_shape='rect', iterations=1, borderType=cv2.BORDER_CONSTANT):
    """
    对图像执行形态学开运算（erosion -> dilation）。

    参数:
        img (np.ndarray): 输入图像。可为灰度图或二值图（0/255）。若为彩色图像，会先转换为灰度再处理。
        kernel_size (tuple or int): 结构元素大小，例如 (3,3) 或 3（等价于 (3,3)）。
        kernel_shape (str): 结构元素形状，'rect'（矩形）、'ellipse'（椭圆）、'cross'（十字）。
        iterations (int): 腐蚀和膨胀的迭代次数（整数>=1）。
        borderType: OpenCV 边界处理方式，默认 cv2.BORDER_CONSTANT。

    返回:
        out (np.ndarray): 开运算后的图像（与输入尺寸一致）。
    """
    # 参数规范化
    if isinstance(kernel_size, int):
        ksize = (kernel_size, kernel_size)
    else:
        ksize = tuple(kernel_size)

    shape_map = {
        'rect': cv2.MORPH_RECT,
        'ellipse': cv2.MORPH_ELLIPSE,
        'cross': cv2.MORPH_CROSS
    }
    kshape = shape_map.get(kernel_shape, cv2.MORPH_RECT)

    # 如果是彩色图像，先转换为灰度（开运算通常用于灰度/二值）
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 确保图像为 uint8
    if gray.dtype != np.uint8:
        # 归一化到 0-255 再转 uint8（防止 float 输入）
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 创建结构元素
    kernel = cv2.getStructuringElement(kshape, ksize)

    # 直接使用 OpenCV 的 morphologyEx 更稳妥（内部执行 erosion + dilation）
    out = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations, borderType=borderType)

    return out


def filter_components_by_area_batch(input_folder, output_folder, thresh_ratio=0.7, min_area=50, max_area=4000):
    """
    批量处理掩膜图像，根据面积条件过滤噪声
    - input_folder: 输入文件夹路径
    - output_folder: 输出文件夹路径
    - thresh_ratio: 阈值比例
    - min_area: 最小面积
    - max_area: 最大面积
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 支持的图像格式
    supported_formats = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}

    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder)
                   if os.path.isfile(os.path.join(input_folder, f)) and
                   os.path.splitext(f)[1].lower() in supported_formats]

    if not image_files:
        print(f"在文件夹 {input_folder} 中没有找到支持的图像文件")
        return

    print(f"找到 {len(image_files)} 个图像文件，开始处理...")

    processed_count = 0
    for filename in image_files:
        try:
            # 读取图像
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"无法读取图像: {filename}")
                continue

            # 归一化
            norm_map = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # 二值化
            _, binary = cv2.threshold(
                norm_map,
                int(thresh_ratio * norm_map.max()),
                255,
                cv2.THRESH_BINARY
            )

            # 连通域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

            # 创建筛选后的 binary
            filtered_binary = np.zeros_like(binary)
            h, w = binary.shape

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                x, y, bw, bh, _ = stats[i]

                touches_border = (x == 0 or y == 0 or x + bw == w or y + bh == h)

                if (min_area <= area <= max_area):
                    filtered_binary[labels == i] = 255

            # 保存结果，保持原文件名
            output_path = os.path.join(output_folder, filename)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(output_path, filtered_binary)

            processed_count += 1
            #print(f"已处理: {filename}")

        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")

    print(f"处理完成! 成功处理 {processed_count}/{len(image_files)} 个文件")


def outline_extraction(original_dir, mask_dir, output_dir, save_contours=False):
    """
    批量处理函数：从原图文件夹和mask文件夹中提取对应图像的轮廓，
    叠加到原始图像上，并保存到输出文件夹。按文件顺序对应位置匹配。

    参数：
    - original_dir: 原始图像文件夹路径
    - mask_dir: 掩码图像文件夹路径
    - output_dir: 输出文件夹路径（会创建如果不存在）
    - save_contours: 是否额外保存独立的轮廓图像（默认False）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取原图和掩码文件列表，按顺序处理
    original_files = [f for f in os.listdir(original_dir) if
                      f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    # 确保文件数量匹配
    if len(original_files) != len(mask_files):
        print(
            f"Warning: Number of original files ({len(original_files)}) does not match mask files ({len(mask_files)}).")
        min_length = min(len(original_files), len(mask_files))
        original_files = original_files[:min_length]
        mask_files = mask_files[:min_length]

    # 按顺序处理对应位置的文件
    for orig_file, mask_file in zip(original_files, mask_files):
        orig_path = os.path.join(original_dir, orig_file)
        mask_path = os.path.join(mask_dir, mask_file)

        try:
            original_image = io.imread(orig_path)
            mask_image = io.imread(mask_path, as_gray=True)
        except Exception as e:
            print(f"Error loading {orig_file} or its mask: {e}. Skipping.")
            continue

        # 二值化掩码
        mask_binary = mask_image > 0

        # 标签化连通组件
        labeled_mask, num_labels = ndi.label(mask_binary)

        # 创建叠加图像副本
        overlay_image = original_image.copy()

        # 创建空白轮廓图像（如果需要）
        if save_contours:
            contour_image = np.zeros_like(original_image) if len(original_image.shape) == 3 else np.zeros_like(
                mask_image)

        # 为每个mask提取并绘制轮廓
        for label in range(1, num_labels + 1):
            single_mask = (labeled_mask == label)
            contours = measure.find_contours(single_mask, 0.5)

            for contour in contours:
                # contour是(y, x)坐标数组
                y, x = contour[:, 0].astype(int), contour[:, 1].astype(int)
                # 确保坐标在界内
                valid = (y >= 0) & (y < overlay_image.shape[0]) & (x >= 0) & (x < overlay_image.shape[1])
                y, x = y[valid], x[valid]
                # 叠加到原始图像和轮廓图像
                if len(overlay_image.shape) == 3:  # RGB
                    overlay_image[y, x] = [255, 255, 255]  # 白色轮廓
                    if save_contours:
                        contour_image[y, x] = [255, 255, 255]  # 白色轮廓
                else:  # 灰度
                    overlay_image[y, x] = 255  # 白色轮廓
                    if save_contours:
                        contour_image[y, x] = 255

        # 保存叠加图像
        base_name, ext = os.path.splitext(orig_file)
        overlay_output_path = os.path.join(output_dir, f"{base_name}_overlay{ext}")
        io.imsave(overlay_output_path, overlay_image)
        print(f"Saved overlay for {orig_file} to {overlay_output_path}")

        # 如果需要，保存轮廓图像
        if save_contours:
            contour_output_path = os.path.join(output_dir, f"{base_name}_contours.png")
            io.imsave(contour_output_path, contour_image)
            print(f"Saved contours for {orig_file} to {contour_output_path}")


def create_video_from_images(folder_path, output_video_path='output.mp4', interval_seconds=2, fps=30):
    """
    从文件夹中读取图片，按顺序生成视频。
    :param folder_path: 图片文件夹路径
    :param output_video_path: 输出视频文件路径（默认 output.mp4）
    :param interval_seconds: 每张图片显示的固定时间间隔（秒）
    :param fps: 视频帧率（默认30）
    """
    print("input_folder_path:", folder_path)
    # 获取文件夹中所有图片文件，并按文件名排序
    images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg','.tif', '.tiff'))]
    images.sort()  # 按文件名排序（假设文件名已有序）

    if not images:
        print("文件夹中没有找到图片文件。")
        return

    # 读取第一张图片，获取尺寸
    first_image_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        print(f"无法读取图片: {first_image_path}")
        return
    height, width, layers = frame.shape

    # 创建视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4编码
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 遍历每张图片
    for image in images:
        image_path = os.path.join(folder_path, image)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"跳过无效图片: {image_path}")
            continue

        # 如果尺寸不一致，调整大小（可选，根据需要）
        if img.shape != (height, width, layers):
            img = cv2.resize(img, (width, height))

        # 为每张图片添加重复帧，实现固定间隔
        num_frames = int(interval_seconds * fps)
        for _ in range(num_frames):
            video.write(img)

    # 释放视频编写器
    video.release()
    print(f"视频已生成: {output_video_path}")

# 使用示例
# create_video_from_images('path/to/your/image/folder', 'my_video.mp4', interval_seconds=1, fps=30)

#把掩膜图叠加到所有图像上
def process_images_with_mask(input_folder, mask_path, output_folder='output'):
    """
    Process all images in the input folder, overlaying contours from the mask image.

    Parameters:
    input_folder (str): Path to folder containing input images
    mask_path (str): Path to the mask image
    output_folder (str): Path to save output images (default: 'output')
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load mask image
    mask_image = io.imread(mask_path, as_gray=True)
    mask_binary = mask_image > 0
    labeled_mask, num_labels = ndi.label(mask_binary)

    # Get list of image files in input folder
    image_files = [f for f in os.listdir(input_folder) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    for image_file in image_files:
        # Load original image
        image_path = os.path.join(input_folder, image_file)
        original_image = io.imread(image_path)

        # Create a copy for overlay
        overlay_image = original_image.copy()

        # Create blank image for contours
        #contour_image = np.zeros_like(original_image) if len(original_image.shape) == 3 else np.zeros_like(mask_image)

        # Draw contours for each labeled region
        for label in range(1, num_labels + 1):
            single_mask = (labeled_mask == label)
            contours = measure.find_contours(single_mask, 0.5)

            for contour in contours:
                y, x = contour[:, 0].astype(int), contour[:, 1].astype(int)
                valid = (y >= 0) & (y < overlay_image.shape[0]) & (x >= 0) & (x < overlay_image.shape[1])
                y, x = y[valid], x[valid]

                # Draw contours (red for RGB, white for grayscale)
                if len(overlay_image.shape) == 3:  # RGB
                    overlay_image[y, x] = [255, 255, 255]  # Red
                    #contour_image[y, x] = [0, 0, 255]  # Red
                else:  # Grayscale
                    overlay_image[y, x] = 255  # White
                    #contour_image[y, x] = 255  # White

        # Save results
        base_name = os.path.splitext(image_file)[0]
        io.imsave(os.path.join(output_folder, f'{base_name}_overlay.png'), overlay_image)
        #io.imsave(os.path.join(output_folder, f'{base_name}_contours.png'), contour_image)



"""
作用：填洞 + 开运算（去毛刺） + 闭运算（平滑边缘）。

用法示例:
    from clean_masks_tif import clean_masks_folder
    clean_masks_folder("in_tifs", "out_tifs", open_radius=2, close_radius=3)
"""

# 仅处理 tif 文件
_VALID_EXT = {".tif", ".tiff"}

def clean_masks_folder(input_dir: str,
                       output_dir: str,
                       open_radius: int = 2,
                       close_radius: int = 3,
                       thresh: float = 0.5,
                       overwrite: bool = False,
                       verbose: bool = True) -> List[Dict]:
    """
    批量清洗 .tif/.tiff 二值 mask，并按原名保存到 output_dir。

    参数:
      input_dir: 原始 mask 文件夹（.tif / .tiff）
      output_dir: 输出文件夹（会自动创建）
      open_radius: 开运算圆盘半径（像素），用于去除细小毛刺
      close_radius: 闭运算圆盘半径（像素），用于填补边缘小凹陷
      thresh: 若图像不是 0/255，则用该阈值 (0..1) 做二值化（向后兼容）
      overwrite: 若输出已存在是否覆盖
      verbose: 是否打印进度

    返回:
      results: List[dict]，每项包含 { "name", "saved_path" or None, "skipped":bool, "reason":str or None }
    """
    inp = Path(input_dir)
    outp = Path(output_dir)
    if not inp.exists() or not inp.is_dir():
        raise ValueError(f"input_dir 不存在或不是文件夹: {input_dir}")
    outp.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(inp.iterdir()) if p.is_file() and p.suffix.lower() in _VALID_EXT]
    results = []
    if verbose:
        print(f"发现 {len(files)} 个 tif 文件，开始处理...")

    for idx, f in enumerate(files, 1):
        info = {"name": f.name, "saved_path": None, "skipped": False, "reason": None}
        out_file = outp / f.name
        if out_file.exists() and not overwrite:
            info["skipped"] = True
            info["reason"] = "exists"
            results.append(info)
            if verbose:
                print(f"[{idx}/{len(files)}] 跳过（已存在）：{f.name}")
            continue

        try:
            # 读取为灰度（skimage 会处理多种 tiff）
            img = io.imread(str(f), as_gray=True)

            # 识别 0/255 二值图（优先）
            if img.dtype == np.uint8 and img.max() == 255:
                mask_bool = (img == 255)
            else:
                # 兼容 float/uint16 等：归一化到 [0,1] 后阈值
                img_f = img.astype(np.float32)
                if img_f.max() > 1.0:
                    img_f = img_f / float(img_f.max())
                mask_bool = img_f > float(thresh)

            # 1) 填充内部空洞
            filled = ndi.binary_fill_holes(mask_bool)

            # 2) 开运算：去除细小毛刺
            if open_radius > 0:
                cleaned = morphology.binary_opening(filled, footprint=disk(int(open_radius)))
            else:
                cleaned = filled

            # 3) 闭运算：填补边缘小凹陷
            if close_radius > 0:
                cleaned = morphology.binary_closing(cleaned, footprint=disk(int(close_radius)))

            # 保存为单通道 uint8 (0/255)
            out_uint8 = (cleaned.astype(np.uint8) * 255)
            cv2.imwrite(str(out_file), out_uint8)

            info["saved_path"] = str(out_file)
            if verbose:
                print(f"[{idx}/{len(files)}] 处理并保存: {f.name}")

        except Exception as e:
            info["skipped"] = True
            info["reason"] = f"error:{repr(e)}"
            if verbose:
                print(f"[{idx}/{len(files)}] 处理失败: {f.name} => {e}")

        results.append(info)

    if verbose:
        print("全部处理完成。")
    return results

def instances_to_semantic(input_dir, output_dir):
    """
    将 input_dir 中的 instance masks (.tif/.tiff) 转为 semantic masks。
    背景 = 0，细胞区域 >0 = 255。
    文件名保持不变，输出到 output_dir。
    """
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "*.tif"))) + \
            sorted(glob.glob(os.path.join(input_dir, "*.tiff")))

    for f in files:
        img = tifffile.imread(f)
        semantic = (img > 0).astype(np.uint8) * 255  # 所有细胞设为255
        out_path = os.path.join(output_dir, os.path.basename(f))
        tifffile.imwrite(out_path, semantic)


def semantic_to_instances(
    src_folder: Union[str, Path],
    dst_folder: Union[str, Path],
    min_area: int = 10,
    connectivity: int = 1,            # 1 -> 4连通, 2 -> 8连通
) -> List[Path]:
    """
    把二值语义 mask (0 背景, 255 细胞) 批量转成实例标签（每张图从1..K编号）。
    参数:
      src_folder: 二值 mask 文件夹 (.tif)
      dst_folder: 保存实例标签的目标文件夹
      min_area: 小区域过滤阈值（像素）
      connectivity: 连通性参数，2 表示 8 连通
    返回:
      保存的文件路径列表（Path）
    """
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)
    saved = []

    for p in sorted(src_folder.glob('*.tif')):
        img = tifffile.imread(str(p))
        bw = (img > 127)  # 确保二值布尔数组


        labels = measure.label(bw, connectivity=connectivity)
        # 面积过滤
        props = regionprops(labels)
        for prop in props:
            if prop.area < min_area:
                labels[labels == prop.label] = 0
        # 重新编号连续
        labels = measure.label(labels > 0, connectivity=connectivity)

        out_path = dst_folder / p.name
        # 保存为 uint16，兼容大量实例
        tifffile.imwrite(str(out_path), labels.astype(np.uint16))
        saved.append(out_path)

    return saved

def labels_to_color_batch(
    label_folder: Union[str, Path],
    out_folder: Union[str, Path],
    seed: Optional[int] = 42,
    bg_color: tuple = (0,0,0),
) -> List[Path]:
    """
    把实例标签（uint16）批量映射为随机颜色 RGB 图并保存为 tif/png。
    参数:
      label_folder: 保存 label 图的文件夹（.tif）
      out_folder: 输出颜色图文件夹
      seed: 随机种子（若为 None 则每次随机）
      bg_color: 背景颜色 (R,G,B)
    返回:
      保存的文件路径列表
    """
    label_folder = Path(label_folder)
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    saved = []

    for p in sorted(label_folder.glob('*.tif')):
        lab = tifffile.imread(str(p)).astype(np.int32)
        maxlabel = int(lab.max())
        # 生成颜色表
        colors = rng.randint(0, 256, size=(maxlabel+1, 3), dtype=np.uint8)
        colors[0] = np.array(bg_color, dtype=np.uint8)
        rgb = colors[lab]  # shape (H,W,3)
        out_p = out_folder / (p.stem + '_color.tif')
        tifffile.imwrite(str(out_p), rgb)
        saved.append(out_p)

    return saved


def overlay_instance_labels(rgb_folder, raw_folder, output_folder, alpha=0.3):
    """
    将彩色实例标签图（RGB）与原始图按顺序叠加并保存。
    要求两个文件夹中的图像数量一致，按文件名排序后逐一对应。

    参数：
        rgb_folder (str): RGB实例标签图文件夹路径
        raw_folder (str): 原始图文件夹路径
        output_folder (str): 输出叠加图保存路径
        alpha (float): 叠加权重（标签透明度，默认0.5）
    """
    rgb_folder = Path(rgb_folder)
    raw_folder = Path(raw_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(rgb_folder.glob("*.tif"))
    raw_files = sorted(raw_folder.glob("*.jpg"))

    if len(rgb_files) != len(raw_files):
        raise ValueError(f"文件数量不匹配：RGB标签 {len(rgb_files)} 张，原图 {len(raw_files)} 张")

    for i, (rgb_path, raw_path) in enumerate(zip(rgb_files, raw_files)):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)

        if rgb.shape != raw.shape:
            rgb = cv2.resize(rgb, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = cv2.addWeighted(raw, 1 - alpha, rgb, alpha, 0)

        out_name = raw_path.stem + "_overlay.tif"
        cv2.imwrite(str(output_folder / out_name), overlay)

        print(f"[{i + 1}/{len(raw_files)}] 已保存叠加图：{out_name}")

    print("✅ 全部叠加完成！")

# 把文件夹中的图像按顺序重命名为 Image {image_index}_t{编号}_mask.tif
def rename_sequential(folder_path, image_index=5):
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".tif")])

    for i, fname in enumerate(files, start=1):
        old_path = os.path.join(folder_path, fname)

        new_name = f"Image {image_index}_t{str(i).zfill(3)}_mask.tif"
        new_path = os.path.join(folder_path, new_name)

        print(f"{fname}  -->  {new_name}")
        os.rename(old_path, new_path)


if __name__ == '__main__':
    '''
    for sigma in [5, 15, 25, 50]:
        background = cv2.GaussianBlur(image, (0, 0), sigma)  # 核大小为0表示由sigma决定
        corrected = cv2.subtract(image, background)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1);
        plt.imshow(background, cmap='gray');
        plt.title(f'Background σ={sigma}')
        plt.subplot(1, 2, 2);
        plt.imshow(corrected, cmap='gray');
        plt.title(f'Corrected σ={sigma}')
        plt.show()
    
    
    #面积过滤
    for id in range(1,35):
        input_folder = f'/home/data/changsy/optical_neural_signal/new_data_20260309/semantic_mask/Image_{str(id).zfill(3)}'
        output_folder = f'/home/data/changsy/optical_neural_signal/new_data_20260309/semantic_mask_denoised/Image_{str(id).zfill(3)}'
        filter_components_by_area_batch(input_folder=input_folder, output_folder=output_folder)
    
    
    #轮廓提取
    ids = [7,32]
    for id in ids:
        original_folder = f"Intermediate experimental results/activity_map_jiekang/activity_map_jiekang_{id}/correct_images_rollingball"
        mask_folder = f"label_summary/jiekang/Image {id}/Image_{id}_filtered"
        output_folder = f"label_summary/jiekang/Image {id}/Image {id} outline_GT"
        outline_extraction(original_folder, mask_folder, output_folder)
    
    
    #总标签叠加
    folder_ids = [32]
    for id in folder_ids:
        process_images_with_mask(input_folder=f'Intermediate experimental results/activity_map_jiekang/activity_map_jiekang_{id}/correct_images_rollingball',
                                 mask_path=f'Intermediate experimental results/activity_map_jiekang/activity_map_jiekang_{id}/global_mask_by_or.tif',
                                 output_folder=f'label_summary/jiekang/Image {id}/Image {id} all')
    '''
    #从images folder中生成视频
    folder_ids = [4,16]
    for id in folder_ids:
        create_video_from_images(f'/home/data/changsy/optical_neural_signal/new_data_20260309/instance_mask/Image {id} overlay',
                        f'/home/data/changsy/optical_neural_signal/new_data_20260309/instance_mask/Image {id} overlay.mp4', interval_seconds=0.25, fps=30)
    '''
                            
    #把npy文件转换为tif标签 
    ids = [5,7,9,12,18,24,32]
    for id in ids:
        input_dir = f'label_jiekang/Image_{id}_rgb'
        output_dir = f'label_jiekang/Image_{id}_filtered'
        transform_npy_to_tif(input_dir=input_dir, output_dir=output_dir)
    
    
    #填洞+去毛刺
    for id in range(1,35):
        input_folder = f'/home/data/changsy/optical_neural_signal/new_data_20260309/semantic_mask_denoised/Image_{str(id).zfill(3)}'
        output_folder = f'/home/data/changsy/optical_neural_signal/new_data_20260309/semantic_mask_cleaned/Image_{str(id).zfill(3)}'
        clean_masks_folder(input_folder, output_folder, open_radius=1, close_radius=1, overwrite=False, verbose=True)
    

    for id in [4,16]:
        #把语义级标签转化为实例级标签
        input_folder = f'/home/data/changsy/optical_neural_signal/new_data_20260309/semantic_mask_relabel_tif/Image {id}'
        output_folder = f'/home/data/changsy/optical_neural_signal/new_data_20260309/instance_mask/Image {id}'
        semantic_to_instances(input_folder, output_folder)
    

    for id in [4,16]:
    #可视化实例级标签——映射为随机颜色
        input_folder = f"/home/data/changsy/optical_neural_signal/new_data_20260309/instance_mask/Image {id}"
        output_folder = f"/home/data/changsy/optical_neural_signal/new_data_20260309/instance_mask/Image {id} colored"
        labels_to_color_batch(input_folder, output_folder)
    
    for id in [4,16]:
    #将彩色RGB标签半透明叠加到原图上
        rgb_folder = f"/home/data/changsy/optical_neural_signal/new_data_20260309/instance_mask/Image {id} colored"
        raw_folder = f"/home/data/changsy/optical_neural_signal/new_data_20260309/data_processed_relabel_jpg/Image {id}"
        output_folder = f"/home/data/changsy/optical_neural_signal/new_data_20260309/instance_mask/Image {id} overlay"
        overlay_instance_labels(rgb_folder, raw_folder, output_folder)
    
    #把实例标签转为语义标签
    instances_to_semantic("Fluorescent cell segmentation/instance_label/label_jiekang/antago_Image_9", 
    "Fluorescent cell segmentation/semantic_label/label_jiekang/Image 9 further_filtered")
    
    
    ids = [2,4,6,8,11,18,24,30,39]
    for id in ids:
        rename_sequential(f'instance_label/label_jidong/Image {id}', image_index=id)
    '''
