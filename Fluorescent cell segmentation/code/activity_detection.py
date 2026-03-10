#!/usr/bin/env python3
"""
stage1_activity_detection.py

Stage 1: compute activity map from a folder of jpg frames, detect candidate ROIs (bboxes),
save visualizations and candidate patches.

Usage:
    python stage1_activity_detection.py --input_dir "path/to/frames" --outdir ./stage1_out

Author: Changsy
"""
import os, glob, re, argparse, csv
import numpy as np
from skimage import io, color, filters, morphology, measure
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib import cm

def natural_sort_key(s):
    # split by numbers for natural sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
#从文件夹中读取荧光图像，并堆叠成stack
def read_frames_from_dir(dpath, take_green=True, verbose=True):
    # collect jpg/png
    files = sorted(glob.glob(os.path.join(dpath, "*.jpg")) + glob.glob(os.path.join(dpath, "*.png")),
                   key=natural_sort_key)
    if len(files) == 0:
        raise ValueError(f"No .jpg or .png files found in {dpath}")
    frames = []
    for f in files:
        img = io.imread(f)
        if img.ndim == 3:
            if take_green and img.shape[2] >= 2:
                g = img[...,1]
                arr = g.astype(np.float32)
            else:
                # convert to grayscale
                arr = color.rgb2gray(img).astype(np.float32)
        else:
            arr = img.astype(np.float32)
        # normalize per image robustly
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            # if already float but probably 0-1
            if arr.max() <= 1.0:
                arr = arr.astype(np.float32)
            else:
                arr = arr / (arr.max() + 1e-12)
        else:
            # integer type
            arr = arr / (np.iinfo(img.dtype).max + 1e-12)
        frames.append(arr)
    stack = np.stack(frames, axis=0)  # T,H,W with values ~0-1
    if verbose:
        print(f"Loaded {len(frames)} frames, shape: {stack.shape}")
    return stack, files

#对整体图像做空间大尺度背景去除
def subtract_spatial_background(stack, bg_sigma=40, scale=1.0):
    # stack: (T,H,W) in 0-1
    # 计算每帧的大尺度背景并减去（或用 mean/background）
    T = stack.shape[0]
    out = np.empty_like(stack)
    for t in range(T):
        frame = stack[t]
        bg = gaussian_filter(frame, sigma=bg_sigma)
        out[t] = frame - scale * bg
    # re-normalize to 0-1 per-frame (可选)
    out = out - np.min(out)
    out = out / (np.max(out) + 1e-12)
    return out

#做像素级时间baseline去除
def temporal_pixel_baseline_subtract(stack, pct=8):
    # stack: T,H,W (0-1)
    baseline = np.percentile(stack, pct, axis=0)  # (H,W)
    out = stack - baseline[None, ...]
    # clip negative to 0 and normalize
    out = np.clip(out, 0, None)
    out = out / (out.max() + 1e-12)
    return out

def temporal_high_pass(stack, win = 20):
    baseline = np.convolve(stack.mean(axis = (1,2)), np.ones(win)/win, mode='same')
    return stack - baseline[:, None, None]

def compute_activity_map(stack,
                         do_saptial_bg = True, bg_sigma = 50,
                         do_pixel_baseline = True, baseline_pet = 10,
                         do_temporal_high_pass = False, win = 20,
                         weights=(0.2,0.4,0.2,0.2)):
    # 1) spatial background subtraction
    if do_saptial_bg:
        stack = subtract_spatial_background(stack, bg_sigma)

    # 2) temporal pixel baseline subtraction
    if do_pixel_baseline:
        stack = temporal_pixel_baseline_subtract(stack, baseline_pet)

    # 3) temporal high pass
    if do_temporal_high_pass:
        stack = temporal_high_pass(stack, win)

    # stack: (T,H,W)
    max_proj = np.max(stack, axis=0)
    p90 = np.percentile(stack, 98, axis=0)
    p10 = np.percentile(stack, 2, axis=0)
    p90_10 = p90 - p10
    std_map = np.std(stack, axis=0)
    diff_sum = np.max(np.abs(np.diff(stack, axis=0)), axis=0)

    def norm01(x):
        p1, p99 = np.percentile(x, (1,99))
        out = np.clip((x - p1) / (p99 - p1 + 1e-12), 0, 1)
        return out

    maps = [norm01(p90_10), norm01(max_proj), norm01(std_map), norm01(diff_sum)]
    w = weights
    act = w[0]*maps[0] + w[1]*maps[1] + w[2]*maps[2] + w[3]*maps[3]
    act = gaussian_filter(act, sigma=1.0)
    # final normalization
    act = (act - act.min()) / (act.max() - act.min() + 1e-12)
    return act, {'p90_10':maps[0], 'max':maps[1], 'std':maps[2], 'diff_sum':maps[3]}


def save_activity_visuals(activity_map, outdir):
    os.makedirs(outdir, exist_ok=True)
    image_id = os.path.basename(outdir)
    # 保存为热力图图像（三维RGB），使用skimage直接保存，尺寸与activity_map一致

    cmap = plt.colormaps['hot']
    activity_rgb = (cmap(activity_map) * 255).astype(np.uint8)[:, :, :3]  # 将activity_map映射为RGB
    io.imsave(os.path.join(outdir, f'activity_map_{image_id}.tif'), activity_rgb)
    '''
    # 同时保存一个matplotlib生成的版本（可能尺寸不同，用于可视化）
    plt.figure(figsize=(6,6))
    plt.imshow(activity_map, cmap='hot')
    plt.axis('off')
    plt.savefig(
        os.path.join(outdir, 'activity_map_jidong_6_visualization_rollingball_test.png'),
        dpi=200,
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close()
    '''

def main(input_dir, outdir, args):
    stack, files = read_frames_from_dir(input_dir, take_green=not args.use_rgb)
    activity_map, maps = compute_activity_map(stack, weights=(args.w1,args.w2,args.w3,args.w4))
    save_activity_visuals(activity_map, outdir)


    print("Done. Visuals and patches saved in:", outdir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    #p.add_argument('--input_dir', required=True, help='Directory containing .jpg/.png frames (will be sorted naturally)')
    #p.add_argument('--outdir', required=True, help='Output directory to save maps, patches, csv')
    p.add_argument('--use_rgb', action='store_true', help='If set, use full RGB->gray conversion; default uses green channel only')
    p.add_argument('--w1', type=float, default=0.0, help='weight for p90-p10 map')
    p.add_argument('--w2', type=float, default=1.0, help='weight for max projection')
    p.add_argument('--w3', type=float, default=0.0, help='weight for std map')
    p.add_argument('--w4', type=float, default=0.0, help='weight for frame-diff sum')
    p.add_argument('--th_scale', type=float, default=0.5, help='scale factor for Otsu threshold (smaller => looser)')
    p.add_argument('--min_area', type=int, default=50, help='minimum connected region area (pixels)')
    p.add_argument('--pad', type=int, default=32, help='padding when extracting bbox patches')
    p.add_argument('--close_disk', type=int, default=3, help='disk radius for binary closing')
    args = p.parse_args()
    for i in range(1,36):
        input_dir = f"/home/data/changsy/optical_neural_signal/new_data_20260309/data_processed/Image_00{i}"
        outdir = f"/home/data/changsy/optical_neural_signal/new_data_20260309/activity_map/Image_00{i}"
        main(input_dir, outdir, args)
