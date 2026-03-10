import os
import glob
import random
from typing import List, Tuple
import numpy as np
import torch
from skimage import io, color
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
import cv2

# -----------------------------
# Utility: read stack in a folder
# -----------------------------
def read_stack_from_folder(folder: str, exts=('.jpg', '.png', '.tif', '.tiff')):
    """
    读取指定文件夹内的所有帧图像，并按字母表严格排序。
    返回: numpy array (T,H,W), float32 类型。
    """
    files = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ]
    files = sorted(files)

    if len(files) == 0:
        raise RuntimeError(f"Folder {folder} is empty.")

    frames = []
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise ValueError(f"文件损坏或无法读取: {f}")

        im = im.astype(np.float32)
        frames.append(im)

    stack = np.stack(frames, axis=0)  # (T, H, W)，这里的 T 就是真实的图像数量
    return stack

# -----------------------------
# Dataset: yields either full-stack samples or tiled patches (deterministic grid)
# -----------------------------
class StackDataset(Dataset):
    """
    Dataset that enumerates either:
      - full stacks (one sample per stack), or
      - grid tiles from each stack (multiple samples per stack), with deterministic tiling order:
          top-to-bottom, left-to-right.

    Args:
      stack_root_dirs: list of directories, each is one stack folder (e.g., train_root/stack1)
      label_root: path to label folder (where files named e.g., stack1_label.png exist), or None.
      patch_size: None -> full stack (no tiling). If int -> tile size (square).
      augment: whether to apply simple spatial augment (random flips/rot90) consistently across frames
    """
    def __init__(self, stack_root_dirs: List[str], label_root: str = None,
                 patch_size: int = None,  augment: bool = False):
        self.stack_dirs = sorted(stack_root_dirs)
        self.label_root = label_root
        self.patch_size = patch_size
        self.augment = augment

        # build index mapping: each item -> (stack_dir, top, left, Hcrop, Wcrop)
        self.index = []
        for sd in self.stack_dirs:
            # read one file to get H,W (lazy: we don't load full stack here for speed)
            # find one image file in folder
            sample_files = sorted(glob.glob(os.path.join(sd, '*')))
            if len(sample_files) == 0:
                raise RuntimeError(f"No files in {sd}")
            # read any first frame to get size
            first = io.imread(sample_files[0])
            if first.ndim == 3:
                H = first.shape[0]; W = first.shape[1]
            else:
                H = first.shape[0]; W = first.shape[1]
            if self.patch_size is None or self.patch_size >= max(H, W):
                # single item full-stack
                self.index.append((sd, 0, 0, H, W))
            else:
                ph = self.patch_size
                # deterministic tiling: rows top->bottom, within row left->right
                y_starts = list(range(0, H, ph))
                x_starts = list(range(0, W, ph))
                for y in y_starts:
                    for x in x_starts:
                        y1 = min(y+ph, H)
                        x1 = min(x+ph, W)
                        y0 = max(0, y1-ph)
                        x0 = max(0, x1-ph)
                        self.index.append((sd, y0, x0, y1-y0, x1-x0))

    def __len__(self):
        return len(self.index)

    def _get_label_path_for_stack(self, stack_dir):
        # label file name expected: <stackname>_label.<ext>
        stackname = os.path.basename(stack_dir.rstrip('/\\'))
        # search in label_root for stackname_label.*
        if self.label_root is None:
            return None
        candidates = []
        for ext in ['.tif', '.tiff']:
            p = os.path.join(self.label_root, f"{stackname}_label{ext}")
            if os.path.exists(p):
                candidates.append(p)
        if len(candidates) == 0:
            raise RuntimeError(f"No label file found for stack {stackname} in label_root {self.label_root}. Expected name {stackname}_label.*")
        return candidates[0]

    def __getitem__(self, idx):
        sd, y0, x0, h, w = self.index[idx]
        # read stack (T,H,W)
        stack = read_stack_from_folder(sd)  # float32
        # crop spatial region
        crop = stack[:, y0:y0+h, x0:x0+w]  # T,h,w
        # normalize per-sample robustly to [0,1]
        p1, p99 = np.percentile(crop, (1,99))
        crop = np.clip((crop - p1) / (p99 - p1 + 1e-12), 0.0, 1.0)

        # label
        label = None
        if self.label_root is not None:
            lab_path = self._get_label_path_for_stack(sd)
            lab_img = io.imread(lab_path)
            if lab_img.ndim == 3:
                lab_img = color.rgb2gray(lab_img)
            lab_img = (lab_img > 127).astype(np.uint8)
            lab_crop = lab_img[y0:y0+h, x0:x0+w]
            label = lab_crop

        # simple spatial augmentations (apply consistently across frames & label)
        if self.augment:
            # 随机水平翻转
            if random.random() < 0.5:
                crop = crop[:, :, ::-1]  # T, H, W -> flip horizontally
                if label is not None:
                    label = label[:, ::-1]
            # 随机垂直翻转
            if random.random() < 0.5:
                crop = crop[:, ::-1, :]
                if label is not None:
                    label = label[::-1, :]
            # 随机 90 度旋转 (0,1,2,3 times)
            k = random.randint(0, 3)
            if k:
                crop = np.rot90(crop, k=k, axes=(1, 2))  # rotate each frame
                if label is not None:
                    label = np.rot90(label, k=k)
            # 随机亮度/对比度（对所有帧使用相同因子）
            if random.random() < 0.5:
                fac = 0.8 + 0.4 * random.random()  # 0.8 - 1.2
                crop = np.clip(crop * fac, 0.0, 1.0)
            # 随机 gamma
            if random.random() < 0.3:
                g = 0.8 + 0.6 * random.random()  # 0.8 - 1.4
                crop = np.clip(np.power(crop, g), 0.0, 1.0)

        crop = crop.copy()
        if label is not None:
            label = label.copy()

        # convert to torch tensors
        # shape to (1,T,H,W) for model input
        x = torch.from_numpy(crop).float().unsqueeze(0)
        if label is None:
            raise RuntimeError("label is None but dataset expects labels.")
        else:
            y = torch.from_numpy(label).float().unsqueeze(0)  # 1,h,w
            return x, y

# -----------------------------
# window dataset
# -----------------------------
class TemporalWindowDataset(Dataset):
    def __init__(self, img_dir, label_dir, t_window=7, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.t_window = t_window
        self.transform = transform
        self.img_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(".jpg")
        ])
        self.label_paths = sorted([
            os.path.join(label_dir, f) for f in os.listdir(label_dir)
            if f.lower().endswith((".tif", ".tiff"))
        ])
        assert len(self.img_paths) == len(self.label_paths), \
            f"Mismatch: {len(self.img_paths)} images vs {len(self.label_paths)} labels"
        self.T = len(self.img_paths)

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        half = self.t_window // 2
        indices = [(idx + i) % self.T for i in range(-half, half + 1)]  # 环状补帧
        imgs = []
        for i in indices:
            img = cv2.imread(self.img_paths[i], cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            imgs.append(img)
        stack = np.stack(imgs, axis=0)  # [T, H, W]

        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)
        label = (label > 127).astype(np.float32)

        stack = torch.from_numpy(stack).unsqueeze(0)  # [1,T,H,W]
        label = torch.from_numpy(label).unsqueeze(0)  # [1,H,W]
        return stack, label


# -----------------------------
# 测试阶段专用 Dataset（兼容 full / temporal 模式）
# -----------------------------
class TestStackDataset(Dataset):
    """
    测试阶段用的数据集类 (支持任意长度的序列)
    支持两种模式：
      - mode="full": 返回整个 stack (1, T, H, W)
      - mode="temporal": 对每一帧构建一个时间窗口 (1, t_window, H, W)
    """

    # 我们可以保留 expected_T 参数以防你的旧代码调用报错，但在内部不再使用它进行截断
    def __init__(self, test_root, mode="full", t_window=7):
        self.stack_dirs = sorted([
            os.path.join(test_root, d)
            for d in os.listdir(test_root)
            if os.path.isdir(os.path.join(test_root, d))
        ])
        self.mode = mode
        self.t_window = t_window

        if len(self.stack_dirs) == 0:
            raise RuntimeError(f"No stack subfolders found in {test_root}")

        self.stacks = []
        for sd in self.stack_dirs:
            frames = sorted([
                os.path.join(sd, f)
                for f in os.listdir(sd)
                if f.lower().endswith((".jpg", ".png", ".tif", ".tiff"))
            ])

            if len(frames) == 0:
                print(f"[WARN] stack {sd} is empty!")
                continue

            # 核心修改：不再使用 frames[:expected_T]，直接保存全部帧
            self.stacks.append(frames)
            # 你可以打印出来确认一下真实长度
            print(f"[INFO] Loaded stack {os.path.basename(sd)} with {len(frames)} frames.")

    def __len__(self):
        if self.mode == "full":
            return len(self.stack_dirs)
        elif self.mode == "temporal":
            return sum(len(frames) for frames in self.stacks)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def __getitem__(self, idx):
        if self.mode == "full":
            stack_dir = self.stack_dirs[idx]
            stack_name = os.path.basename(stack_dir)

            # 注意：不再传递 expected_T 参数
            stack = read_stack_from_folder(stack_dir)  # (T,H,W)
            p1, p99 = np.percentile(stack, (1, 99))
            stack = np.clip((stack - p1) / (p99 - p1 + 1e-12), 0.0, 1.0)
            x = torch.from_numpy(stack).float().unsqueeze(0)  # (1,T,H,W)
            return x, stack_name

        elif self.mode == "temporal":
            cum_lens = np.cumsum([len(f) for f in self.stacks])
            stack_id = np.searchsorted(cum_lens, idx, side="right")
            frame_id = idx if stack_id == 0 else idx - cum_lens[stack_id - 1]
            frame_list = self.stacks[stack_id]
            T = len(frame_list)

            half = self.t_window // 2
            indices = [(frame_id + i) % T for i in range(-half, half + 1)]

            imgs = []
            for i in indices:
                img = cv2.imread(frame_list[i], cv2.IMREAD_GRAYSCALE)
                img = img.astype(np.float32) / 255.0
                imgs.append(img)
            stack = np.stack(imgs, axis=0)
            stack = torch.from_numpy(stack).float().unsqueeze(0)

            current_img_path = frame_list[frame_id]
            stack_name = os.path.splitext(os.path.basename(current_img_path))[0]

            return stack, stack_name
        else:
            raise ValueError(f"Unsupported mode '{self.mode}'")


# 自定义collate函数（兼容3D输入）
def custom_collate(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)  # [B, 1, T, H, W]
    ys = torch.stack(ys, dim=0)  # [B, 1, H, W]
    return xs, ys

# -----------------------------
# Build train/val loaders using KFold on stacks
# -----------------------------
def build_fold_loaders(train_root: str, train_label_root: str,
                       fold_idx: int, n_folds: int = 5,
                       batch_size: int = 1, patch_size: int = None,
                       num_workers: int = 4,
                       augment: bool = True,
                       mode: str = "full",            # "full" or "temporal"
                       t_window: int = 7):            # temporal模式下时间窗口长度
    """
    Args:
        train_root: 含多个stack子文件夹的根路径，每个子文件夹含多帧图像
        train_label_root: 含各stack对应label文件的路径
        fold_idx: 当前fold的索引 (0..n_folds-1)
        n_folds: K折数量
        batch_size: 批大小
        patch_size: optional patch crop size
        num_workers: dataloader线程数
        augment: 是否启用增强
        mode: "full"=整stack模式，"temporal"=时间窗口模式
        t_window: temporal模式下的时间窗口长度（奇数）
    Returns:
        train_loader, val_loader, train_dirs, val_dirs
    """

    # ---- 列出所有stack文件夹 ----
    stack_dirs = sorted([
        os.path.join(train_root, d)
        for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
    ])
    if len(stack_dirs) == 0:
        raise RuntimeError(f"No stack subfolders found in {train_root}")

    # ---- KFold划分 ----
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(kf.split(stack_dirs))
    train_idx, val_idx = splits[fold_idx]
    train_dirs = [stack_dirs[i] for i in train_idx]
    val_dirs = [stack_dirs[i] for i in val_idx]

    # ---- 根据模式选择Dataset ----
    if mode == "full":
        # 模式1：整stack训练
        train_ds = StackDataset(
            train_dirs, label_root=train_label_root,
            patch_size=patch_size, augment=augment
        )
        val_ds = StackDataset(
            val_dirs, label_root=train_label_root,
            patch_size=patch_size, augment=False
        )

    elif mode == "temporal":
        # 模式2：局部时间窗口训练（带环状补帧）
        def make_temporal_concat(stack_dir_list):
            datasets = []
            for stack_dir in stack_dir_list:
                stack_name = os.path.basename(stack_dir)
                label_dir = os.path.join(train_label_root, stack_name)
                if not os.path.exists(label_dir):
                    raise FileNotFoundError(f"Missing label folder: {label_dir}")
                ds = TemporalWindowDataset(stack_dir, label_dir, t_window=t_window)
                datasets.append(ds)
            return ConcatDataset(datasets)

        train_ds = make_temporal_concat(train_dirs)
        val_ds = make_temporal_concat(val_dirs)

    else:
        raise ValueError(f"Unsupported mode '{mode}', expected 'full' or 'temporal'.")

    # ---- 构建DataLoader ----
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=custom_collate, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, num_workers // 2), collate_fn=custom_collate
    )

    return train_loader, val_loader, train_dirs, val_dirs

# -----------------------------
# Example usage snippet (to integrate into main)
# -----------------------------
if __name__ == '__main__':
    '''
    # Example minimal demo: build fold 0 loaders
    train_root = "dataset_for_window/train"        # contains stack1, stack2, ...
    train_label_root = "dataset_for_window/train_label"         # contains stack1_label.png, stack2_label.png, ...
    fold = 1
    train_loader, val_loader, tr_dirs, v_dirs = build_fold_loaders(train_root, train_label_root,
                                                                   fold_idx=fold, n_folds=5,
                                                                   batch_size=1, patch_size=None,
                                                                    num_workers=4,
                                                                   augment=True, mode='temporal',
                                                                   t_window=7)
    print("Train stacks nums:", len(tr_dirs))
    print("Val stacks nums:", len(v_dirs))
    # iterate
    for batch in train_loader:
        x,y = batch  # x: [B,1,T,h,w], y: [B,1,h,w]
        print(x.shape, y.shape)
        break
    '''
    test_root = "dataset_for_window/test"
    test_ds = TestStackDataset(test_root, mode = 'temporal', t_window=7)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    for batch in test_loader:
        x, _ = batch
        print(x.shape)
        break