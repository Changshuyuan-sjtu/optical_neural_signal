import os, glob, random, math, time
import argparse
import torch
from model import *
from tools4dataset import *
from train_utils import *
from model_v2 import *

# -----------------------------
# 主函数入口
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type = str, required = True,
                        help="训练数据文件夹，包含 stack1, stack2, ...")
    parser.add_argument("--train_label_root", type = str, required = True,
                        help="训练标签文件夹，包含 stackX_label.png")
    parser.add_argument("--outdir", type = str, default = "./results",
                        help="输出保存文件夹")
    parser.add_argument("--n_folds", type = int, default = 7,
                        help="交叉验证折数")
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--batch_size", type = int, default = 16)
    parser.add_argument("--patch_size", type = int, default = None,
                        help="规则裁剪块大小，如 256；默认None表示整张图")
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--device", type = str, default = "cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_id", type = int, default = 2, help="指定使用的GPU编号")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # 五折交叉验证
    # -----------------------------
    for fold_idx in range(args.n_folds):
        print(f"\n===== Fold {fold_idx+1}/{args.n_folds} =====")

        output_dir = os.path.join(args.outdir, f"fold_{fold_idx+1}")
        os.makedirs(output_dir, exist_ok=True)

        # 构建数据加载器
        train_loader, val_loader, train_dirs, val_dirs = build_fold_loaders(
            train_root=args.train_root,
            train_label_root=args.train_label_root,
            fold_idx=fold_idx,
            n_folds=args.n_folds,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            expected_T=190,
            num_workers=args.num_workers,
            augment=True,
            mode = 'temporal',
            t_window = 7
        )

        '''
        print("train_loader type:", type(train_loader))

        batch = next(iter(train_loader))
        print("train batch type:", type(batch))
        if isinstance(batch, tuple):
            x, y = batch
            print("x.shape", x.shape, "y.shape", y.shape, "y.sum()", y.sum().item())
        else:
            print("Batch has no labels! len:", len(batch))
            print("Type of first element:", type(batch[0]))
        '''

        print(f"Train stacks ({len(train_dirs)}): {[os.path.basename(d) for d in train_dirs]}")
        print(f"Val stacks ({len(val_dirs)}): {[os.path.basename(d) for d in val_dirs]}")

        # 初始化网络
        model = TemporalUNet(base=32).to(device)

        # 损失函数和优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 7125, eta_min = 1e-7)
        best_val_loss = float("inf")
        model_save_path = os.path.join(output_dir, f"best_model_fold{fold_idx+1}.pth")
        train_loss_list = []
        val_loss_list = []
        val_dices_list = []
        # -----------------------------
        # 训练循环
        # -----------------------------
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
            val_loss = validate(model, val_loader, device)
            print(f"Train Loss: {train_loss:.4f} | Val total Loss: {val_loss[0]:.4f} | Val dice: {val_loss[1]:.4f}")

            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss[0])
            val_dices_list.append(val_loss[1])

            if val_loss[0] < best_val_loss:
                best_val_loss = val_loss[0]
                torch.save(model.state_dict(), model_save_path)
                print(f"✅ Best model saved (val_loss={val_loss[0]:.4f})")
        # save loss curve and dice curve
        plot_loss_curve(train_loss_list, os.path.join(output_dir, f'Fold {fold_idx+1} training loss.png'), f'Fold {fold_idx+1} training loss')
        plot_loss_curve(val_loss_list, os.path.join(output_dir, f'Fold {fold_idx+1} validation loss.png'), f'Fold {fold_idx+1} validation loss')
        plot_dice_curve(val_dices_list, os.path.join(output_dir, f'Fold {fold_idx+1} validation dice.png'), f'Fold {fold_idx+1} validation dice')

        print(f"Fold {fold_idx+1} finished. Best val loss = {best_val_loss:.4f}")

    print("===== All folds done =====")

if __name__ == "__main__":
    main()