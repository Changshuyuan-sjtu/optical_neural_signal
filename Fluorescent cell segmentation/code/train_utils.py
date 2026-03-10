import torch
import torch.nn.functional as F
from Loss import dice_loss
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

# ---------------------------
# Training loop skeleton
# ---------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc='Training'):

        x, y = batch

        x = x.to(device)  # B,1,T,H,W
        y = y.to(device)  # B,1,H,W
        optimizer.zero_grad()
        with torch.amp.autocast('cuda:2', enabled=scaler is not None):
            logits = model(x)  # B,1,H,W
            probs = torch.sigmoid(logits)
            bce = F.binary_cross_entropy_with_logits(logits, y)
            dloss = dice_loss(probs, y)
            loss = bce + dloss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * x.shape[0]
        scheduler.step()
    return running_loss / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    tot_loss = 0.0
    dices = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            x,y = batch
            x = x.to(device); y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(probs, y)
            tot_loss += loss.item() * x.shape[0]
            pred = (probs > 0.5).float()
            # compute dice
            intersection = (pred * y).sum()
            dice = (2. * intersection) / (pred.sum() + y.sum() + 1e-8)
            dices.append(dice.item())
    return tot_loss / len(loader.dataset), np.mean(dices)


def plot_loss_curve(loss_list, save_path, title):
    # 创建图形和坐标轴
    plt.figure(figsize=(10, 6))
    # 生成x轴数据（epoch序号）
    x = range(1, len(loss_list) + 1)
    # 绘制损失曲线
    plt.plot(x, loss_list, 'b-', linewidth=2, label='Training Loss')
    # 设置标题和坐标轴标签
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    # 设置x轴刻度为整数
    plt.xticks(np.arange(1, len(loss_list) + 1, max(1, len(loss_list) // 10)))
    # 添加网格
    plt.grid(True, alpha=0.3)
    # 添加图例
    plt.legend(fontsize=12)
    # 自动调整布局
    plt.tight_layout()
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # 显示图片
    plt.show()
    print(f"损失曲线已保存至: {save_path}")


def plot_dice_curve(dice_list, save_path, title):
    # 创建图形和坐标轴
    plt.figure(figsize=(10, 6))
    # 生成x轴数据（epoch序号）
    x = range(1, len(dice_list) + 1)
    # 绘制损失曲线
    plt.plot(x, dice_list, 'b-', linewidth=2, label='Valid dice')
    # 设置标题和坐标轴标签
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Dice', fontsize=12)
    # 设置x轴刻度为整数
    plt.xticks(np.arange(1, len(dice_list) + 1, max(1, len(dice_list) // 10)))
    # 添加网格
    plt.grid(True, alpha=0.3)
    # 添加图例
    plt.legend(fontsize=12)
    # 自动调整布局
    plt.tight_layout()
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # 显示图片
    plt.show()
    print(f"Dice曲线已保存至: {save_path}")
