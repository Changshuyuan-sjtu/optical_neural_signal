import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 3D卷积Block
# ---------------------------
def conv3d_block(in_ch, out_ch, k=(3,3,3), stride=(1,1,1), padding=1):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.LeakyReLU(0.1, inplace=True)
    )

# ---------------------------
# 3D Encoder (不改)
# ---------------------------
class TemporalEncoder(nn.Module):
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        self.b1 = conv3d_block(in_ch, base, k=(3,7,7), stride=(1,2,2), padding=(1,3,3))
        self.b2 = conv3d_block(base, base*2, k=(3,3,3), stride=(2,2,2), padding=1)
        self.b3 = conv3d_block(base*2, base*4, k=(3,3,3), stride=(2,2,2), padding=1)
        self.b4 = conv3d_block(base*4, base*8, k=(3,3,3), stride=(2,1,1), padding=1)

    def forward(self, x):
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)
        return [f1, f2, f3, f4]


# ======================================================
# 🔥 时间聚合 ( Δ=max-min + 可学习时间注意力)
# ======================================================
class TemporalPoolToSpatial(nn.Module):
    """通过 ΔF/F + 可学习时间加权 聚合时间维度到空间特征中"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        # 时间注意模块
        self.time_fc = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_ch, in_ch, kernel_size=1)
        )

    def forward(self, feat3d):
        # feat3d: [B, C, T, H, W]
        B, C, T, H, W = feat3d.shape

        # 计算亮度变化（ΔF = max - min）
        delta = feat3d.max(dim=2).values - feat3d.min(dim=2).values  # [B,C,H,W]

        # 计算时间权重 (可学习)
        t_feat = feat3d.mean(dim=[3, 4])  # -> [B,C,T]
        att = self.time_fc(t_feat)  # -> [B,C,T]
        att = torch.softmax(att, dim=-1)
        att = att.view(B, C, T, 1, 1)

        # 加权求和
        weighted = (feat3d * att).sum(dim=2)  # [B,C,H,W]

        # 将 ΔF 与 加权特征融合
        x = weighted + delta

        # 转换为2D特征
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# ---------------------------
# 2D UNet Decoder
# ---------------------------
def conv2d_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class UNet2DDecoder(nn.Module):
    def __init__(self, in_ch=128, base=64):
        super().__init__()
        self.enc1 = conv2d_block(in_ch, base)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = conv2d_block(base, base*2)
        self.enc3 = conv2d_block(base*2, base*4)
        self.enc4 = conv2d_block(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)
        self.dec3 = conv2d_block(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.dec2 = conv2d_block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)
        self.dec1 = conv2d_block(base*2, base)
        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x, skips=None):
        e1 = self.enc1(x)
        e2 = self.pool(e1); e2 = self.enc2(e2)
        e3 = self.pool(e2); e3 = self.enc3(e3)
        e4 = self.pool(e3); e4 = self.enc4(e4)
        u3 = self.up3(e4); u3 = torch.cat([u3, e3], dim=1); d3 = self.dec3(u3)
        u2 = self.up2(d3); u2 = torch.cat([u2, e2], dim=1); d2 = self.dec2(u2)
        u1 = self.up1(d2); u1 = torch.cat([u1, e1], dim=1); d1 = self.dec1(u1)
        out = self.outc(d1)
        return out


# ---------------------------
# 主网络
# ---------------------------
class TemporalUNet(nn.Module):
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        self.encoder = TemporalEncoder(in_ch, base=base)
        # 使用 Δ + attention 的新pooler 替代旧 TemporalPoolToSpatial
        self.poolers = nn.ModuleList([
            TemporalPoolToSpatial(base, base*2),
            TemporalPoolToSpatial(base*2, base*4),
            TemporalPoolToSpatial(base*4, base*8),
            TemporalPoolToSpatial(base*8, base*16)
        ])
        self.decoder = UNet2DDecoder(in_ch=base*16, base=base*4)

    def forward(self, x):
        feats = self.encoder(x)
        pooled = [pool(f) for pool, f in zip(self.poolers, feats)]
        deepest = pooled[-1]
        out = self.decoder(deepest, skips=None)
        out = F.interpolate(out, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        return out


# ---------------------------
# 测试
# ---------------------------
if __name__ == '__main__':
    model = TemporalUNet()
    input = torch.randn(1, 1, 190, 512, 512)
    out = model(input)
    print(out.shape)
