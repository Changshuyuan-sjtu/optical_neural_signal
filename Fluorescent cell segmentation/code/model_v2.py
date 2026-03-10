import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Model: Temporal Encoder + 2D UNet Decoder
# ---------------------------
def conv3d_block(in_ch, out_ch, k=(3,3,3), stride=(1,1,1), padding=1):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.LeakyReLU(0.1, inplace=True)
    )

class TemporalEncoder(nn.Module):
    """3D卷积模块用来逐步下采样时间和空间维度，保留中间计算结果用以残差连接"""
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        # output shapes (T,H,W change according to stride)
        self.b1 = conv3d_block(in_ch, base, k=(3,7,7), stride=(1,2,2), padding=(1,3,3))  # spatial halve
        self.b2 = conv3d_block(base, base*2, k=(3,3,3), stride=(2,2,2), padding=1)    # temporal down x2, spatial down x2
        self.b3 = conv3d_block(base*2, base*4, k=(3,3,3), stride=(2,2,2), padding=1)  # further down
        self.b4 = conv3d_block(base*4, base*8, k=(3,3,3), stride=(2,1,1), padding=1)  # temporal down, keep spatial

    def forward(self, x):
        # x: B, C=1, T, H, W
        f1 = self.b1(x)  # B, base, T, H/2, W/2
        f2 = self.b2(f1) # B, base*2, T/2, H/4, W/4
        f3 = self.b3(f2) # B, base*4, T/4, H/8, W/8
        f4 = self.b4(f3) # B, base*8, T/8, H/8, W/8 (temporal reduced)
        return [f1, f2, f3, f4]

class TemporalPoolToSpatial(nn.Module):
    """通过 mean + 1x1 conv 把时间维度聚合到空间特征中"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, feat3d):
        # feat3d: B, C, T', H', W'
        #x = feat3d.mean(dim = 2)
        x = feat3d.max(dim=2)[0]  # average time: B,C,H',W'
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# 2D UNet decoder blocks
def conv2d_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

# 在 UNet2DDecoder 中增加一个 upsample + conv helper
class UNet2DDecoder(nn.Module):
    def __init__(self, in_ch=128, base=16):
        super().__init__()
        # we'll keep encoder-like convs to build hierarchical decoder backbone
        # but we will primarily use the up blocks and concatenation with skips
        self.conv_initial = conv2d_block(in_ch, base*8)  # process deepest input

        # progressive up blocks (upsample -> concat skip -> conv block)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base*8 + base*8, base*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base*4),
            nn.ReLU(inplace=True),
            conv2d_block(base*4, base*4)
        )  # output: base*4, 128x128 if input 64x64

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base*4 + base*4, base*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base*2),
            nn.ReLU(inplace=True),
            conv2d_block(base*2, base*2)
        )  # next: base*2, 256x256

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base*2 + base*2, base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            conv2d_block(base, base)
        )  # final: base, 512x512

        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, deepest, skips):
        """
        deepest: (B, C_deep, Hd, Wd)  # e.g., 64x64
        skips: [p3, p2, p1] in order from deeper->shallower OR shallow->deep (we'll assume [p3,p2,p1])
        where p3 ~ 64x64, p2 ~ 128x128, p1 ~ 256x256
        """
        # initial processing of deepest
        x = self.conv_initial(deepest)  # (B, base*8, 64,64)

        # up stage 1: up to 128x128 and concat with skip p3 (aligned)
        p3, p2, p1 = skips  # p3 ~ 64, p2 ~128, p1 ~256
        # ensure p3 has same spatial size as x
        if p3.shape[-2:] != x.shape[-2:]:
            p3 = F.interpolate(p3, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, p3], dim=1)
        x = self.up3(x)  # now (B, base*4, 128,128)

        # up stage 2: up to 256x256 and concat with p2
        if p2.shape[-2:] != x.shape[-2:]:
            p2 = F.interpolate(p2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, p2], dim=1)
        x = self.up2(x)  # (B, base*2, 256,256)

        # up stage 3: up to 512x512 and concat with p1
        if p1.shape[-2:] != x.shape[-2:]:
            p1 = F.interpolate(p1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, p1], dim=1)
        x = self.up1(x)  # (B, base, 512,512)

        out = self.outc(x)  # (B,1,512,512)
        return out


class TemporalUNet(nn.Module):
    def __init__(self, in_ch=1, base=16):
        super().__init__()
        self.encoder = TemporalEncoder(in_ch, base=base)
        # poolers to convert 3D features to 2D features for decoder skips
        self.poolers = nn.ModuleList([
            TemporalPoolToSpatial(base, base*2),     # for f1 -> map to 2D
            TemporalPoolToSpatial(base*2, base*4),  # for f2
            TemporalPoolToSpatial(base*4, base*8),  # for f3
            TemporalPoolToSpatial(base*8, base*16)  # for f4 (deep)
        ])
        # we will take deepest pooled feature as decoder input and use others as skip-levels
        # choose in_ch for decoder equals pooled deepest channels
        decoder_in = base*16
        self.decoder = UNet2DDecoder(in_ch=decoder_in, base=base)  # base scaled for decoder

    def forward(self, x):
        feats = self.encoder(x)  # f1..f4 (3D)
        pooled = [pool(f) for pool, f in zip(self.poolers, feats)]  # p1..p4 (2D)
        # Expected shapes (example):
        # p1: (B, C1, 256,256)  <- shallowest
        # p2: (B, C2, 128,128)
        # p3: (B, C3, 64,64)
        # p4: (B, C4, 64,64)    <- deepest

        # We'll use deepest p4 as decoder input and use p3,p2,p1 as skips
        deepest = pooled[-1]  # p4
        # Build skip list in the order expected by decoder: [p3, p2, p1]
        skip_p3 = pooled[-2]  # p3
        skip_p2 = pooled[-3]  # p2
        skip_p1 = pooled[-4]  # p1

        out = self.decoder(deepest, skips=[skip_p3, skip_p2, skip_p1])
        return out


if __name__ == '__main__':
    model = TemporalUNet(base=32)
    input = torch.randn(16, 1, 7, 512, 512)
    out = model(input)
    print(out.shape)