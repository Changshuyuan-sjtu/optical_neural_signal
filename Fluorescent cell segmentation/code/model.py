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

class UNet2DDecoder(nn.Module):
    def __init__(self, in_ch=128, base=64):
        super().__init__()
        # encoder-like feature channels expected: [base, base*2, base*4, base*8] mapped to 2D via pooler
        self.enc1 = conv2d_block(in_ch, base)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = conv2d_block(base, base*2)
        self.enc3 = conv2d_block(base*2, base*4)
        self.enc4 = conv2d_block(base*4, base*8)
        # decoder
        self.up3 = nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)
        self.dec3 = conv2d_block(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.dec2 = conv2d_block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)
        self.dec1 = conv2d_block(base*2, base)
        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x, skips):
        # x: B, C, H', W'  (from deepest pool->spatial)
        # skips: list of skip features [s1, s2, s3] already pooled to appropriate sizes
        e1 = self.enc1(x)     # B,base,H',W'
        e2 = self.pool(e1)
        e2 = self.enc2(e2)    # B, base*2, H'/2, W'/2
        e3 = self.pool(e2)
        e3 = self.enc3(e3)
        e4 = self.pool(e3)
        e4 = self.enc4(e4)
        # decoder with skip connections: here we pair e4 with skips[-1] etc.
        u3 = self.up3(e4)  # up to e3 size
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)
        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        out = self.outc(d1)
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
        self.decoder = UNet2DDecoder(in_ch=decoder_in, base=base*4)  # base scaled for decoder

    def forward(self, x):
        # x: (B,1,T,H,W)
        feats = self.encoder(x)  # list f1..f4 with shapes (B,c,t,h,w)
        pooled = [pool(f) for pool, f in zip(self.poolers, feats)]  # each -> (B,C2,H',W')
        # we will align spatial sizes: pooled[-1] is deepest; others should be progressively larger
        # For UNet decoder we need e1,e2,e3 derived from pooled features
        # To simplify: we will upsample pooled[-1] to size of pooled[0] and feed decoder which itself will downsample inside
        # But better: pass pooled[-1] as input, and inside decoder pooling will create hierarchy
        deepest = pooled[-1]
        # For skip connections inside decoder, we can't directly inject the pooled skip features unless we adapt sizes.
        # For simplicity in this code: we will set encoder-like e1..e3 inside decoder from computed pooled features by resizing.
        # Resize pooled features to desired sizes inside decoder call by stacking as a list in forward.
        # Here we simply supply deepest as x and decoder will use internal pooling for skip - this is a pragmatic choice.
        out = self.decoder(deepest, skips=None)  # decoder currently ignores external skips (we simplified)
        # upsample to original resolution if needed
        out = F.interpolate(out, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        return out


if __name__ == '__main__':
    model = TemporalUNet()
    input = torch.randn(1, 1, 7, 512, 512)
    out = model(input)
    print(out.shape)