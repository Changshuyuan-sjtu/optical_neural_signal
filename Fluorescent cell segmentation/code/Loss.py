# ---------------------------
# Dice loss utility
# ---------------------------
def dice_loss(pred, target, smooth=1e-6):
    # pred: probabilities [B,1,H,W]; target: binary [B,1,H,W]
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    loss = 1 - ((2. * inter + smooth) / (denom + smooth))
    return loss.mean()