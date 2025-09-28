import torch
import torch.nn.functional as F

def weighted_cross_entropy_loss(pred, mask, gamma0=0.5, gamma1=2.5):
    bce_loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    weights = gamma0 * (1 - mask) + gamma1 * mask
    weighted_bce = (weights * bce_loss)

    return weighted_bce.mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()