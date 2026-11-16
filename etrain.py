import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from net.Network import Net, load_checkpoint
from utils.data_val import get_loader
from utils.utils import adjust_lr, clip_gradient


def structure_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    pred = pred.sigmoid()
    intersection = (pred * mask * weight).sum(dim=(2, 3))
    union = ((pred + mask) * weight).sum(dim=(2, 3))
    wiou = 1 - (intersection + 1) / (union - intersection + 1)
    return (bce + wiou).mean()


def dice_loss(pred, target):
    pred = pred.flatten(1)
    target = target.flatten(1)
    numerator = 2 * (pred * target).sum(dim=1) + 1
    denominator = (pred.square() + target.square()).sum(dim=1) + 1
    return (1 - numerator / denominator).mean()


def train(train_loader, model, optimizer, epoch, opt, log):
    model.train()
    total_steps = len(train_loader)
    total_loss = 0
    for step, (images, masks, edges, (text_id,), (text_mask,), (text_type_id,)) in enumerate(train_loader, 1):
        optimizer.zero_grad()
        images, masks, edges = images.cuda(), masks.cuda(), edges.cuda()
        preds = model(images, text_id.cuda(), text_mask.cuda(), text_type_id.cuda())
        loss_final = structure_loss(preds[0], masks)
        loss_init = sum(structure_loss(pred, masks) for pred in preds[1:4])
        loss_edge = sum(dice_loss(pred, edges) for pred in preds[4:8])
        loss = loss_final + loss_init + loss_edge
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        total_loss += loss.detach()

        if step == 1 or step % 20 == 0 or step == total_steps:
            message = (
                f'Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{step:04d}/{total_steps:04d}], '
                f'Total_loss: {loss.item():.4f}, edgeLoss: {loss_edge.item():.4f}, '
                f'Loss_init: {loss_init.item():.4f}, Loss_final: {loss_final.item():.4f}'
            )
            print(f'{datetime.now()} {message}')
            print(f'[Train Info]: {message}', file=log, flush=True)

    average = (total_loss / total_steps).item()
    print(f'[Train Info]: Epoch [{epoch:03d}/{opt.epoch:03d}], Loss_AVG: {average:.4f}', file=log, flush=True)
    return average


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--trainsize', type=int, default=416)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=50)
    parser.add_argument('--bert_path', default='/home/lsl/lsl-IML/TIFS2026/models/bert-base-uncased')
    parser.add_argument('--pvt_checkpoint', default='/home/lsl/lsl-IML/TIFS2026/models/pvt_v2_b2.pth')
    parser.add_argument('--load', type=Path, help='checkpoint to resume from')
    parser.add_argument('--gpu_id', default='0,1')
    parser.add_argument('--train_root', type=Path, default='/home/lsl/lsl-IML/IML-DS/train/')
    parser.add_argument('--text_dir', default='Text', help='text directory inside the training root')
    parser.add_argument('--save_path', type=Path, default='training_output')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    torch.backends.cudnn.benchmark = True
    model = Net(opt)
    model.bert.requires_grad_(False)
    if opt.load is not None:
        load_checkpoint(model, opt.load)
        print(f'Loaded {opt.load}')
    model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)

    root = opt.train_root
    train_loader = get_loader(
        image_root=str(root / 'Tp'), gt_root=str(root / 'Gt'), edge_root=str(root / 'Edge'),
        text_root=str(root / opt.text_dir), bert_path=opt.bert_path, batchsize=opt.batchsize,
        trainsize=opt.trainsize, num_workers=4,
    )
    if len(train_loader) == 0:
        raise ValueError(f'No training batches found in {root}')

    opt.save_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(opt.save_path / 'summary'))
    try:
        with (opt.save_path / 'log.log').open('a', encoding='utf-8') as log:
            print('Config: ' + ', '.join(f'{key}={value}' for key, value in vars(opt).items()), file=log, flush=True)
            for epoch in range(1, opt.epoch):
                lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
                writer.add_scalar('learning_rate', lr, global_step=epoch)
                try:
                    average = train(train_loader, model, optimizer, epoch, opt, log)
                except KeyboardInterrupt:
                    torch.save(model.state_dict(), opt.save_path / f'Net_epoch_{epoch + 1}.pth')
                    raise
                writer.add_scalar('Loss-epoch', average, global_step=epoch)
                if epoch >= 50:
                    torch.save(model.state_dict(), opt.save_path / f'Net_epoch_{epoch}.pth')
    finally:
        writer.close()


if __name__ == '__main__':
    main()
