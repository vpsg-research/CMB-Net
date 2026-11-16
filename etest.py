import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from net.Network import Net, load_checkpoint
from utils.data_val import test_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=416)
    parser.add_argument('--bert_path', default='/home/lsl/lsl-IML/TIFS2026/models/bert-base-uncased')
    parser.add_argument('--pvt_checkpoint', default='/home/lsl/lsl-IML/TIFS2026/models/pvt_v2_b2.pth')
    parser.add_argument('--gpu_id', default='0,1')
    parser.add_argument('--pth_path', type=Path, required=True)
    parser.add_argument('--test_root', type=Path, default='/home/lsl/lsl-IML/IML-DS/test/')
    parser.add_argument('--text_dir', default='Text', help='use GPT_Text to test with GPT captions')
    parser.add_argument('--save_path', type=Path, default='result')
    parser.add_argument('--datasets', nargs='+', default=['C1', 'Coverage', 'Columbia', 'NC16'])
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    model = Net(opt)
    load_checkpoint(model, opt.pth_path)
    model.cuda().eval()

    with torch.no_grad():
        for name in opt.datasets:
            root = opt.test_root / name
            output = opt.save_path / name
            output.mkdir(parents=True, exist_ok=True)
            loader = test_dataset(
                image_root=str(root / 'Tp'), gt_root=str(root / 'Gt'), text_root=str(root / opt.text_dir),
                bert_path=opt.bert_path, testsize=opt.testsize,
            )
            for _ in range(loader.size):
                image, gt, filename, (text_id,), (text_mask,), (text_type_id,), _ = loader.load_data()
                pred = model(image.cuda(), text_id.cuda(), text_mask.cuda(), text_type_id.cuda())[0]
                pred = F.interpolate(pred, size=(gt.height, gt.width), mode='bilinear', align_corners=False)
                mask = pred.sigmoid().cpu().numpy().squeeze()
                mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
                imageio.imwrite(output / filename, (mask * 255).astype(np.uint8))
            print(f'{name}: saved {loader.size} masks to {output}')


if __name__ == '__main__':
    main()
