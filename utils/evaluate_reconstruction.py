
from metrics import PSNR, SSIM, LPIPS
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


parser = argparse.ArgumentParser('Used for evaluating edit quality')
parser.add_argument('--expdir', help='where the generated samples are')
args = parser.parse_args()

expdir = [os.path.join(args.expdir, f) for f in sorted(os.listdir(args.expdir)) if 'test_imgs' in f][-1]
print(expdir)
trn = transforms.ToTensor()

psnr = PSNR(float)
ssim = SSIM(float)
lpips = LPIPS(float)


def load_generated_images(path, N=100):
    images = []
    for i in range(N):
        f = os.path.join(path, '{:04d}_rgb.png'.format(i))
        if not os.path.exists(f):
            return
        images.append(trn(Image.open(f)))
    return torch.stack(images)


def load_real_images(path, N=100):
    images = []
    for i in range(N):
        f = os.path.join(path, '{:04d}_gt.png'.format(i))
        if not os.path.exists(f):
            return
        images.append(trn(Image.open(f)))
    return torch.stack(images)


def get_metrics(fake, real):
    np_fake, np_real = fake.permute(0, 2, 3, 1).numpy().astype(np.float64), real.permute(0, 2, 3, 1).numpy().astype(np.float64)
    psnr_total = 0
    ssim_total = 0
    total = 0

    for x, y in zip(np_fake, np_real):
        psnr_total += psnr(x, y)
        ssim_total += ssim(x, y)
        total += 1

    return psnr_total / total, ssim_total / total, lpips(fake * 2 - 1, real * 2 - 1).mean().item()


generated = load_generated_images(expdir)
real = load_real_images(expdir)
psnr_num, ssim_num, lpips_num = get_metrics(generated, real)
msg = f'PSNR: {psnr_num} SSIM: {ssim_num} LPIPS: {lpips_num}'
with open(os.path.join(expdir, 'numbers.txt'), 'w') as f:
    f.write(msg)
print(msg)
