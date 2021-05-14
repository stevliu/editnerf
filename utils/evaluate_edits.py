from metrics import PSNR, SSIM, LPIPS
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import torch
import os
import json
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


parser = argparse.ArgumentParser('Used for evaluating edit quality')
parser.add_argument('--dataset', choices=['photoshapes', 'dosovitskiy_chairs'], help='dataset to edit')
parser.add_argument('--expdir', help='where the generated samples are')
parser.add_argument('--src', type=int, help='original instance number')
parser.add_argument('--tgt', type=int, help='target instance number')
args = parser.parse_args()

N_imgs = {'dosovitskiy_chairs': 36, 'photoshapes': 40}[args.dataset]
datadir = os.path.join('data', args.dataset)
with open(os.path.join(datadir, 'instances.txt')) as f:
    instances = [x.strip() for x in f.readlines()]
src_instance_name = instances[args.src]
tgt_instance_name = instances[args.tgt]
expdir = args.expdir
trn = transforms.ToTensor()


def load_generated_images(path):
    images = []
    for i in range(N_imgs):
        f = '{:04d}_rgb.png'.format(i)
        images.append(trn(Image.open(os.path.join(path, f))))
    return torch.stack(images)


def load_real_images():
    images = []
    src_dir = os.path.join(datadir, src_instance_name)
    with open(os.path.join(src_dir, 'transforms_train.json')) as f:
        data_src = [x['file_path'] for x in json.load(f)['frames']]
    tgt_dir = os.path.join(datadir, tgt_instance_name)
    # HACK: if the pose nums match up, then the poses are also identical
    if args.dataset == 'photoshapes':
        for name in data_src:
            pose_num = name.split('_')[-1]
            for prefix in ['train', 'val', 'test']:
                # try to find the render in train/test/val/current folder
                fp = os.path.join(tgt_dir, prefix, f'{tgt_instance_name}_{pose_num}.png')
                if os.path.exists(fp):
                    images.append(trn(Image.open(fp)))
    elif args.dataset == 'dosovitskiy_chairs':
        for name in data_src:
            fp = os.path.join(tgt_dir, f'{name}.png')
            if os.path.exists(fp):
                images.append(trn(Image.open(fp)))
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


generated_images = load_generated_images(expdir)
real_images = load_real_images()

assert generated_images.shape[0] == real_images.shape[0]

psnr = PSNR(float)
ssim = SSIM(float)
lpips = LPIPS(float)

utils.save_image(generated_images, os.path.join(expdir, 'gen.png'))
utils.save_image(real_images, os.path.join(expdir, 'reals.png'))
psnr_num, ssim_num, lpips_num = get_metrics(generated_images, real_images)
msg = f'PSNR: {psnr_num} SSIM: {ssim_num} LPIPS: {lpips_num}'
with open(os.path.join(expdir, 'numbers.txt'), 'w') as f:
    f.write(msg)
print(msg)
