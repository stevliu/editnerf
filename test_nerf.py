import random

import torch
import os
import numpy as np

from rendering import render_path
from dataset import load_data
from inputs import config_parser
from model import create_nerf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


def test():
    parser = config_parser()
    args = parser.parse_args()

    images, poses, style, i_test, i_train, bds_dict, dataset, hwfs, near_fars, _ = load_data(args)
    images_test, poses_test, style_test, hwfs_test, nf_test = images[i_test], poses[i_test], style[i_test], hwfs[i_test], near_fars[i_test]
    images_train, poses_train, style_train, hwfs_train, nf_train = images[i_train], poses[i_train], style[i_train], hwfs[i_train], near_fars[i_train]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    np.save(os.path.join(basedir, expname, 'poses.npy'), poses_train.cpu())
    np.save(os.path.join(basedir, expname, 'hwfs.npy'), hwfs_train.cpu())

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    with torch.no_grad():
        if args.render_test:
            if args.shuffle_poses:
                print('Shuffling test poses')
                permutation = list(range(len(poses_test)))
                random.shuffle(permutation)
                poses_test = poses_test[permutation]
            testsavedir = os.path.join(basedir, expname, 'test_imgs{:06d}'.format(start))
            os.makedirs(testsavedir, exist_ok=True)
            _, _, psnr = render_path(poses_test.to(device), style_test, hwfs_test, args.chunk, render_kwargs_test, nfs=nf_test, gt_imgs=images_test, savedir=testsavedir)
            print('Saved test set w/ psnr', psnr)

        if args.render_train:
            if args.shuffle_poses:
                print('Shuffling train poses')
                permutation = list(range(len(poses_train)))
                random.shuffle(permutation)
                poses_train = poses_train[permutation]
            trainsavedir = os.path.join(basedir, expname, 'train_imgs{:06d}'.format(start))
            os.makedirs(trainsavedir, exist_ok=True)
            _, _, psnr = render_path(poses_train.to(device), style_train, hwfs_train, args.chunk, render_kwargs_test, nfs=nf_train, gt_imgs=images_train, savedir=trainsavedir)
            print('Saved train set w/ psnr', psnr)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    test()
