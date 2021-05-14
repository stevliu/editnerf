import os
import json

import torch
import numpy as np
import imageio
import torchvision


def trans_t(t): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_theta(th): return torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, trainskip=1, testskip=1, skip_val_test=False):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        trn_fname = os.path.join(basedir, 'transforms_{}.json'.format(s))
        with open(trn_fname, 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []

        if s == 'train':
            skip = max(trainskip, 1)
        else:
            skip = max(testskip, 1)
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            if skip_val_test and s in ('val', 'test'):
                # HACK: we don't have images for test/val views, but we'd at least like to see the rendered views
                imgs.append(np.zeros(all_imgs[-1][-1].shape))
            else:
                imgs.append(imageio.imread(fname, ignoregamma=True, pilmode='RGB'))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]

    if 'focal' not in meta:
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
    else:
        focal = meta['focal']

    return imgs, poses, [H, W, focal], i_split


def load_chairs(basedir, args):
    all_imgs = []
    all_poses = []
    all_i_split = [[], [], []]
    all_style_inds = []
    ref_imgs = []
    hwfs = []
    count = 0

    if args.real_image_dir:
        instance_names = [args.real_image_dir]
    else:
        with open(os.path.join(basedir, 'instances.txt')) as f:
            instances = [x.strip() for x in f.readlines()]
            instance_names = [os.path.join(basedir, instance_name) for instance_name in instances]
            if args.instance >= 0:
                instance_names = [instance_names[args.instance]]

    for instance in instance_names:
        imgs, poses, hwf, i_split = load_blender_data(instance, args.trainskip, args.testskip, skip_val_test=args.real_image_dir)
        hwfs += [hwf for _ in range(imgs.shape[0])]
        N_train, N_val, N_test = [len(x) for x in i_split]
        train, val, test = imgs[:N_train], imgs[N_train:N_train + N_val], imgs[N_train + N_val:N_train + N_val + N_test]
        train_poses, val_poses, test_poses = poses[:N_train], poses[N_train:N_train + N_val], poses[N_train + N_val:N_train + N_val + N_test]

        imgs = np.concatenate([train, val, test])
        poses = np.concatenate([train_poses, val_poses, test_poses])
        for i in range(3):
            all_i_split[i].append(count + i_split[i])
        all_style_inds.append(torch.zeros((imgs.shape[0])).long() + len(all_imgs))
        ref_imgs.append(imgs[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        if len(all_imgs) >= args.N_instances:
            break
        count += imgs.shape[0]

    for i in range(3):
        all_i_split[i] = np.concatenate(all_i_split[i], 0)

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    # View examples instances we're training on
    todir = os.path.join(args.basedir, args.savedir if args.savedir else args.expname)
    os.makedirs(todir, exist_ok=True)
    ref_imgs = torch.from_numpy(np.stack(ref_imgs, 0))[:128, :, :, :3].permute(0, 3, 1, 2)
    torchvision.utils.save_image(ref_imgs, os.path.join(todir, 'ref.png'))

    return imgs, poses, torch.tensor(hwfs), all_i_split, torch.cat(all_style_inds, dim=0)
