import os
from tqdm import tqdm

import numpy as np
import torch
from torchvision import utils

from inputs import config_parser
from model import create_nerf
from rendering import render_path


def load_dataset(instance, config, N_instances, num_canvases=9, expname=None, use_cached=True):
    parser = config_parser()
    if expname is None:
        args = parser.parse_args(['--config', config])
    else:
        args = parser.parse_args(['--config', config, '--expname', expname])
        N_instances = 1

    poses, hwfs = get_poses_hwfs(args, instance, N_instances, num_canvases=num_canvases)

    if use_cached:
        cache = {k: [] for k in ['alphas', 'features', 'weights']}

        # You can choose to save the cache on disk.
        # Saving the cache takes a while, and loading the cache takes a while the first time, but this is faster if you're editing the same instance frequently.

        # cache_dir = os.path.join(args.expname, 'cache')
        # if not os.path.exists(f'{cache_dir}/{instance}_{num_canvases-1}.pt'):
        #     alphas, features, weights = get_cache(config, instance, expname, save=True)
        #     cache['alphas'] = alphas[:num_canvases]
        #     cache['features'] = features[:num_canvases]
        #     cache['weights'] = weights[:num_canvases]
        # else:
        #     for j in range(len(poses)):
        #         data = torch.load(f'{cache_dir}/{instance}_{j}.pt')
        #         cache['alphas'].append(data['alphas'].cuda())
        #         cache['features'].append(data['features'].cuda())
        #         cache['weights'].append(data['weights'].cuda())

        # Computes and loads the cache. Computing the cache doesn't take noticably longer than rendering.
        alphas, features, weights = get_cache(config, instance, expname, num_canvases)
        cache['alphas'] = alphas[:num_canvases]
        cache['features'] = features[:num_canvases]
        cache['weights'] = weights[:num_canvases]
    else:
        cache = None

    return poses, hwfs, cache, args


def load_model(instance, config, expname=None):
    ''' returns: nerf model and style code '''
    parser = config_parser()
    if expname is None:
        args = parser.parse_args(['--config', config])
    else:
        args = parser.parse_args(['--config', config, '--expname', expname])
    render_kwargs_train, render_kwargs_test, _, _, optimizer, styles = create_nerf(args, return_styles=True)
    if styles.shape[0] == 1:
        basedir_args = parser.parse_args(['--config', config])
        basedir_styles = create_nerf(basedir_args, return_styles=True)[-1]
        styles = torch.cat([styles[:1], basedir_styles[1:]])
    return render_kwargs_train, render_kwargs_test, optimizer, styles


def get_cache(config, instance, expname=None, num_canvases=9, save=False):
    parser = config_parser()
    if expname is None:
        args = parser.parse_args(['--config', config])
        N_instances = None
    else:
        args = parser.parse_args(['--config', config, '--expname', expname])
        N_instances = 1
    cachedir = os.path.join(args.expname, 'cache')
    with torch.no_grad():
        render_kwargs_train, render_kwargs_test, _, _, optimizer, styles = create_nerf(args, return_styles=True)
        if N_instances is None:
            N_instances = styles.shape[0]
        nfs = [[args.blender_near, args.blender_far] for _ in range(10)]
        os.makedirs(cachedir, exist_ok=True)
        poses, hwfs = get_poses_hwfs(args, instance, N_instances, num_canvases)
        style = styles[instance].repeat((poses.shape[0], 1))
        rgbs, disps, _, alphas, features, weights = render_path(poses.cuda(), style, hwfs, args.chunk, render_kwargs_test, nfs=nfs, verbose=False, get_cached='color')
        if save:
            for j, (a, f, w) in enumerate(zip(alphas, features, weights)):
                torch.save({'alphas': a, 'features': f, 'weights': w}, f'{cachedir}/{instance}_{j}.pt')
        utils.save_image(torch.tensor(rgbs[:1]).permute(0, 3, 1, 2).cpu(), os.path.join(args.expname, '{:03d}.png'.format(instance)))
        return alphas, features, weights


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


def get_poses_hwfs(args, instance, N_instances, num_canvases=9):
    all_poses = torch.tensor(np.load(os.path.join(args.expname, 'poses.npy')))
    all_hwfs = torch.tensor(np.load(os.path.join(args.expname, 'hwfs.npy')))
    N_per_instance = all_poses.shape[0] // N_instances

    if N_per_instance == 1:
        # single view case, just find views along same elevation
        basepose = all_poses[instance].cpu()
        hwf = all_hwfs[instance]
        poses, hwfs = generate_flythrough(basepose, hwf)
        poses = poses[:num_canvases]
        hwfs = hwfs[:num_canvases]
    else:
        ps, pe = instance * N_per_instance, (instance + 1) * N_per_instance
        poses = all_poses[ps:pe][::4][:num_canvases]
        hwfs = all_hwfs[::4][:num_canvases]
    return poses, hwfs


def generate_flythrough(pose, hwf, num_poses=10):
    w2c = pose[:3, :3].T
    theta = np.arcsin(-w2c[0][0].item()) * 180 / np.pi
    phi = np.arcsin(w2c[1][2].item()) * 180 / np.pi
    rho = np.linalg.norm(pose[:3, 3])
    poses = [pose_spherical(theta - 90 + dtheta, phi - 90, rho) for dtheta in np.linspace(0, 360, num_poses)]
    return torch.stack(poses), torch.stack([hwf for _ in range(num_poses)])


def load_config(config):
    parser = config_parser()
    return parser.parse_args(['--config', config])


def transfer_codes(config):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    N_per_transfer = 8
    N_transfer = 10

    parser = config_parser()
    args = parser.parse_args(['--config', config])
    render_kwargs_train, render_kwargs_test, _, _, optimizer, styles = create_nerf(args, return_styles=True)
    nfs = [[args.blender_near, args.blender_far] for _ in range(N_per_transfer)]
    os.makedirs(f'{args.expname}/transfer_codes', exist_ok=True)

    with torch.no_grad():
        for i1 in range(N_transfer):
            for i2 in tqdm(range(N_transfer)):
                if i1 == i2:
                    continue

                s1, s2 = styles[i1].unsqueeze(dim=0), styles[i2].unsqueeze(dim=0)
                poses_1, hwfs_1 = get_poses_hwfs(args, i1, styles.shape[0], num_canvases=N_per_transfer)
                poses_2, hwfs_2 = get_poses_hwfs(args, i2, styles.shape[0], num_canvases=N_per_transfer)
                rgb1 = render_path(poses_1, s1, hwfs_1, 4096, render_kwargs_test, nfs=nfs, verbose=False)[0][0]
                rgb2 = render_path(poses_2, s2, hwfs_2, 4096, render_kwargs_test, nfs=nfs, verbose=False)[0][0]
                utils.save_image(torch.tensor(rgb1).permute(2, 0, 1), f'{args.expname}/transfer_codes/{i1}.png')
                utils.save_image(torch.tensor(rgb2).permute(2, 0, 1), f'{args.expname}/transfer_codes/{i2}.png')

                take_color = torch.cat([s1[:, :32], s2[:, 32:]], dim=1)
                take_shape = torch.cat([s2[:, :32], s1[:, 32:]], dim=1)
                # i1 with shape from i2 is the same as i2 color from i1, so don't duplicate
                color_from = render_path(poses_1, take_color.repeat((N_per_transfer, 1)), hwfs_1, 4096, render_kwargs_test, nfs=nfs, verbose=False)[0]
                shape_from = render_path(poses_2, take_shape.repeat((N_per_transfer, 1)), hwfs_2, 4096, render_kwargs_test, nfs=nfs, verbose=False)[0]
                utils.save_image(torch.tensor(color_from).permute(0, 3, 1, 2), f'{args.expname}/transfer_codes/{i1}_color_from_{i2}.png')
                utils.save_image(torch.tensor(shape_from).permute(0, 3, 1, 2), f'{args.expname}/transfer_codes/{i1}_shape_from_{i2}.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()

    transfer_codes(args.config)
