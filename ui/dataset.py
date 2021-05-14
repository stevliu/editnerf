import random
import numpy as np

import torch
from torch import nn
from run_nerf_helpers import get_rays

import LBFGS


class NerfDataset():
    def __init__(self, images, poses, positive_masks, negative_masks, style, hwfs, device, edit_type, lr, randneg=0, N_rays=0, style_optimizer='adam', optimize_code=True, use_cached=False):
        self.images = images
        self.poses = poses
        self.hwfs = hwfs
        self.style = style.to(device)
        self.edit_type = edit_type
        self.use_cached = use_cached
        masks = (positive_masks + negative_masks).clamp_(0, 1)
        self.positive_masks = positive_masks
        self.negative_masks = negative_masks

        s = images.shape[1]
        # TODO: speed this up
        coords = [[(i, j) for i in range(s) for j in range(s) if masks[k][i][j].norm(p=2) > 1e-4] for k in range(images.shape[0])]
        for imgnum in range(len(coords)):
            if self.negative_masks[imgnum].sum() == 0:
                if randneg >= int(s * s):
                    coords[imgnum] = [(i, j) for i in range(s) for j in range(s)]
                elif randneg > 0:
                    coords[imgnum] += [(random.randint(0, s - 1), random.randint(0, s - 1)) for _ in range(randneg)]
        if N_rays > 0:
            for c in coords:
                random.shuffle(c)
            coords = [c[:N_rays] for c in coords]

        self.coords = [torch.tensor(x) for x in coords]
        self.d = self.style.shape[1] // 2
        self.optimize_code = optimize_code

        if optimize_code and self.edit_type == 'color':
            self.params = [nn.Parameter(self.style[:, self.d:])]
        elif optimize_code and self.edit_type == 'addition' or self.edit_type == 'removal':
            self.params = [nn.Parameter(self.style[:, :self.d])]

        self.optimizer_name = style_optimizer
        if style_optimizer == 'adam':
            self.style_optimizer = torch.optim.Adam(self.params, lr=lr)
        elif style_optimizer == 'lbfgs':
            self.style_optimizer = LBFGS.FullBatchLBFGS(self.params, lr)

        # TODO: extend this for multiple views
        self.shape_features = None
        self.color_features = None
        self.weights = None
        self.device = device

    def optimize_styles(self):
        if self.optimize_code:
            self.style_optimizer.step()
            self.style_optimizer.zero_grad()

    def get_data_batch(self, all_rays=False, imgnum=-1):
        # update parameters
        self.optimize_styles()

        if self.edit_type == 'color':
            self.style = torch.cat([self.style[:, :self.d], self.params[0]], dim=1)
        else:
            self.style = torch.cat([self.params[0], self.style[:, self.d:]], dim=1)

        if imgnum >= 0:
            img_i = imgnum
        else:
            img_i = np.random.choice(range(len(self.poses)))

        target = self.images[img_i].to(self.device)
        pose = self.poses[img_i, :3, :4].to(self.device)
        positive_mask = self.positive_masks[img_i].to(self.device)
        style = self.style[0].to(self.device).detach()

        H, W, focal = self.hwfs[img_i]
        H, W = int(H), int(W)
        rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

        coords = self.coords[img_i]

        if not all_rays:
            rand_idx = torch.randperm(coords.shape[0])
            rand_idx = rand_idx[:512]
            select_coords = coords[rand_idx]
        else:
            select_coords = coords

        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = torch.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        positive_mask_rays = (positive_mask[select_coords[:, 0], select_coords[:, 1]].sum(dim=1) > 0).float()

        if self.use_cached and self.shape_features is not None:
            shape_features = self.shape_features[img_i][rand_idx]
        else:
            shape_features = None

        if self.use_cached and self.color_features is not None:
            color_features = self.color_features[img_i][rand_idx]
        else:
            color_features = None

        if self.use_cached and self.weights is not None:
            weights = self.weights[img_i][rand_idx]
        else:
            weights = None

        style = [style for _ in range(rays_o.shape[0])]
        style = torch.stack(style)

        return batch_rays, target_s, style, positive_mask_rays, shape_features, color_features, weights

    def __len__(self):
        return len(self.poses)
