import numpy as np
import torch, os
from torch import nn

from load_blender import load_chairs
from run_nerf_helpers import get_rays, img2mse
from rendering import render
from utils import LBFGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

class NerfDataset():
    def __init__(self, images, poses, style, style_inds, i_train, hwfs, near_fars, device, args):
        self.images = images
        self.poses = poses
        self.style = style.to(device)
        self.learned_style = args.use_styles

        if args.N_viewdirs_reg > 0:
            self.raydirs = torch.stack([get_rays(int(hwf[0].item()), int(hwf[1].item()), hwf[2], pose.cuda())[1].cpu() for hwf, pose in zip(hwfs, poses[:, :3, :4])], 0)
            self.raydirs = self.raydirs[i_train] #train views only
            self.raydirs = self.raydirs.view(-1, 3) # [N*H*W, 3]
            rand_idx = torch.randperm(self.raydirs.shape[0])
            self.raydirs = self.raydirs[rand_idx]

        if self.learned_style:
            self.style = nn.Parameter(self.style)
            self.params = [self.style]
            self.lr = args.lrate 
            self.optimizer_name = args.style_optimizer
            if args.style_optimizer == 'adam':
                self.style_optimizer = torch.optim.Adam(self.params, lr=self.lr)
            elif args.style_optimizer == 'lbfgs':
                self.style_optimizer = LBFGS.FullBatchLBFGS(self.params, lr=self.lr)
            self.load_styles(args, os.path.join(args.basedir, args.expname))
            #TODO: add back lbfgs

        self.style_inds = style_inds
        self.i_train = i_train
        self.device = device
        self.hwfs = hwfs
        self.near_fars = near_fars

        self.i_batch = 0
        self.N_rand = args.N_rand
        self.N_viewdirs_reg = args.N_viewdirs_reg
        self.precrop_iters = args.precrop_iters #if not args.unseen else 0
        self.precrop_frac = args.precrop_frac 
        self.i = 0
        self.start = 0
                
    def get_features(self):
        return self.style

    def load_styles(self, args, chkpt_dir):
        if not os.path.exists(chkpt_dir) or args.real_image_dir: return 

        ckpts = [os.path.join(chkpt_dir, f) for f in sorted(os.listdir(chkpt_dir)) if 'tar' in f]

        if not args.no_reload and (args.load_it != 0 or len(ckpts) > 0):
            if args.load_it != 0: 
                ckpt_path = os.path.join(chkpt_dir, '{:06d}.tar'.format(args.load_it))
            else:
                ckpt_path = ckpts[-1]

            ckpt = torch.load(ckpt_path)
            self.style = ckpt['styles']
            self.params = [self.style]
            self.style_optimizer = torch.optim.Adam(self.params, lr=self.lr)
            self.style_optimizer.load_state_dict(ckpt['style_optimizer'])
            print('Loaded styles from', ckpt_path)
        else:
            print('No styles to load')

    def get_closure(self, train_fn, optimizer):
        optimizer.zero_grad()
        def fn():
            optimizer.zero_grad()
            self.style_optimizer.zero_grad()
            batch_rays, target_s, style, H, W, focal, near, far, _ = self.get_data_batch(optimize_style=False)
            train_fn.update({'near': near, 'far': far})
            rgb, disp, acc, extras = render(H, W, focal, style=style, rays=batch_rays, **train_fn)
            loss = img2mse(rgb, target_s) + img2mse(extras['rgb0'], target_s)
            return loss
        return fn

    def optimize_styles(self, train_fn=None, optimizer=None, loss=None):
        if self.learned_style:
            if self.optimizer_name == 'adam':
                self.style_optimizer.step()
                self.style_optimizer.zero_grad()
            elif self.optimizer_name == 'lbfgs':
                if loss is None:
                    loss = self.get_closure(train_fn, optimizer)()
                    loss.backward()
                else:
                    options = {'closure': self.get_closure(train_fn, optimizer), 'current_loss': loss, 'max_ls': 10, 'ls_debug': False}
                    loss = self.style_optimizer.step(options)[0]
                    if not isinstance(loss, torch.Tensor):
                        # LBFGS sometimes gets stuck and hangs forever. In this case, just reset the optimizer. 
                        print('Resetting optimizer')
                        self.style_optimizer = LBFGS.FullBatchLBFGS(self.params)
                
    def get_data_batch(self, optimize_style=True, **kwargs):
        if optimize_style:
            self.optimize_styles(**kwargs)

        # Random from one image
        img_i = np.random.choice(self.i_train)
        target = self.images[img_i].to(self.device)
        pose = self.poses[img_i, :3,:4].to(self.device)
        style = self.style[self.style_inds[img_i]].to(self.device)
        near, far = self.near_fars[img_i]

        if self.N_viewdirs_reg != 0:
            viewdirs_reg = self.raydirs[self.i_batch:self.i_batch + self.N_viewdirs_reg].to(self.device)
            self.i_batch += self.N_viewdirs_reg
            if self.i_batch >= self.raydirs.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(self.raydirs.shape[0])
                self.raydirs = self.raydirs[rand_idx]
                self.i_batch = 0
        else:
            viewdirs_reg = None
        
        if self.N_rand is not None:
            H, W, focal = self.hwfs[img_i]
            H, W = int(H), int(W)
            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            if self.i < self.precrop_iters:
                dH = int(H//2 * self.precrop_frac)
                dW = int(W//2 * self.precrop_frac)
                starth, endh, nbinsh = H//2 - dH, H//2 + dH - 1, 2*dH
                startw, endw, nbinsw = W//2 - dW, W//2 + dW - 1, 2*dW
                if self.i == self.start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {self.precrop_iters}")                
            else:
                starth, endh, nbinsh = 0, H-1, H
                startw, endw, nbinsw = 0, W-1, W
        
            coords = torch.stack(torch.meshgrid(
                        torch.linspace(starth, endh, nbinsh), 
                        torch.linspace(startw, endw, nbinsw)), -1).long()
            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            
            select_inds = np.random.choice(coords.shape[0], size=[self.N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)

            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            style = torch.stack([style for _ in range(rays_o.shape[0])])
            
        self.i += 1
        return batch_rays, target_s, style, H, W, focal, near, far, viewdirs_reg
    
def load_data(args):
    print('Loading data')
    images, poses, hwfs, i_split, style_inds = load_chairs(args.datadir, args)
    i_train, i_val, i_test = i_split

    near = args.blender_near
    far = args.blender_far

    near_fars = torch.zeros((images.shape[0], 2))
    near_fars[:, 0] = near
    near_fars[:, 1] = far

    if args.white_bkgd:
        print('Using whitening trick')
        assert images.shape[-1] == 4
        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    else:
        images = images[...,:3]
            
    print('Loaded', images.shape, args.datadir)

    bds_dict = {'near' : near, 'far' : far}
    N_instances = 1 if args.real_image_dir else args.N_instances
    style_vectors = torch.randn((N_instances, args.style_dim)).cuda() 
    
    # Move training data to GPU
    images = torch.tensor(images, device='cpu')
    poses = torch.tensor(poses, device='cpu')
    dataset = NerfDataset(images, poses, style_vectors, style_inds, i_train, hwfs, near_fars, device, args)
    return images, poses, dataset.get_features()[style_inds], i_test, i_train, bds_dict, dataset, hwfs, near_fars, style_inds
