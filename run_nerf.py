import os
import copy

import numpy as np
import torch

from run_nerf_helpers import img2mse, mse2psnr

from inputs import config_parser
from dataset import load_data
from model import create_nerf
from rendering import render, render_path
from utils.pidfile import exit_if_job_done, mark_job_done

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)


def train():
    parser = config_parser()
    args = parser.parse_args()

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.savedir if args.savedir else args.expname
    print('Experiment dir:', expname)

    # Load data
    images, poses, style, i_test, i_train, bds_dict, dataset, hwfs, near_fars, style_inds = load_data(args)
    _, poses_test, style_test, hwfs_test, nf_test = images[i_test], poses[i_test], style[i_test], hwfs[i_test], near_fars[i_test]
    _, poses_train, style_train, hwfs_train, nf_train = images[i_train], poses[i_train], style[i_train], hwfs[i_train], near_fars[i_train]

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    np.save(os.path.join(basedir, expname, 'poses.npy'), poses_train.cpu())
    np.save(os.path.join(basedir, expname, 'hwfs.npy'), hwfs_train.cpu())
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    print(render_kwargs_train['network_fine'])
    old_coarse_network = copy.deepcopy(render_kwargs_train['network_fn']).state_dict()
    old_fine_network = copy.deepcopy(render_kwargs_train['network_fine']).state_dict()

    global_step = start
    real_image_application = (args.real_image_dir is not None)
    optimize_mlp = not real_image_application
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    loss = None

    if start == 0:
        # if we're starting from scratch, delete all the logs in that directory.
        if os.path.exists(os.path.join(basedir, expname, 'log.txt')):
            os.remove(os.path.join(basedir, expname, 'log.txt'))
    start = start + 1

    for i in range(start, args.n_iters + 1):
        # Sample random ray batch
        batch_rays, target_s, style, H, W, focal, near, far, viewdirs_reg = dataset.get_data_batch(train_fn=render_kwargs_train, optimizer=optimizer, loss=loss)
        render_kwargs_train.update({'near': near, 'far': far})

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, focal, style=style, chunk=args.chunk, rays=batch_rays, viewdirs_reg=viewdirs_reg, **render_kwargs_train)
        optimizer.zero_grad()

        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if args.var_param > 0:
            var = extras['var']
            var0 = extras['var0']
            var_loss = var.mean(dim=0)
            var_loss_coarse = var0.mean(dim=0)

            loss += args.var_param * var_loss
            loss += args.var_param * var_loss_coarse
            var_loss = var_loss.item()
            var_loss_coarse = var_loss_coarse.item()
        else:
            var_loss = 0
            var_loss_coarse = 0

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0).item()
        else:
            psnr0 = -1

        if args.weight_change_param >= 0:
            weight_change_loss_coarse = 0.
            for k, v in render_kwargs_train['network_fn'].named_parameters():
                if 'weight' in k:
                    diff = (old_coarse_network[k] - v).pow(2).mean()
                    weight_change_loss_coarse += diff
            weight_change_loss_fine = 0.
            for k, v in render_kwargs_train['network_fine'].named_parameters():
                if 'weight' in k:
                    diff = (old_fine_network[k] - v).pow(2).mean()
                    weight_change_loss_fine += diff
            weight_change_loss = weight_change_loss_coarse + weight_change_loss_fine
            loss = loss + args.weight_change_param * weight_change_loss
        else:
            weight_change_loss = torch.tensor(0.)

        loss.backward()
        if optimize_mlp:
            optimizer.step()

        # NOTE: IMPORTANT!
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################
        #####           end            #####

        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            state_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'styles': dataset.style,
                'style_optimizer': dataset.style_optimizer.state_dict()
            }
            if args.N_importance > 0:
                state_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()
            torch.save(state_dict, path)
            print('Saved checkpoints at', path)

        if i % args.i_testset == 0 and i > 0:
            if real_image_application:
                style_test = dataset.get_features().repeat((poses_test.shape[0], 1))
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                render_path(poses_test.to(device), style_test, hwfs_test, args.chunk, render_kwargs_test, nfs=nf_test, savedir=testsavedir, maximum=100)
            print('Saved test set')

        if i % args.i_trainset == 0 and i > 0:
            if real_image_application:
                style_train = dataset.get_features().repeat((poses_train.shape[0], 1))
            trainsavedir = os.path.join(basedir, expname, 'trainset_{:06d}'.format(i))
            os.makedirs(trainsavedir, exist_ok=True)
            with torch.no_grad():
                render_path(poses_train.to(device), style_train, hwfs_train, args.chunk, render_kwargs_test, nfs=nf_train, savedir=trainsavedir, maximum=100)
            print('Saved train set')

        if i % args.i_print == 0 or i == 1:
            log_str = f"[TRAIN] Iter: {i} Loss: {loss.item()} PSNR: {psnr.item()} PSNR0: {psnr0} Var loss: {var_loss} Var loss coarse: {var_loss_coarse} Weight change loss: {weight_change_loss}"
            with open(os.path.join(basedir, expname, 'log.txt'), 'a+') as f:
                f.write(log_str + '\n')
            print(log_str)

        global_step += 1

        if real_image_application and global_step - start == args.n_iters_real:
            return

        if real_image_application and global_step - start == args.n_iters_code_only:
            optimize_mlp = True
            dataset.optimizer_name = 'adam'
            dataset.style_optimizer = torch.optim.Adam(dataset.params, lr=dataset.lr)
            print('Starting to jointly optimize weights with code')


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    if args.instance != -1:
        # Allows for scripting over single instance experiments.
        exit_if_job_done(os.path.join(args.basedir, args.expname))
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        train()
        mark_job_done(os.path.join(args.basedir, args.expname))
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        train()
