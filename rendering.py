import os

import imageio
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

import torchvision
from run_nerf_helpers import get_rays, sample_pdf, to8b, to_disp_img, img2mse, mse2psnr


DEBUG = False


def render(H=None, W=None, focal=None, style=None, alpha=None, feature=None, weights=None, chunk=1024 * 32, rays=None, c2w=None, ndc=False,
           near=0., far=1., use_viewdirs=False, viewdirs_reg=None, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        style = style.repeat(rays_o.shape[0] * rays_o.shape[1], 1)
        if alpha is not None:
            # Color feature caching case
            alpha = alpha.view(style.shape[0], -1, 1)
        if weights is not None:
            weights = weights.view(-1, weights.shape[-1])
        if feature is not None:
            # shape feature caching case
            feature = feature.view(-1, feature.shape[2], feature.shape[3])
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        if viewdirs_reg is not None:
            viewdirs_reg = viewdirs_reg / torch.norm(viewdirs_reg, dim=-1, keepdim=True)
            viewdirs_reg = torch.reshape(viewdirs_reg, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    if isinstance(near, float) or isinstance(near, int) or len(near.shape) < 2:
        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])

    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, style, alpha, feature, weights, chunk, viewdirs_reg=viewdirs_reg, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)
    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_rays(ray_batch,
                style_batch,
                network_fn,
                network_query_fn,
                N_samples,
                alpha=None,
                feature=None,
                weights=None,
                lindisp=False,
                perturb_coarse=0.,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                viewdirs_reg=None,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb_coarse: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    t_vals = torch.linspace(0., 1., steps=N_samples)
    style_batch_coarse = style_batch.repeat([N_samples, 1])

    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb_coarse > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    if weights is not None and perturb_coarse == 0.:
        rgb_map_0 = None
    else:
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
        raw = network_query_fn(pts, style_batch_coarse, viewdirs, network_fn, None, None)
        rgb_map, disp_map, acc_map, weights, raws, features = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        raw_alpha = raws[..., 3]

        if viewdirs_reg is not None:
            # randomly pick some rays, and then randomly pick a point along each of those rays
            N_pts = pts.shape[0]
            N_views = viewdirs_reg.shape[0]
            rand_point = np.random.randint(0, pts.shape[1], size=(N_pts,))
            pts_reg_coarse = pts[torch.Tensor(list(range(N_pts))).long(), rand_point]
            pts_reg_coarse = pts_reg_coarse[None, :].expand((N_views, N_pts, 3))
            style_batch_reg = style_batch_coarse[:int(N_pts * N_views)]
            raws_reg = network_query_fn(pts_reg_coarse, style_batch_reg, viewdirs_reg, network_fn, None, None)
            raw_rgb_reg_coarse = torch.sigmoid(raws_reg[..., :3])
            rgb_variance_coarse = raw_rgb_reg_coarse.var(dim=0).sum(dim=1)  # [N_pts]

        if N_importance > 0:
            rgb_map_0, disp_map_0, acc_map_0, weights_0 = rgb_map, disp_map, acc_map, weights

    if N_importance > 0:
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        style_batch = torch.cat([style_batch_coarse, style_batch.repeat([N_importance, 1])])
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, style_batch, viewdirs, run_fn, alpha, feature)
        rgb_map, disp_map, acc_map, weights, raws, features = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        raw_alpha = raws[..., 3]

        if viewdirs_reg is not None:
            # randomly pick some rays, and then randomly pick a point along each of those rays
            N_pts = pts.shape[0]
            N_views = viewdirs_reg.shape[0]
            rand_point = np.random.randint(0, pts.shape[1], size=(N_pts,))
            pts_reg_fine = pts[torch.Tensor(list(range(N_pts))).long(), rand_point]
            pts_reg_fine = pts_reg_fine[None, :].expand((N_views, N_pts, 3))
            style_batch_reg = style_batch[:int(N_pts * N_views)]
            raws_reg = network_query_fn(pts_reg_fine, style_batch_reg, viewdirs_reg, run_fn, None, None)
            raw_rgb_reg = torch.sigmoid(raws_reg[..., :3])
            rgb_variance = raw_rgb_reg.var(dim=0)  # [N_pts, 3]
            rgb_variance = rgb_variance.sum(dim=1)  # [N_pts]

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'weights': weights}
    if features is not None:
        ret['raw_alpha'] = raw_alpha
        ret['features'] = features
    if N_importance > 0 and rgb_map_0 is not None:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['weights0'] = weights_0
    if viewdirs_reg is not None:
        ret['var0'] = rgb_variance_coarse
        ret['var'] = rgb_variance
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")
    return ret


def render_path(render_poses, styles, hwfs, chunk, render_kwargs, nfs=None, gt_imgs=None, alpha_cache=None, feature_cache=None, weights_cache=None, savedir=None, maximum=1000, get_cached=None, get_rgbs=False, verbose=True, cb=None, update_cb=None):
    render_kwargs['network_fine'].get_cached = get_cached
    rgbs = []
    disps = []
    alphas = []
    features = []
    weights = []

    total_psnr = 0.
    total_psnr0 = 0

    N = len(render_poses)
    s = N // maximum if len(render_poses) > maximum else 1
    if gt_imgs is not None:
        gt_imgs = gt_imgs[::s].cuda()
    render_poses = render_poses[::s].cuda()
    styles = styles[::s].cuda()
    hwfs = hwfs[::s].cuda()

    iterator = zip(render_poses, styles)
    if verbose:
        iterator = tqdm(iterator, total=len(styles))

    for i, (c2w, style) in enumerate(iterator):
        if cb is not None:
            cb(i)
        H, W, focal = hwfs[i]
        H, W = int(H), int(W)
        if nfs is not None:
            near, far = nfs[i]
            render_kwargs.update({'near': near, 'far': far})

        alpha = alpha_cache[i] if alpha_cache is not None else None
        feature = feature_cache[i] if feature_cache is not None else None
        weight = weights_cache[i] if weights_cache is not None else None
        rgb, disp, acc, additional = render(H, W, focal, style=style, chunk=chunk, weights=weight, c2w=c2w[:3, :4], alpha=alpha, feature=feature, **render_kwargs)

        if 'rgb0' in additional:
            rgb0 = additional['rgb0']

        if gt_imgs is not None:
            gt_img = gt_imgs[i]
            mse_loss = img2mse(rgb, gt_img)
            psnr = mse2psnr(mse_loss)
            total_psnr += psnr.item()
            if 'rgb0' in additional:
                mse_loss0 = img2mse(rgb0, gt_img)
                psnr0 = mse2psnr(mse_loss0)
                total_psnr0 += psnr0.item()

        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if update_cb:
            update_cb(i, rgbs[-1])

        if get_cached:
            alphas.append(additional['raw_alpha'])
            features.append(additional['features'])
            weights.append(additional['weights0'])

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            disp8 = to8b(to_disp_img(disps[-1]))
            imageio.imwrite(os.path.join(savedir, '{:04d}_rgb.png'.format(i)), rgb8)
            imageio.imwrite(os.path.join(savedir, '{:04d}_disp.png'.format(i)), disp8)
            if gt_imgs is not None:
                gt_img = to8b((gt_imgs[i]).cpu().numpy())
                imageio.imwrite(os.path.join(savedir, '{:04d}_gt.png'.format(i)), gt_img)

    if gt_imgs is not None:
        with open(os.path.join(savedir, 'log.txt'), 'a+') as f:
            torchvision.utils.save_image(torch.tensor(rgbs).cpu().permute(0, 3, 1, 2), 'rgbs.png')
            torchvision.utils.save_image(torch.tensor(gt_imgs).cpu().permute(0, 3, 1, 2), 'gt.png')
            msg = f'psnr0, psnr1, {total_psnr0/len(render_poses)}, {total_psnr/len(render_poses)}'
            f.write(msg + '\n')
            print(msg)

    total_psnr = total_psnr / len(rgbs)
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    if get_cached:
        render_kwargs['network_fine'].get_cached = None
        return rgbs, disps, total_psnr, alphas, features, weights
    else:
        return rgbs, disps, total_psnr


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - torch.exp(-act_fn(raw) * dists)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples] #last distance is placeholder
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.

    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std
        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    norm = torch.sum(weights, -1) + 1e-5
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / norm)
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    if raw.shape[-1] > 4:
        return rgb_map, disp_map, acc_map, weights, raw[..., :4], raw[..., 4:]
    else:
        return rgb_map, disp_map, acc_map, weights, raw, None


def batchify_rays(rays_flat, style, alpha, feature, weights, chunk=1024 * 32, N_samples=64, N_importance=0, viewdirs_reg=None, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}

    for i in range(0, rays_flat.shape[0], chunk):
        if alpha is not None:
            alpha_chunk = alpha[i:i + chunk]
        else:
            alpha_chunk = None
        if feature is not None:
            feature_chunk = feature[i:i + chunk]
        else:
            feature_chunk = None
        if weights is not None:
            weights_chunk = weights[i:i + chunk]
        else:
            weights_chunk = None
        ret = render_rays(rays_flat[i:i + chunk], style[i:i + chunk], alpha=alpha_chunk, feature=feature_chunk, weights=weights_chunk, N_samples=N_samples, N_importance=N_importance, viewdirs_reg=viewdirs_reg, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret
