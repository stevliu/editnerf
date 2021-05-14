import copy
import os

import torch
import numpy as np
import imageio
from torch.nn import functional as F
from torchvision import utils, transforms
from PIL import Image

from ui_utils import renormalize, show, labwidget, paintwidget, mean_colors
from rendering import render_path, render
from editing_utils import load_dataset, load_model, load_config, generate_flythrough
from dataset import NerfDataset
from run_nerf_helpers import img2mse, get_rays, to8b, to_disp_img

##########################################################################
# UI
##########################################################################

IMG_SIZE = 128
VERBOSE = True
N_ITERS = {'color': 100, 'removal': 100, 'addition': 10000}
LR = 0.001
N_RAYS = {'color': 64, 'removal': 8192}


class NeRFEditingApp(labwidget.Widget):
    def __init__(self, instance, config, use_cached=True, expname=None, edit_type=None, num_canvases=9, shape_params='fusion_shape_branch', color_params='color_branch', randneg=8192, device='cuda:0'):
        super().__init__(style=dict(border="3px solid gray", padding="8px", display="inline-block"))
        torch.set_default_tensor_type('torch.cuda.FloatTensor' if device == 'cuda:0' else 'cpu')
        self.edit_type = edit_type
        self.instance = instance
        self.num_canvases = num_canvases
        self.shape_params = shape_params
        self.color_params = color_params
        self.size = IMG_SIZE
        self.randneg = randneg
        self.device = device
        self.msg_out = labwidget.Div()
        self.editing_canvas = paintwidget.PaintWidget(image='', width=self.size * 3, height=self.size * 3).on('mask', self.change_mask)
        self.editing_canvas.index = -1
        self.copy_canvas = paintwidget.PaintWidget(image='', width=self.size * 2, height=self.size * 2).on('mask', self.copy)
        self.copy_mask = None
        inline = dict(display='inline', border="2px solid gray")

        self.toggle_rgbs_disps_btn = labwidget.Button('show depth', style=inline).on('click', self.toggle_rgb_disps)
        self.positive_mask_btn = labwidget.Button(self.pad('edit color'), style=inline).on('click', self.positive_mask)
        self.addition_mask_btn = labwidget.Button(self.pad('add shape'), style=inline).on('click', self.add)
        self.sigma_mask_btn = labwidget.Button(self.pad('remove shape'), style=inline).on('click', self.sigma_mask)
        self.color_from_btn = labwidget.Button(self.pad('transfer color'), style=inline).on('click', self.color_from)
        self.shape_from_btn = labwidget.Button(self.pad('transfer shape'), style=inline).on('click', self.shape_from)
        self.execute_btn = labwidget.Button(self.pad('execute'), style=inline).on('click', self.execute_edit)
        self.brushsize_textbox = labwidget.Textbox(5, desc='brushsize: ', size=3).on('value', self.change_brushsize)

        self.target = None
        self.use_color_cache = True

        self.color_style = dict(display='inline', border="2px solid white")
        trn = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        bg_img = trn(Image.open('bg.png').convert('RGB'))
        bg_img = renormalize.as_url(bg_img * 2 - 1)
        self.color_pallete = [labwidget.Image(src=bg_img, style=self.color_style).on('click', self.set_color)]
        self.color_pallete[-1].index = 0
        self.color_pallete[-1].color_type = 'bg'

        for color in mean_colors.colors.values():
            image = torch.zeros(3, 32, 32)
            image[0, :, :] = color[0]
            image[1, :, :] = color[1]
            image[2, :, :] = color[2]
            image = image / 255. * 2 - 1
            self.color_pallete.append(labwidget.Image(src=renormalize.as_url(image), style=self.color_style).on('click', self.set_color))
            self.color_pallete[-1].index = len(self.color_pallete) - 1
            self.color_pallete[-1].color_type = 'color'
            # TODO: Highlight the white box with black for clarity

        self.color = None
        self.mask_type = None
        self.real_canvas_array = []
        self.real_images_array = []
        self.positive_masks = []

        train, test, optimizer, styles = load_model(instance, config, expname=expname)
        poses, hwfs, cache, args = load_dataset(instance, config, num_canvases=num_canvases, N_instances=styles.shape[0], expname=expname, use_cached=use_cached)
        self.parentdir = load_config(config).expname
        self.expname = expname if expname else self.parentdir
        self.savedir = os.path.join(self.expname, str(instance))
        os.makedirs(self.savedir, exist_ok=True)
        self.poses = poses.to(device)
        self.cache = cache
        self.chunk = args.chunk
        self.near = args.blender_near
        self.far = args.blender_far
        self.nfs = [[self.near, self.far]] * self.poses.shape[0]
        self.hwfs = hwfs.to(device)
        self.old_fine_network = dict(copy.deepcopy(test['network_fine']).named_parameters())
        self.train_kwargs = train
        self.test_kwargs = test
        self.optimizer = None
        self.all_instance_styles = styles
        self.instance_style = styles[instance].unsqueeze(dim=0).to(device)

        if cache is not None:
            self.weights = cache['weights']
            self.alphas = cache['alphas']
            self.features = cache['features']
        else:
            self.weights = None
            self.alphas = None
            self.features = None

        self.trn = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
        self.transfer_instances_array = [labwidget.Image(src='').on('click', self.change_target) for _ in range(12)]
        self.addition_instances_array = [labwidget.Image(src='').on('click', self.change_target) for _ in range(12)]
        images, disps = self.render(self.poses, self.instance_style, verbose=False, get_disps=True)
        for i, image in enumerate(images):
            resized = F.interpolate(image.unsqueeze(dim=0), size=(self.size, self.size)).squeeze(dim=0)
            disp_img = torch.from_numpy(to8b(to_disp_img(disps[i]))).unsqueeze(dim=0) / 255.
            resized_disp = F.interpolate(disp_img.unsqueeze(dim=0), size=(self.size, self.size)).squeeze(dim=0)
            self.real_images_array.append(labwidget.Image(
                src=renormalize.as_url(resized)).on('click', self.set_editing_canvas))
            self.real_images_array[-1].index = i
            self.real_canvas_array.append(paintwidget.PaintWidget(
                image=renormalize.as_url(image),
                width=self.size * 3, height=self.size * 3).on('mask', self.change_mask))
            self.real_canvas_array[-1].index = i
            self.real_canvas_array[-1].negative_mask = ''
            self.real_canvas_array[-1].resized_image = renormalize.as_url(resized)
            self.real_canvas_array[-1].resized_disp = renormalize.as_url(resized_disp)
            self.real_canvas_array[-1].disp = renormalize.as_url(disp_img)
            self.real_canvas_array[-1].orig = renormalize.as_url(image)
            self.positive_masks.append(torch.zeros(image.shape).cpu())
        self.show_rgbs = True

        self.change_brushsize()
        self.editname_textbox = labwidget.Datalist(choices=self.saved_names(), style=inline)
        self.save_btn = labwidget.Button('save', style=inline).on('click', self.save)
        self.load_btn = labwidget.Button('load', style=inline).on('click', self.load)

    def pad(self, s, total=14):
        white = ' ' * ((total - len(s)) // 2)
        return white + s + white

    def make_trasparent(self):
        for button in [self.sigma_mask_btn, self.positive_mask_btn, self.addition_mask_btn, self.color_from_btn, self.shape_from_btn]:
            button.style = {'display': 'inline', 'color': 'grey', 'border': "1px solid grey"}

    def negative_mask(self):
        self.mask_type = 'negative'
        if self.editing_canvas.image != '':
            self.editing_canvas.mask = self.real_canvas_array[self.editing_canvas.index].negative_mask

    def positive_mask(self):
        self.mask_type = 'positive'
        self.make_trasparent()
        self.positive_mask_btn.style = {'display': 'inline', 'color': 'black', 'border': "2px solid black"}
        self.editing_canvas.mask = ''

    def sigma_mask(self):
        self.mask_type = 'sigma'
        self.make_trasparent()
        self.sigma_mask_btn.style = {'display': 'inline', 'color': 'black', 'border': "2px solid black"}
        self.editing_canvas.mask = ''

    def from_editing_canvas(self):
        self.real_canvas_array[self.editing_canvas.index].image = self.editing_canvas.image

    def update_canvas(self, images, disps=None):
        for i, image in enumerate(images):
            resized_rgb = F.interpolate(image.unsqueeze(dim=0), size=(self.size, self.size)).squeeze(dim=0)
            self.real_images_array[i].src = renormalize.as_url(resized_rgb)
            self.real_canvas_array[i].image = renormalize.as_url(image)
            self.real_canvas_array[i].resized_image = renormalize.as_url(resized_rgb)
            if disps is not None:
                disp_img = torch.from_numpy(to8b(to_disp_img(disps[i]))).unsqueeze(dim=0) / 255.
                resized_disp = F.interpolate(disp_img.unsqueeze(dim=0), size=(self.size, self.size)).squeeze(dim=0)
                self.real_canvas_array[i].resized_disp = renormalize.as_url(resized_disp)
                self.real_canvas_array[i].disp = renormalize.as_url(disp_img)

        if self.editing_canvas.index >= 0:
            self.editing_canvas.image = self.real_canvas_array[self.editing_canvas.index].image

    def toggle_rgb_disps(self):
        self.show_rgbs = not self.show_rgbs
        for i in range(len(self.real_canvas_array)):
            if self.show_rgbs:
                self.real_images_array[i].src = self.real_canvas_array[i].resized_image
            else:
                self.real_images_array[i].src = self.real_canvas_array[i].resized_disp
        if self.show_rgbs:
            self.toggle_rgbs_disps_btn.label = 'show depth'
        else:
            self.toggle_rgbs_disps_btn.label = 'show rgbs'

    def set_color(self, evt):
        for i in range(len(self.color_pallete)):
            self.color_pallete[i].style = {'display': 'inline', 'border': "2px solid white"}
        evt.target.style = {'display': 'inline', 'border': "1px solid black"}
        if evt.target.color_type == 'bg':
            self.negative_mask()
        else:
            image = renormalize.from_url(evt.target.src) / 2 + 0.5
            image = image * 255
            self.color = [int(x) * 2 / 255. - 1 for x in image[:, 0, 0]]
            color = torch.zeros((3, self.size * 2, self.size * 2)).cpu()
            color[0, :, :] = self.color[0]
            color[1, :, :] = self.color[1]
            color[2, :, :] = self.color[2]
            self.color = color

    def change_brushsize(self):
        brushsize = int(self.brushsize_textbox.value)
        for c in self.real_canvas_array:
            c.brushsize = brushsize
        self.editing_canvas.brushsize = brushsize
        self.copy_canvas.brushsize = brushsize

    def set_editing_canvas(self, evt):
        self.editing_canvas.image = self.real_canvas_array[evt.target.index].image
        self.editing_canvas.index = self.real_canvas_array[evt.target.index].index
        if self.mask_type == 'negative':
            self.editing_canvas.mask = self.real_canvas_array[evt.target.index].negative_mask
        else:
            self.editing_canvas.mask = ''

    def add(self, ev):
        self.edit_type = 'addition'
        self.make_trasparent()
        self.addition_mask_btn.style = {'display': 'inline', 'color': 'black', 'border': "2px solid black"}
        self.display_addition_instance()

    def color_from(self, ev):
        self.edit_type = 'color_from'
        self.make_trasparent()
        self.color_from_btn.style = {'display': 'inline', 'color': 'black', 'border': "2px solid black"}
        self.display_transfer_instance()

    def shape_from(self, ev):
        self.edit_type = 'shape_from'
        self.make_trasparent()
        self.shape_from_btn.style = {'display': 'inline', 'color': 'black', 'border': "2px solid black"}
        self.display_transfer_instance()

    def display_transfer_instance(self):
        for i in range(12):
            self.transfer_instances_array[i].src = renormalize.as_url(self.trn(Image.open(os.path.join(self.parentdir, 'instances', '{:03d}.png'.format(i)))) * 2 - 1)
            self.transfer_instances_array[i].index = i

    def display_addition_instance(self):
        for i in range(12):
            self.addition_instances_array[i].src = renormalize.as_url(self.trn(Image.open(os.path.join(self.parentdir, 'instances', '{:03d}.png'.format(i)))) * 2 - 1)
            self.addition_instances_array[i].index = i

    def render(self, poses, style, verbose=True, get_disps=False, update=False, update_cache=True, inds=None, use_cache=True):
        def cb(i):
            if verbose and VERBOSE:
                self.msg_out.print(f'Rendering view: {i+1}/{len(poses)}', replace=True)

        def update_cb(i, rgb):
            if update:
                img = torch.tensor(rgb).permute(2, 0, 1) * 2 - 1
                resized = F.interpolate(img.unsqueeze(dim=0), size=(self.size, self.size)).squeeze(dim=0)
                self.real_images_array[i].src = renormalize.as_url(resized)
            else:
                pass

        with torch.no_grad():
            styles = style.repeat((poses.shape[0], 1))
            if self.use_color_cache and use_cache and self.alphas and self.features and self.weights:
                if inds:
                    alpha_cache = [self.alphas[i] for i in inds]
                    feature_cache = [self.features[i] for i in inds]
                    weights_cache = [self.weights[i] for i in inds]
                else:
                    alpha_cache = self.alphas
                    feature_cache = self.features
                    weights_cache = self.weights
                images, disps, _, = render_path(poses, styles, self.hwfs, self.chunk, self.test_kwargs, nfs=self.nfs, alpha_cache=alpha_cache, feature_cache=feature_cache, weights_cache=weights_cache, verbose=False, cb=cb, update_cb=update_cb)
            else:
                images, disps, _, alphas, features, weights = render_path(poses, styles, self.hwfs, self.chunk, self.test_kwargs, nfs=self.nfs, verbose=False, cb=cb, update_cb=update_cb, get_cached='color')
                if update_cache:
                    self.alphas = alphas
                    self.features = features
                    self.weights = weights

            images = torch.tensor(images).permute(0, 3, 1, 2) * 2 - 1
            if get_disps:
                return images, disps
            else:
                return images

    def target_transfer(self, instancenum, index):
        self.copy_canvas.mask = ''
        self.copy_canvas.index = index
        self.copy_canvas.instance_style = self.all_instance_styles[instancenum].unsqueeze(dim=0)
        rgb = self.render(self.poses[index].unsqueeze(dim=0), self.copy_canvas.instance_style.squeeze(dim=0), verbose=False, use_cache=False)
        self.copy_canvas.image = renormalize.as_url(F.interpolate(rgb, size=(self.size, self.size))[0])

    def change_mask(self, ev):
        if self.mask_type == 'positive' or self.mask_type == 'sigma':
            i = self.editing_canvas.index
            orig_img = renormalize.from_url(self.editing_canvas.image)
            mask = renormalize.from_url(self.editing_canvas.mask) / 2 + 0.5
            mask = F.interpolate(mask.unsqueeze(dim=0), size=(self.size * 2, self.size * 2)).squeeze()
            if self.mask_type == 'positive':
                self.edit_type = 'color'
                if self.color is None:
                    self.show_msg('Please select a color.')
                    if ev.target.image != '':
                        self.real_canvas_array[ev.target.index].negative_mask = ''
                    return
                edited_img = orig_img * (1 - mask) + mask * self.color
            elif self.mask_type == 'sigma':
                self.edit_type = 'removal'
                edited_img = orig_img * (1 - mask) + mask * torch.ones((3, self.size * 2, self.size * 2)).to(mask.device)
            self.positive_masks[i] += mask
            self.real_canvas_array[i].image = renormalize.as_url(edited_img)
            self.editing_canvas.image = renormalize.as_url(edited_img)
            self.real_images_array[i].src = renormalize.as_url(F.interpolate(edited_img.unsqueeze(dim=0), size=(self.size, self.size)).squeeze())
            self.editing_canvas.mask = ''
        elif self.mask_type == 'negative':
            i = ev.target.index
            self.real_canvas_array[i].negative_mask = self.editing_canvas.mask
        elif self.copy_mask is not None:
            self.paste()
        else:
            if ev.target.image != '':
                self.real_canvas_array[ev.target.index].negative_mask = ''

    def render_editing_canvas(self, style):
        index = self.editing_canvas.index
        pose = self.poses[index].unsqueeze(dim=0)
        self.editing_canvas.image = renormalize.as_url(self.render(pose, style, verbose=False, inds=[index], use_cache=self.edit_type == 'color_from', update_cache=False)[0])

    def change_target(self, ev):
        self.target = ev.target.index
        if self.edit_type == 'color_from':
            target_style = self.all_instance_styles[self.target].unsqueeze(dim=0).cuda()
            new_style = torch.cat([self.instance_style[:, :32], target_style[:, 32:]], dim=1)
            self.render_editing_canvas(new_style)
        elif self.edit_type == 'shape_from':
            target_style = self.all_instance_styles[self.target].unsqueeze(dim=0).cuda()
            new_style = torch.cat([target_style[:, :32], self.instance_style[:, 32:]], dim=1)
            self.render_editing_canvas(new_style)
        elif self.edit_type == 'addition':
            if self.editing_canvas.image != '':
                self.target_transfer(self.target, self.editing_canvas.index)

    def copy(self, ev):
        self.copy_mask = self.copy_canvas.mask
        tgt_style = self.copy_canvas.instance_style
        index = self.copy_canvas.index
        area = renormalize.from_url(self.copy_mask, target='pt', size=(256, 256))[0]
        t, l, b, r = positive_bounding_box(area)
        H, W, focal = self.hwfs[0]
        H, W = H.item(), W.item()

        with torch.no_grad():
            rays_o, rays_d = get_rays(int(H), int(W), focal, self.poses[index])
            rays_o, rays_d = rays_o[t:b, l:r], rays_d[t:b, l:r]
            rays_o, rays_d = rays_o.contiguous().view(-1, rays_o.shape[-1]), rays_d.contiguous().view(-1, rays_d.shape[-1])
            batch_rays = torch.stack([rays_o, rays_d], 0)
            # render the rays under the editing canvas color style
            style = torch.cat([tgt_style[:, :32], self.instance_style[:, 32:]], dim=1)
            style = style.repeat((batch_rays.shape[1], 1))
            rgb, disp, acc, extras = render(H, W, focal.item(), style=style, rays=batch_rays, **self.test_kwargs)

        self.copy_canvas.rgb = rgb.view(b - t, r - l, -1).cpu() * 2 - 1
        self.copy_canvas.mask = ''

    def paste(self):
        if self.copy_mask is None:
            self.show_msg('Please select a region to copy first.')
            return

        copy_to = renormalize.from_url(self.editing_canvas.mask, target='pt', size=(256, 256))[0]
        area = renormalize.from_url(self.copy_mask, target='pt', size=(256, 256))[0]
        t, l, b, r = positive_bounding_box(area)
        area = area[t:b, l:r]

        target_rgb = self.copy_canvas.rgb
        source_rgb = renormalize.from_url(self.editing_canvas.image).permute(1, 2, 0)
        rendered_img = paste_clip_at_center(source_rgb, target_rgb, centered_location(copy_to), area)[0].permute(2, 0, 1)

        self.editing_canvas.mask = ''
        self.editing_canvas.image = renormalize.as_url(rendered_img)
        self.positive_masks[self.editing_canvas.index] += copy_to
        self.real_images_array[self.editing_canvas.index].src = renormalize.as_url(F.interpolate(rendered_img.unsqueeze(dim=0), size=(self.size, self.size)).squeeze())
        self.from_editing_canvas()

    def execute_edit(self):
        if self.edit_type == 'color':
            self.toggle_grad()
            self.toggle_color_edit()
            self.create_color_dataset()
            self.optimize()
        if self.edit_type == 'removal':
            self.use_color_cache = False
            self.toggle_grad()
            self.toggle_shape_edit()
            self.create_remove_dataset()
            self.get_cache()
            self.optimize()
        if self.edit_type == 'addition':
            self.use_color_cache = False
            self.toggle_grad()
            self.toggle_shape_edit()
            self.create_addition_dataset()
            self.optimize()
        if self.edit_type == 'color_from':
            target_style = self.all_instance_styles[self.target].unsqueeze(dim=0).cuda()
            self.instance_style = torch.cat([self.instance_style[:, :32], target_style[:, 32:]], dim=1)
        if self.edit_type == 'shape_from':
            self.use_color_cache = False
            target_style = self.all_instance_styles[self.target].unsqueeze(dim=0).cuda()
            self.instance_style = torch.cat([target_style[:, :32], self.instance_style[:, 32:]], dim=1)
        rgbs, disps = self.render(self.poses, self.instance_style, get_disps=True, update=True)
        self.use_color_cache = True
        self.update_canvas(rgbs, disps)

    def get_image_dataset(self):
        images = []
        poses = []
        positive_masks = []
        negative_masks = []

        for i in range(self.num_canvases):
            # TODO: speed the .sum() up by having an edited field
            if self.real_canvas_array[i].negative_mask != '' or self.positive_masks[i].sum() != 0:
                image = renormalize.from_url(self.real_canvas_array[i].image) / 2 + 0.5
                if self.real_canvas_array[i].negative_mask != '':
                    negative_mask = renormalize.from_url(self.real_canvas_array[i].negative_mask) / 2 + 0.5
                    negative_mask = F.interpolate(negative_mask.unsqueeze(dim=0), size=(self.size * 2, self.size * 2)).squeeze()
                    negative_masks.append((negative_mask > 0).float().clamp_(0, 1))
                else:
                    negative_masks.append(torch.zeros(self.positive_masks[i].shape).cpu())
                if self.positive_masks[i].sum() != 0:
                    positive_masks.append(self.positive_masks[i].clamp_(0, 1))
                else:
                    positive_masks.append(torch.zeros(negative_masks[-1].shape).cpu())
                images.append(image)
                poses.append(self.poses[i])
        images = torch.stack(images).permute(0, 2, 3, 1)
        positive_masks = torch.stack(positive_masks).permute(0, 2, 3, 1)
        negative_masks = torch.stack(negative_masks).permute(0, 2, 3, 1)
        poses = torch.stack(poses)
        return images, positive_masks, negative_masks, poses

    def create_color_dataset(self):
        if self.color_params == 'color_branch':
            self.optimizer = torch.optim.Adam(params=list(self.train_kwargs['network_fine'].color_branch()), lr=LR, betas=(0.9, 0.999))
        elif self.color_params == 'color_code':
            self.optimizer = None
        elif self.color_params == 'whole_network':
            self.optimizer = torch.optim.Adam(params=list(self.train_kwargs['network_fine'].parameters()), lr=LR, betas=(0.9, 0.999))
        images, positive_masks, negative_masks, poses = self.get_image_dataset()
        self.dataset = NerfDataset(images, poses, positive_masks, negative_masks, self.instance_style, self.hwfs, self.device, self.edit_type, N_rays=N_RAYS[self.edit_type], optimize_code=True, lr=LR)

    def create_remove_dataset(self):
        if self.shape_params == 'fusion_shape_branch':
            self.optimizer = torch.optim.Adam(params=list(self.train_kwargs['network_fine'].fusion_shape_branch()), lr=LR, betas=(0.9, 0.999))
            optimize_code = False
        elif self.shape_params == 'shape_branch':
            self.optimizer = torch.optim.Adam(params=list(self.train_kwargs['network_fine'].shape_branch()), lr=LR, betas=(0.9, 0.999))
            optimize_code = False
        elif self.shape_params == 'shape_code':
            self.optimizer = None
            optimize_code = True
        elif self.shape_params == 'whole_network':
            self.optimizer = torch.optim.Adam(params=list(self.train_kwargs['network_fine'].parameters()), lr=LR, betas=(0.9, 0.999))
            optimize_code = True
        images, positive_masks, negative_masks, poses = self.get_image_dataset()
        self.dataset = NerfDataset(images, poses, positive_masks, negative_masks, self.instance_style, self.hwfs, self.device, self.edit_type, randneg=self.randneg, lr=LR, N_rays=N_RAYS[self.edit_type], optimize_code=optimize_code, use_cached=True)

    def create_addition_dataset(self):
        params = list(self.train_kwargs['network_fine'].fusion_shape_branch()) + list(self.train_kwargs['network_fn'].fusion_shape_branch())
        self.optimizer = torch.optim.Adam(params=params, lr=LR, betas=(0.9, 0.999))
        images, positive_masks, negative_masks, poses = self.get_image_dataset()
        self.dataset = NerfDataset(images, poses, positive_masks, negative_masks, self.instance_style, self.hwfs, self.device, self.edit_type, randneg=self.randneg, lr=LR)

    def get_cache(self):
        if self.shape_params == 'fusion_shape_branch' and self.color_params == 'color_branch':
            with torch.no_grad():
                self.train_kwargs['network_fine'].get_cached = 'shape' if self.edit_type in ('addition', 'removal') else 'color'
                self.train_kwargs.update({'near': self.near, 'far': self.far})
                H, W, f = self.hwfs[0]
                features, weights = [], []
                for i in range(len(self.dataset)):
                    batch_rays, _, style, _, _, _, _ = self.dataset.get_data_batch(all_rays=True, imgnum=i)
                    rgb, disp, acc, extras = render(H, W, f, style=style, rays=batch_rays, **self.train_kwargs)
                    features.append(extras['features'])
                    weights.append(extras['weights0'])
                if self.edit_type in ('addition', 'removal'):
                    self.dataset.shape_features = features
                else:
                    self.dataset.color_features = features
                self.dataset.weights = weights
                self.train_kwargs['network_fine'].get_cached = None

    def optimize(self):
        niter = N_ITERS[self.edit_type]
        H, W, f = self.hwfs[0]
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = LR
        for i in range(niter):
            batch_rays, target_s, style, mask_rays, shape_features, color_features, weights = self.dataset.get_data_batch()
            if shape_features is not None:
                features = shape_features
            elif color_features is not None:
                features = color_features
            else:
                features = None
            self.train_kwargs.update({'near': self.near, 'far': self.far})
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            rgb, disp, acc, extras = render(H, W, f, style=style, rays=batch_rays, feature=features, weights=weights, **self.train_kwargs)

            loss = img2mse(rgb, target_s)
            if self.edit_type == 'addition':
                loss += img2mse(extras['rgb0'], target_s)

            weight_change_loss = torch.tensor(0.)
            for k, v in self.train_kwargs['network_fine'].named_parameters():
                if 'weight' in k:
                    weight_change_loss += (self.old_fine_network[k] - v).pow(2).mean()
            weight_change_loss = 10 * weight_change_loss
            loss += weight_change_loss

            if self.edit_type == 'removal':
                sigma_loss = 0.01 * (mask_rays * (-extras['weights'] * (extras['weights'] + 1e-7).log()).sum(dim=1)).mean()
                loss += sigma_loss
            else:
                sigma_loss = torch.tensor(0.)

            loss.backward()
            if self.optimizer is not None:
                self.optimizer.step()

            if VERBOSE:
                self.msg_out.print(f'Iter {i+1}/{niter}, Loss: {loss.item():.4f}', replace=True)
            else:
                self.msg_out.print(f'Iter {i+1}/{niter}', replace=True)

    def toggle_grad(self):
        for n, p in self.train_kwargs['network_fn'].named_parameters():
            p.requires_grad_(True)
        for n, p in self.train_kwargs['network_fine'].named_parameters():
            p.requires_grad_(True)

    def toggle_color_edit(self):
        self.train_kwargs['perturb'] = 0
        self.train_kwargs['perturb_coarse'] = 0

        for n, p in self.train_kwargs['network_fn'].named_parameters():
            p.requires_grad_(False)
        for n, p in self.train_kwargs['network_fine'].named_parameters():
            if not any([s in n for s in ['style_linears.2', 'rgb_linear.', 'views_linears.']]):
                p.requires_grad_(False)

    def toggle_shape_edit(self):
        if self.edit_type == 'addition':
            self.train_kwargs['perturb'] = 1
            self.train_kwargs['perturb_coarse'] = 1
        else:
            self.train_kwargs['perturb'] = 0
            self.train_kwargs['perturb_coarse'] = 0

    def save(self):
        if self.editname_textbox.value == '':
            self.show_msg('Please enter a name to save your file')
            return

        savedir = os.path.join(self.savedir, self.editname_textbox.value)

        # clear the savedir if conflicting
        if os.path.exists(savedir):
            for x in os.listdir(savedir):
                os.remove(os.path.join(savedir, x))

        os.makedirs(savedir, exist_ok=True)
        for i in range(self.num_canvases):
            if self.real_canvas_array[i].negative_mask != '':
                torch.save(self.real_canvas_array[i].negative_mask, os.path.join(savedir, f'{i}_neg.pt'))
            if self.positive_masks[i].sum() != 0:
                image = renormalize.from_url(self.real_canvas_array[i].image) / 2 + 0.5
                utils.save_image(image, os.path.join(savedir, f'{i}_rgb.png'))
                torch.save(self.positive_masks[i].clamp_(0, 1), os.path.join(savedir, f'{i}_pos.pt'))
        with open(os.path.join(savedir, 'edit_type.txt'), 'w') as f:
            f.write(f'{self.edit_type}')
        self.show_msg('Done saving')

    def load(self):
        if self.editname_textbox.value == '':
            self.show_msg('Please enter a file name to load')
            return
        savedir = os.path.join(self.savedir, self.editname_textbox.value)
        if not os.path.exists(savedir):
            self.show_msg(f'{savedir} does not exist')
            return
        with open(os.path.join(savedir, 'edit_type.txt')) as f:
            self.edit_type = f.readlines()[0].strip()
        trn = transforms.ToTensor()
        for i in range(self.num_canvases):
            if os.path.exists(os.path.join(savedir, f'{i}_rgb.png')):
                image = trn(Image.open(os.path.join(savedir, f'{i}_rgb.png'))) * 2 - 1
                self.real_canvas_array[i].image = renormalize.as_url(image)
                self.real_canvas_array[i].resized_image = renormalize.as_url(F.interpolate(image.unsqueeze(dim=0), size=(self.size, self.size)).squeeze())
                self.real_images_array[i].src = self.real_canvas_array[i].resized_image
            if os.path.exists(os.path.join(savedir, f'{i}_pos.pt')):
                self.positive_masks[i] = torch.load(os.path.join(savedir, f'{i}_pos.pt'))
            if os.path.exists(os.path.join(savedir, f'{i}_neg.pt')):
                self.real_canvas_array[i].negative_mask = torch.load(os.path.join(savedir, f'{i}_neg.pt'))

    def saved_names(self):
        return [x for x in os.listdir(self.savedir) if os.path.exists(os.path.join(self.savedir, x, 'edit_type.txt'))]

    def show_msg(self, msg):
        self.msg_out.clear()
        self.msg_out.print(msg, replace=False)

    def widget_html(self):
        def h(w):
            return w._repr_html_()
        html = f'''<div {self.std_attrs()}>
        <div style="display:inline-block; width:{1.00 * self.size + 2}px;
          text-align:center">
        {h(self.toggle_rgbs_disps_btn)}
        </div>
        <div style="display:inline-block; width:{5.00 * self.size + 2}px;
          text-align:right">
        {h(self.editname_textbox)}
        {h(self.save_btn)}
        {h(self.load_btn)}
        </div>

        <div style="margin-top: 8px; margin-bottom: 8px;">
        <div style="display:inline-block; width:{1.00 * self.size + 2}px;
          text-align:center">
        {h(self.positive_mask_btn)}
        </div>
        <div style="display:inline-block; width:{1.00 * self.size + 2}px;
          text-align:center">
        {h(self.sigma_mask_btn)}
        </div>
        <div style="display:inline-block; width:{1.00 * self.size + 2}px;
          text-align:center">
        {h(self.addition_mask_btn)}
        </div>
        <div style="display:inline-block; width:{1.00 * self.size + 2}px;
          text-align:center">
        {h(self.color_from_btn)}
        </div>
        <div style="display:inline-block; width:{1.00 * self.size + 2}px;
          text-align:center">
        {h(self.shape_from_btn)}
        </div>
        <div style="display:inline-block; width:{1.00 * self.size + 2}px;
          text-align:center">
        {h(self.execute_btn)}
        </div>
        </div>

        <div>

        <div style="display:inline-block;
          width:{(self.size + 2) * 4}px;
          height:{40}px;
          vertical-align:top;
          overflow-y: scroll;
          text-align:center">
          {show.html([[x] for x in self.color_pallete])}
        </div>
        <div style="display:inline-block;
          width:{(self.size + 2) * 2 + 12}px;
          height:{40}px;
          vertical-align:top;
          overflow-y: scroll;
          text-align:center">
        {h(self.msg_out)}
        </div>
        <div>

        <div style="width:{(self.size + 2) * 6 + 20}px;">
        <hr style="border:2px dashed gray; background-color: white">
        </div>

        <div>
        {h(self.editing_canvas)}
        <div style="display:inline-block;
          width:{(self.size + 2) * 3 + 20}px;
          height:{(self.size + 2) * 3 + 20}px;
          vertical-align:top;
          overflow-y: scroll;
          text-align:center">
          {show.html([[c] for c in self.real_images_array])}
        </div>
        </div>

        <div style="width:{(self.size + 2) * 6 + 20}px;">
        <hr style="border:2px dashed gray; background-color: white">
        </div>

        <div>
        <div style="display:inline-block;
        width:{(self.size + 2) * 6 + 20}px;
        height:{140}px;
        vertical-align:top;
        overflow-y: scroll;
        text-align:center">
        {show.html([[c] for c in self.transfer_instances_array])}
        </div>
        </div>

        <div style="width:{(self.size + 2) * 6 + 20}px;">
        <hr style="border:2px dashed gray; background-color: white">
        </div>

        <div>
        {h(self.copy_canvas)}
        <div style="display:inline-block;
        width:{(self.size + 2) * 4 + 20}px;
        height:{(self.size * 2)}px;
        vertical-align:top;
        overflow-y: scroll;
        text-align:center">
        {show.html([[c] for c in self.addition_instances_array])}
        </div>
        </div>

        </div>
        '''
        return html

##########################################################################
# Utility functions
##########################################################################


def positive_bounding_box(data):
    pos = (data > 0)
    v, h = pos.sum(0).nonzero(), pos.sum(1).nonzero()
    left, right = v.min().item(), v.max().item()
    top, bottom = h.min().item(), h.max().item()
    return top, left, bottom + 1, right + 1


def centered_location(data):
    t, l, b, r = positive_bounding_box(data)
    return (t + b) // 2, (l + r) // 2


def paste_clip_at_center(source, clip, center, area=None):
    source = source.unsqueeze(dim=0).permute(0, 3, 1, 2)
    clip = clip.unsqueeze(dim=0).permute(0, 3, 1, 2)
    target = source.clone()
    t, l = (max(0, min(e - s, c - s // 2))
            for s, c, e in zip(clip.shape[2:], center, source.shape[2:]))
    b, r = t + clip.shape[2], l + clip.shape[3]
    # TODO: consider copying over a subset of channels.
    target[:, :, t:b, l:r] = clip if area is None else (
        (1 - area)[None, None, :, :].to(target.device) *
        target[:, :, t:b, l:r] +
        area[None, None, :, :].to(target.device) * clip)
    target = target.squeeze().permute(1, 2, 0)
    return target, (t, l, b, r)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=int)
    parser.add_argument('--randneg', type=int, default=8192)
    parser.add_argument('--config')
    parser.add_argument('--expname')
    parser.add_argument('--editname')
    parser.add_argument('--second_editname')
    parser.add_argument('--shape_params', default='fusion_shape_branch')
    parser.add_argument('--color_params', default='color_branch')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()

    writer = NeRFEditingApp(instance=args.instance, expname=args.expname, config=args.config, shape_params=args.shape_params, color_params=args.color_params, randneg=args.randneg, num_canvases=9, use_cached=False)
    expname = writer.expname

    editnames = [args.editname]
    if args.second_editname:
        editnames.append(args.second_editname)

    for editname in editnames:
        if editname:
            savedir = os.path.join(expname, editname)

            if args.shape_params != 'fusion_shape_branch':
                savedir = os.path.join(savedir, args.shape_params)
            if args.color_params != 'color_branch':
                savedir = os.path.join(savedir, args.color_params)
            if args.randneg != 8192:
                savedir += f'_{args.randneg}'

            os.makedirs(savedir, exist_ok=True)
            print('Working in', savedir)

            # load and execute the edit
            writer.editname_textbox.value = editname
            writer.load()
            writer.execute_edit()
        else:
            savedir = os.path.join(expname, 'flythroughs', str(args.instance))
            os.makedirs(savedir, exist_ok=True)

        all_poses = torch.tensor(np.load(os.path.join(expname, 'poses.npy')))
        all_hwfs = torch.tensor(np.load(os.path.join(expname, 'hwfs.npy')))
        if args.expname:
            N_per_instance = 1
        else:
            N_per_instance = all_poses.shape[0] // writer.all_instance_styles.shape[0]
        ps, pe = args.instance * N_per_instance, (args.instance + 1) * N_per_instance
        all_poses = all_poses[ps:pe]

        if args.video:
            all_poses, all_hwfs = generate_flythrough(all_poses[0].cpu(), all_hwfs[0], num_poses=100)

        nfs = [[writer.near, writer.far]] * all_poses.shape[0]
        styles = writer.instance_style.repeat((all_poses.shape[0], 1))

        with torch.no_grad():
            print(f'Saving samples in {savedir}')
            rgbs, disps, psnr = render_path(all_poses, styles, all_hwfs, writer.chunk, writer.test_kwargs, nfs=nfs, savedir=savedir, verbose=True)
            if args.video:
                imageio.mimwrite(os.path.join(savedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(os.path.join(savedir, 'disps.mp4'), to8b(disps / np.max(disps)), fps=30, quality=8)
