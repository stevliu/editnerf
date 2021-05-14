import torch.nn.functional as F
import torch.nn as nn
import torch
torch.autograd.set_detect_anomaly(True)


class StyleMLP(nn.Module):
    def __init__(self, style_dim=8, embed_dim=128, style_depth=1):
        super().__init__()
        self.activation = F.relu

        lin_block = nn.Linear
        first_block = nn.Linear(style_dim, embed_dim)
        self.mlp = nn.ModuleList([first_block] + [lin_block(embed_dim, embed_dim) for _ in range(style_depth - 1)])

    def forward(self, x):
        for i, layer in enumerate(self.mlp):
            x = self.activation(layer(x))
        return x

# Model


class NeRF(nn.Module):
    def __init__(self, D_mean=4, W_mean=256, D_instance=4, W_instance=256, D_fusion=4, W_fusion=256, D_sigma=1, W_sigma=256, D_rgb=2, W_rgb=128, W_bottleneck=8, input_ch=3, input_ch_views=3, output_ch=4, style_dim=64, embed_dim=128, style_depth=1, shared_shape=True, use_styles=True, separate_codes=True, use_viewdirs=True, **kwargs):
        super(NeRF, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.use_viewdirs = use_viewdirs
        self.use_styles = use_styles
        self.separate_codes = separate_codes
        self.shared_shape = shared_shape
        self.get_cached = None  # Updated by render_path to get cache
        self.activation = F.relu

        if shared_shape:
            self.mean_network = nn.Sequential(*[nn.Linear(input_ch, W_mean)], *[nn.Sequential(nn.ReLU(), nn.Linear(W_mean, W_mean)) for i in range(D_mean - 2)])
            self.mean_output = nn.Sequential(nn.ReLU(), nn.Linear(W_mean, W_instance))

        if separate_codes:
            style_dim = style_dim // 2

        self.style_dim = style_dim
        self.embed_size = style_dim if embed_dim < 0 else embed_dim

        pts_inp_dim = (input_ch + self.embed_size) if use_styles else input_ch
        view_inp_dim = (input_ch_views + self.embed_size) if use_styles else input_ch_views

        self.instance_network = nn.Sequential(*[nn.Sequential(nn.Linear(pts_inp_dim, W_instance), nn.ReLU())], *[nn.Sequential(nn.Linear(W_instance, W_instance), nn.ReLU()) for i in range(D_instance - 1)])
        self.instance_to_fusion = nn.Linear(pts_inp_dim + W_instance, W_fusion)
        self.fusion_network = nn.Sequential(*[nn.Sequential(nn.Linear(W_fusion, W_fusion), nn.ReLU()) for i in range(D_fusion - 1)])

        if use_viewdirs:
            if D_sigma > 1:
                self.sigma_linear = nn.Sequential(*[nn.Sequential(nn.Linear(W_fusion, W_sigma), nn.ReLU())], *[nn.Sequential(nn.Linear(W_sigma, W_sigma), nn.ReLU()) for _ in range(D_sigma - 2)], *[nn.Linear(W_sigma, 1)])
            else:
                self.sigma_linear = nn.Linear(W_fusion, 1)

            self.bottleneck_linear = nn.Linear(W_fusion, W_bottleneck)
            self.rgb_network = nn.Sequential(*[nn.Sequential(nn.Linear(view_inp_dim + W_bottleneck, W_rgb), nn.ReLU())], *[nn.Sequential(nn.Linear(W_rgb, W_rgb), nn.ReLU()) for i in range(D_rgb - 2)])
            self.rgb_linear = nn.Linear(W_rgb, 3)
        else:
            self.output_linear = nn.Linear(W_fusion, output_ch)

        if self.embed_size > 0:
            # One inputs to instance network, fusion network, and color branch
            self.style_linears = nn.ModuleList([StyleMLP(style_dim, self.embed_size, style_depth) for i in range(3)])
        else:
            self.style_linears = nn.ModuleList([nn.Identity() for i in range(3)])

        # self.num_parameters()

    def forward(self, x, styles, alpha=None, feature=None):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        if self.separate_codes:
            styles_sigma, styles_rgb = styles[:, :self.style_dim], styles[:, self.style_dim:]
        else:
            styles_sigma, styles_rgb = styles, styles

        if alpha is None:
            # Have to compute sigma

            if feature is None:
                # Compute mean shape features
                if self.shared_shape:
                    mean_output = self.mean_output(self.mean_network(input_pts))

                # Prepare input to instance network
                if self.use_styles:
                    h = torch.cat([input_pts, self.style_linears[0](styles_sigma)], dim=1)
                else:
                    h = input_pts

                h = self.instance_network(h)

                # Prepare input to fusion network
                if self.use_styles:
                    h = torch.cat([self.style_linears[1](styles_sigma), h], -1)
                h = torch.cat([input_pts, h], -1)
                instance_output = self.instance_to_fusion(h)

                # Add shared shape features to instance features
                if self.shared_shape:
                    h = instance_output + mean_output
                else:
                    h = instance_output

                shape_features = h
            else:
                # Cached instance_output + mean_output
                h = feature

            h = self.activation(h)
            fusion_output = self.fusion_network(h)
            alpha = self.sigma_linear(fusion_output)
            color_feature = self.bottleneck_linear(fusion_output)
        else:
            color_feature = feature

        if self.use_viewdirs:
            if self.use_styles:
                style_embedding = self.style_linears[2](styles_rgb)
                h = torch.cat([color_feature, input_views, style_embedding], -1)
            else:
                h = torch.cat([color_feature, input_views], -1)
            rgb = self.rgb_linear(self.rgb_network(h))
            if self.get_cached:
                if self.get_cached == 'color':
                    outputs = torch.cat([rgb, alpha, color_feature], -1)
                elif self.get_cached == 'shape':
                    outputs = torch.cat([rgb, alpha, shape_features], -1)
                else:
                    raise NotImplementedError()
            else:
                outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        return outputs

    def color_branch(self):
        return list(self.rgb_linear.parameters()) + list(self.rgb_network.parameters()) + list(self.style_linears[2].parameters()) + list(self.bottleneck_linear.parameters())

    def shape_branch(self):
        return list(self.sigma_linear.parameters())

    def fusion_shape_branch(self):
        return list(self.fusion_network.parameters()) + list(self.sigma_linear.parameters())

    def num_parameters(self):
        total = 0
        for n, p in self.named_parameters():
            total += p.numel()
        print('Total parameters:', total)
