def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("-f", type=str, help='ipynb hack')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--savedir", type=str, default=None, help='where to save, overrides expname')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/', help='input data directory')
    parser.add_argument("--real_image_dir", type=str, default=None, help='directory containing real images')

    # training options
    parser.add_argument("--n_iters", type=int, default=10000000, help='number of iterations to train')
    parser.add_argument("--n_iters_real", type=int, default=25000, help='number of iterations to train')
    parser.add_argument("--n_iters_code_only", type=int, default=25000, help='number of iterations to train')
    parser.add_argument("--N_viewdirs_reg", type=int, default=0, help='number of viewdirs to regularize radiance at each point')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4, help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024 * 4, help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64, help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--var_param", type=float, default=0, help='if > 0, penalizes variance of color at a point')
    parser.add_argument("--weight_change_param", type=float, default=0, help='if > 0, penalizes deviation from original model. useful for real image fitting')
    parser.add_argument("--precrop_iters", type=int, default=0, help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float, default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--style_optimizer", type=str, default='adam', help='options : adam/lbfgs')

    # learned code options
    parser.add_argument("--use_styles", action='store_true', help='use learned styles')
    parser.add_argument("--style_dim", type=int, default=64, help='style dimension for nerf')
    parser.add_argument("--style_depth", type=int, default=1, help='num layers for style embedding for nerf')
    parser.add_argument("--embed_dim", type=int, default=-1, help='embedding dimension for nerf style vector, -1 means just do identity')
    parser.add_argument("--separate_codes", action='store_true', help='dedicate half of one vector to be shape and one for rgb')

    # network architecture options
    parser.add_argument("--shared_shape", action='store_true', help='use instance independent shared shape branch')

    parser.add_argument("--D_mean", type=int, default=4, help='layers in mean shape network')
    parser.add_argument("--W_mean", type=int, default=256, help='channels per layer inn mean shape network')
    parser.add_argument("--D_instance", type=int, default=4, help='layers in instance network')
    parser.add_argument("--W_instance", type=int, default=256, help='channels per layer in instance network')
    parser.add_argument("--D_fusion", type=int, default=4, help='layers in fusion network')
    parser.add_argument("--W_fusion", type=int, default=256, help='channels per layer in fusion network')
    parser.add_argument("--D_sigma", type=int, default=1, help='layers in density network')
    parser.add_argument("--W_sigma", type=int, default=256,  help='channels per layer in density network')
    parser.add_argument("--D_rgb", type=int, default=2, help='layers in rgb network')
    parser.add_argument("--W_rgb", type=int, default=128, help='channels per layer in rgb network')
    parser.add_argument("--W_bottleneck", type=int, default=8, help='channels after 1st layer of rgb network')

    parser.add_argument("--i_embed", type=int, default=0, help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, help='log2 of max freq for positional encoding (2D direction)')

    # rendering options:
    parser.add_argument("--blender_near", type=float, default=2., help='near parameter')
    parser.add_argument("--blender_far", type=float, default=6., help='far parameter')
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0, help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--perturb_coarse", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    parser.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # data options
    parser.add_argument("--N_instances", type=int, default=10, help='number of instances to train on')
    parser.add_argument("--instance", type=int, default=-1, help='instance number to train on')
    parser.add_argument("--skip_loading", action='store_true', help='skip loading the model from last checkpoint')
    parser.add_argument("--testskip", type=int, default=8, help='will load 1/N images from test/val sets')
    parser.add_argument("--trainskip", type=int, default=1, help='will load 1/N images from train sets')

    # visualization options
    parser.add_argument("--shuffle_poses", action='store_true', help='shuffle test set poses')
    parser.add_argument("--render_test", action='store_true', help='render the test set instead of render_poses path')
    parser.add_argument("--render_train", action='store_true', help='render the train set instead of render_poses path')

    # model loading options
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--load_from", type=str, default=None, help='path to load model from in test time')
    parser.add_argument("--load_it", type=int, default=0, help='iteration to load')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10000000,
                        help='frequency of testset saving')
    parser.add_argument("--i_trainset", type=int, default=10000000,
                        help='frequency of trainset saving')
    parser.add_argument("--i_video", type=int, default=10000000,
                        help='frequency of render_poses video saving')

    return parser
