## Reconstruction results

# First, generate the images and find their corresponding ground truth. 
python test_nerf.py --config configs/dosovitskiy_chairs/config.txt --render_test
python test_nerf.py --config configs/photoshapes/config.txt --render_test

# Then, compute metrics on these images. 
python utils/evaluate_reconstruction.py --expdir logs/dosovitskiy_chairs 
python utils/evaluate_reconstruction.py --expdir logs/photoshapes

## Real Image Fitting
python run_nerf.py --config configs/dosovitskiy_chairs/config.txt --real_image_dir data/real_chairs/shape00001_charlton --N_rand 512 --n_iters_real 10000 --n_iters_code_only 1000 --style_optimizer lbfgs --i_testset 1000 --i_weights 1000 --savedir real_chairs/shape00001_charlton --testskip 1