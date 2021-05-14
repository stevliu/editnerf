conda create --name editnerf python=3.6
conda activate editnerf
conda install --file requirements.txt
pip install lpips imageio-ffmpeg
cd torchsearchsorted
pip install .
cd ../