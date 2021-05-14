import os
import json
import numpy as np
from tqdm import tqdm
from torchvision import transforms, utils
from PIL import Image

N = 10000
basedir = 'data/carla'
trn = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
todir = 'data/carla'


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


focal = float(955.4050067376327)

for i in tqdm(range(N)):
    imgnum = '{:06d}'.format(i)

    target_dir = os.path.join(todir, imgnum)
    os.makedirs(target_dir, exist_ok=True)

    img_file = os.path.join(basedir, 'carla_images', f'{imgnum}.png')
    pose_file = os.path.join(basedir, 'carla_poses', f'{imgnum}_extrinsics.npy')
    pose = np.load(pose_file).astype(np.float64)  # saved as c2w

    frames_train = [{'file_path': '0', 'transform_matrix': listify_matrix(pose)}]
    frames_test = [{'file_path': '0', 'transform_matrix': listify_matrix(pose)}]
    utils.save_image(trn(Image.open(img_file)), os.path.join(target_dir, '0.png'))

    # Note: dividing by 2 because we're resizing the original image from 512 --> 256
    with open(target_dir + '/' + 'transforms_train.json', 'w') as out_file:
        json.dump({'focal': focal / 2, 'frames': frames_train}, out_file, indent=4)
    with open(target_dir + '/' + 'transforms_val.json', 'w') as out_file:
        json.dump({'focal': focal / 2, 'frames': frames_test}, out_file, indent=4)
    with open(target_dir + '/' + 'transforms_test.json', 'w') as out_file:
        json.dump({'focal': focal / 2, 'frames': frames_test}, out_file, indent=4)
