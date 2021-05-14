import os
import json
from tqdm import tqdm
import numpy as np
from numpy import sin, cos
from PIL import Image
from torchvision import transforms, utils

basedir = 'data/dosovitskiy_chairs/rendered_chairs'
todir = 'data/dosovitskiy_chairs/'
FOCAL = 1.93699312 * 4 / 6  # Additional * 4 / 6 factor from resizing 600 -> 400

# Resize and crop so that the chair takes up most of the image
trn = transforms.Compose([transforms.Resize(400), transforms.CenterCrop(256), transforms.ToTensor()])


def to_rad(deg):
    return deg / 180 * np.pi


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


for instance in tqdm(os.listdir(basedir)):
    instance_dir = os.path.join(basedir, instance)
    save_dir = os.path.join(todir, instance)
    os.makedirs(save_dir, exist_ok=True)
    if not os.path.isdir(instance_dir):
        continue
    transforms_train = {'frames': [], 'focal': FOCAL}
    transforms_test = {'frames': [], 'focal': FOCAL}
    transforms_val = {'frames': [], 'focal': FOCAL}
    poses = []
    imgs = [x for x in os.listdir(os.path.join(instance_dir, 'renders')) if '.png' in x]
    for i, img_name in enumerate(sorted(imgs, key=lambda x: int(x[6:9]))):
        data = img_name.split('_')[2:5]

        phi, theta, rho = [int(x[1:4]) for x in data]
        phi = 90 - phi  # differs from convention used in http://www.cs.binghamton.edu/~reckert/460/3dview.html
        phi, theta = [to_rad(x) for x in [phi, theta]]

        w2c = np.zeros((4, 4))
        w2c[0][0] = -sin(theta)
        w2c[0][1] = cos(theta)
        w2c[0][2] = 0
        w2c[1][0] = -cos(phi) * cos(theta)
        w2c[1][1] = -cos(phi) * sin(theta)
        w2c[1][2] = sin(phi)
        w2c[2][0] = -sin(phi) * cos(theta)
        w2c[2][1] = -sin(phi) * sin(theta)
        w2c[2][2] = -cos(phi)
        w2c[2][3] = rho
        w2c[3][3] = 1
        w2c[2] *= -1  # http://www.cs.binghamton.edu/~reckert/460/3dview.html has z axis convention flipped
        transform_matrix = np.linalg.inv(w2c)
        poses.append(transform_matrix)

        if i % 10 in (0, 1):
            transforms_dict = transforms_test
        elif i % 10 in (2, 3):
            transforms_dict = transforms_val
        else:
            transforms_dict = transforms_train

        resized_image = trn(Image.open(os.path.join(instance_dir, 'renders', img_name)).convert('RGB'))
        utils.save_image(resized_image, os.path.join(save_dir, img_name))

        transforms_dict['frames'].append({'file_path': img_name.split('.png')[0], 'transform_matrix': listify_matrix(transform_matrix)})
    with open(os.path.join(save_dir, 'transforms_train.json'), 'w') as out_file:
        json.dump(transforms_train, out_file, indent=4)
    with open(os.path.join(save_dir, 'transforms_val.json'), 'w') as out_file:
        json.dump(transforms_val, out_file, indent=4)
    with open(os.path.join(save_dir, 'transforms_test.json'), 'w') as out_file:
        json.dump(transforms_test, out_file, indent=4)
