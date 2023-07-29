import numpy as np
import os
from skimage import io

orig_dir = '../train_dataset/train/gt'
destination_dir = '../train_dataset/train/gt_normed'

for img_fname in os.listdir(orig_dir):
    img = io.imread(os.path.join(orig_dir, img_fname)).astype(bool)
    if len(img.shape) == 3:
        img = img[..., 0]
    gt_img = np.zeros_like(img).astype('uint8')
    gt_img[img] = 1
    img_base_name = os.path.basename(img_fname)
    print(img_base_name)
    io.imsave(os.path.join(destination_dir, img_fname), gt_img)
