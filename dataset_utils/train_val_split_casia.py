import glob
import os
import numpy as np
import shutil


inputDir = '../CASIAv2/images'
input_gt_dir = '../CASIAv2/ground_truth'
val_gt_dir = '../train_dataset/val/gt'
test_gt_dir = '../train_dataset/test/gt'
train_gt_dir = '../train_dataset/train/gt'
val_img_dir = '../train_dataset/val/images'
test_img_dir = '../train_dataset/test/images'
train_img_dir = '../train_dataset/train/images'

images_list = glob.glob(os.path.join(inputDir, '*.jpg'))
length = len(images_list)

indices = np.arange(0, length, 1, dtype=int)
np.random.shuffle(indices)

test_max_index = round(length * 0.1)

print(indices)

for index in range(0, test_max_index):
    img = images_list[indices[index]]
    img_name = os.path.basename(img)[:-4]
    print(img_name)
    shutil.copyfile(os.path.join(inputDir, img_name + '.jpg'), os.path.join(test_img_dir, img_name + '.jpg'))
    shutil.copyfile(os.path.join(input_gt_dir, img_name + '.png'), os.path.join(test_gt_dir, img_name + '.png'))

for index in range(test_max_index, 2 * test_max_index):
    img = images_list[indices[index]]
    img_name = os.path.basename(img)[:-4]
    print(img_name)
    shutil.copyfile(os.path.join(inputDir, img_name + '.jpg'), os.path.join(val_img_dir, img_name + '.jpg'))
    shutil.copyfile(os.path.join(input_gt_dir, img_name + '.png'), os.path.join(val_gt_dir, img_name + '.png'))

for index in range(2* test_max_index, length):
    img = images_list[indices[index]]
    img_name = os.path.basename(img)[:-4]
    print(img_name)
    shutil.copyfile(os.path.join(inputDir, img_name + '.jpg'), os.path.join(train_img_dir, img_name + '.jpg'))
    shutil.copyfile(os.path.join(input_gt_dir, img_name + '.png'), os.path.join(train_gt_dir, img_name + '.png'))
