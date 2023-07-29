import matplotlib.pyplot as plt
from skimage import io
import os

gt_dir = '../train_dataset/test/gt'
orig_img_dir = '../train_dataset/test/images'
pred_dir = '../work_dirs/casia512_320k/fpn_pvtv2_b5/preds'
save_dir = '../work_dirs/casia512_320k/fpn_pvtv2_b5/image_gt_pred'

for img_fname in os.listdir(pred_dir):
    pred = io.imread(os.path.join(pred_dir, img_fname))
    base_fname = img_fname[:-4]
    print(base_fname)
    gt = io.imread(os.path.join(gt_dir, base_fname + '.png'))
    img = io.imread(os.path.join(orig_img_dir, base_fname + '.jpg'))

    fig, ax = plt.subplots(ncols=3, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title('Orig image')
    ax[1].imshow(gt, cmap='gray')
    ax[1].set_title('Gt')
    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title('Pred')
    plt.savefig(os.path.join(save_dir, f'{base_fname}.png'))