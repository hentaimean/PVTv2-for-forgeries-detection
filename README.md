# Applying PVTv2-Seg to Detecting Forgeries Digital Images

## Libs:
- [MMSegmentation v0.30.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0)
- [mmcv-full v1.7.0](https://github.com/open-mmlab/mmcv/tree/v1.7.0)
- PyTorch v1.10.0 with CUDA 11.3

## Usage

1. Install mmcv-full and MMSegmentation ([link](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)).
2. Download the CASIAv2 dataset ([link](https://github.com/namtpham/casia2groundtruth))
3. Use train_val_split_casia.py to divide the dataset into train, val and test
4. Use casia_copy_normed_masks.py to normalize masks
5. Put casia.py in /mmsegmentation/mmseg/datasets for registering the dataset
6. Start train_notebook.ipynb
7. Start test_notebook.ipynb
8. Use draw_gt_preds.py to draw the results

## Results

|    Class         | Dice  | Acc   | Fscore | Precision | Recall | IoU   |
| :--------------: | :---: | :---: | :----: | :-------: | :----: | :---: |
|    background    | 96.94 | 98.25 | 96.94  |   95.67   | 98.25  | 94.07 |
|    defect        | 53.11 | 44.09 | 53.11  |   66.77   | 44.09  | 36.16 |

![Tp_D_CRN_M_N_art10115_cha00086_11526](https://github.com/hentaimean/PVTv2-for-forgeries-detection/assets/106330825/941d65be-08e3-4da2-b3a5-1ab7cd2c6908)
![Tp_D_CRD_S_B_ani00071_ani00064_00191](https://github.com/hentaimean/PVTv2-for-forgeries-detection/assets/106330825/90274849-97cd-40a2-83f9-8bb4ed5f1906)



## Original article and repository
- [PVTv2 on GitHub](https://github.com/whai362/PVTv2-Seg)
- [PVTv2 on arXiv.org](https://arxiv.org/pdf/2106.13797.pdf)

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Contact
E-mail: hentaimean@mail.ru

