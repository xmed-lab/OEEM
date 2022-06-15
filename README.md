## OEEM
Note: this code is expected to be ready at middle June.

Yi Li*, Yiduo Yu*, Yiwen Zou*, Tianqi Xiang, Xiaomeng Li, "Online Easy Example Mining for Weakly-supervised Gland Segmentation from Histology Images", MICCAI 2022 (Accepted). [[paper](https://arxiv.org/)]

### 1. Introduction
This framework is designed for histology images, containing two stages. The first classification stage generates pseudo-masks for pathes. And the segmentation stage uses OEEM to mitigate the noise in pseudo-masks dynamically.

![framework visualization](segmentation/demo/oeem_vis.png)

### 1. Environment

This code has been tested with Python 3.7, PyTorch 1.10.2, CUDA 11.3 mmseg 0.8.0 and mmcv 1.4.0 on Ubuntu 20.04.

### 2. Preparation

Download resources (dataset, weights), then link to codes.
```shell
git clone https://github.com/XMed-Lab/OEEM.git
cd OEEM
ln -s OEEM_resources/glas_cls classification/glas
ln -s OEEM_resources/glas_seg segmentation/glas
ln -s OEEM_resources/weights classification/weights
ln -s OEEM_resources/weights segmentation/weights
```

Install library dependencies
```shell
pip install -r requirements.txt
```

Install mmsegentation.
```shell
cd segmentation
pip install -U openmim
mim install mmcv-full==1.4.0
pip install -v -e .
```

### 3. Training

Train classification model.

```shell
python classification/train.py -d 0 -m res38d
```

Generate pseudo-mask (WSI size). The output will be in `[model_name]_best_train_pseudo_mask` folder.

```shell
python classification/prepare_seg_inputs.py -d 0 -ckpt res38d_best
```

Split WSI pseudo-mask to patches for segmentation.

```shell
python segmentation/tools/crop_img_and_gt.py segmentation/glas/images classification/res38d_best_train_pseudo_mask segmentation/glas
```

Train segmentation model.

```shell
cd segmentation
bash tools/dist_train.sh configs/pspnet_oeem/pspnet_wres38-d8_10k_histo.py 1 runs/oeem
```

### 4. Testing

Test segmentation model.

```shell
cd segmentation
bash tools/dist_test.sh configs/pspnet_oeem/pspnet_wres38-d8_10k_histo_test.py runs/oeem/[name of best ckpt] 1
```

Merge patches and evaluation.

```shell
python tools/merge_patches.py glas/test_patches glas/test_wsi 2
python tools/count_miou.py glas/test_wsi glas/gt_val 2
```

Results compared with WSSS for natural images:
| Method  | Dice   |  mIoU  |
| ---------- | :-----------:  | :-----------: |
| SEAM | 66.11%   | 79.59%     |
| Adv-CAM | 68.54%   | 81.33%     |
| SC-CAM | 71.52%   | 83.40%     |
| Ours | 77.56%   | 87.36%     |

### 5. Citation

```
@misc{https://doi.org/10.48550/arxiv.2206.06665,
  doi = {10.48550/ARXIV.2206.06665},
  url = {https://arxiv.org/abs/2206.06665},
  author = {Li, Yi and Yu, Yiduo and Zou, Yiwen and Xiang, Tianqi and Li, Xiaomeng},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Online Easy Example Mining for Weakly-supervised Gland Segmentation from Histology Images},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

### License

This repository is released under MIT License (see LICENSE file for details).
