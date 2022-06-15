import numpy as np
from PIL import Image
import cv2
import os, sys
import shutil
import math

imgd = sys.argv[1]
gtd = sys.argv[2]
outd = sys.argv[3]

img_type = '.bmp'
gt_type = '.png'
crop_size = 256
stride = 192

out_gtd = os.path.join(outd, 'pseudo_train_256_192')
os.makedirs(out_gtd, exist_ok=True)
out_imgd = os.path.join(outd, 'img_train_256_192')
os.makedirs(out_imgd, exist_ok=True)

gts = {}
for f in os.listdir(gtd):
    gts[f] = ''

for f in os.listdir(imgd):
    img = cv2.imread(os.path.join(imgd, f), flags=3)
    if f.replace(img_type, gt_type) in gts:
        gt = np.array(Image.open(os.path.join(gtd, f.replace(img_type, gt_type))))
    h, w = img.shape[:2]

    h_n = int(math.ceil(max(0, float(h - crop_size)) / stride)) + 1
    w_n = int(math.ceil(max(0, float(w - crop_size)) / stride)) + 1

    for h_i in range(h_n):
        for w_i in range(w_n):
            x2 = int(min(w, w_i * stride + crop_size))
            y2 = int(min(h, h_i * stride + crop_size))
            x1 = max(0, x2 - crop_size)
            y1 = max(0, y2 - crop_size)
            crop_img = img[y1:y2, x1:x2, :]
            out_name = f.replace(img_type, '_{:d}_{:d}_{:d}_{:d}_{:d}_{:d}.jpg'.format(h, w, y1, y2, x1, x2))
            cv2.imwrite(os.path.join(out_imgd, out_name), crop_img)
            crop_gt = gt[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(out_gtd, out_name.replace('.jpg', gt_type)), crop_gt)

