import cv2, sys, os
from PIL import Image
import numpy as np

pred = sys.argv[1]
gt = sys.argv[2]
cls_num = int(sys.argv[3])

recorder = [[0, 0] for _ in range(cls_num)]
for f in os.listdir(pred):
    p = np.array(Image.open(os.path.join(pred, f)))
    g = cv2.imread(os.path.join(gt, f))[:, :, 0]
    for i in range(cls_num):
        recorder[i][0] += ((p == i) * (g == i)).sum()
        recorder[i][1] += (((p == i) + (g == i)) > 0).sum()

v = 0
for i in range(cls_num):
    v += recorder[i][0] / recorder[i][1]
mIoU = v / cls_num
dice = 2 * mIoU / (1 + mIoU)

print('mIoU ' + str(mIoU) + ', dice ' + str(dice))
