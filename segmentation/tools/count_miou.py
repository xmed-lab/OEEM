import cv2, sys, os

pred = sys.argv[1]
gt = sys.argv[2]
cls_num = int(sys.argv[3])

recorder = [[0, 0] for _ in range(cls_num)]
for f in os.listdir(pred):
    p = cv2.imread(os.path.join(pred, f))[:, :, 0]
    g = cv2.imread(os.path.join(gt, f))[:, :, 0]
    for i in range(cls_num):
        recorder[i][0] += ((p == i) * (g == i)).sum()
        recorder[i][1] += (((p == i) + (g == i)) > 0).sum()

v = 0
for i in range(cls_num):
    v += recorder[i][0] / recorder[i][1]
print(v / cls_num)
