from PIL import Image
import numpy as np
import os, sys

ind = sys.argv[1]
outd = sys.argv[2]
cls_num = int(sys.argv[3])

os.makedirs(outd, exist_ok=True)

recorder = {}
for f in os.listdir(ind):
    cam = np.load(os.path.join(ind, f))
    fn, h, w, y1, y2, x1, x2 = f.strip('.npy').split('_')
    if fn not in recorder:
        recorder[fn] = [np.zeros((cls_num, int(h), int(w))), np.zeros((cls_num, int(h), int(w)))]
    recorder[fn][0][:, int(y1):int(y2), int(x1):int(x2)] += cam
    recorder[fn][1][:, int(y1):int(y2), int(x1):int(x2)] += 1

for k, v in recorder.items():
    cam = v[0] / v[1]
    pred = cam.argmax(0).astype('uint8')
    pred = Image.fromarray(pred)
    pred.save(os.path.join(outd, k + '.png'))
