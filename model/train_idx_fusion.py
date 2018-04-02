#coding=utf-8
import numpy as np
import math
from skimage import io
import random



gt_label = io.imread("../gt16_fusion.tif")

label_num = 17
erery_cut = 20000

image_w = gt_label.shape[1]   #4768
image_h = gt_label.shape[0]   #1202
# stop = int(image_h*0.4)
idx = 0
data_txt = []
for ci in range(1,label_num):
    index1 = np.where(gt_label==ci)[0]
    index2 = np.where(gt_label==ci)[1]
    length = len(index1)
    print(ci)
    print(length)
    if length>=erery_cut:
        select_index=random.sample(range(0, length), erery_cut)
    else:
        select_index = random.sample(range(0, length), length)
    for i in range(len(select_index)):
        org_x = index1[select_index[i]]
        org_y = index2[select_index[i]]
        data_txt.append((org_y,org_x))
        idx=idx+1
        # data_txt.append((str(idx),str(org_x),str(org_y)))
print (idx)
num = len(data_txt)
index = np.arange(0,num,1)
random.shuffle(index)
data_txt=np.array(data_txt)
data_txt = data_txt[index]
np.savetxt("./trainidx_fusion20000.txt",data_txt, fmt="%d")