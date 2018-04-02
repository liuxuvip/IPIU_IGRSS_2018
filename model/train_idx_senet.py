#coding=utf-8
import numpy as np
import math
from skimage import io
import random
gt_label = io.imread("./gt5_road.tif")
label_path = io.imread('./road_detection/Canny_result.tif')

label_num = 6
erery_cut = 15000
erery_cut_high = 8000
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
    if ci==5:
        erery_cut = erery_cut_high

    if length>=erery_cut:
        select_index=random.sample(range(0, length), erery_cut)
    else:

        select_index1 = random.sample(range(0, length), length)
        if erery_cut-length>length:
            select_index2 = random.sample(range(0, length), length)
        else:
            select_index2 = random.sample(range(0, length), erery_cut-length)
        select_index = select_index1+select_index2
    for i in range(len(select_index)):
        org_x = index1[select_index[i]]
        org_y = index2[select_index[i]]
        data_txt.append((org_y+1192,org_x+1202,ci-1))
        idx=idx+1

index_high1 = np.where(label_path==1)[0]
index_high2 = np.where(label_path==1)[1]
length = len(index_high1)
select_index1 = random.sample(range(0, length), 7000)
for i in range(len(select_index1)):
    org_x = index_high1[select_index1[i]]
    org_y = index_high2[select_index1[i]]
    data_txt.append((org_y,org_x,4))
    idx=idx+1

print (idx)
num = len(data_txt)
index = np.arange(0,num,1)
random.shuffle(index)
data_txt=np.array(data_txt)
data_txt = data_txt[index]

np.savetxt("trainidx_senet15000.txt",data_txt, fmt="%d")