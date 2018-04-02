import sys
import cv2
import numpy as np
from PIL import Image
from skimage import io
import random
import os
from matplotlib import pyplot
from collections import Counter

### gt path
gt_path = '/home/bidlc/Desktop/2018IEEE_Contest/Phase2/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif'
gt_label=io.imread(gt_path)

road_cate = [10,11,12,13,14]
for i in range(len(road_cate)):
    cate = road_cate[i]
    index = np.where(gt_label==cate)
    gt_label[index]=10

other_cate = [15,16,17,18,19,20]
for i in range(len(other_cate)):
    cate = other_cate[i]
    index = np.where(gt_label==cate)
    gt_label[index]=cate-4

print(np.max(gt_label))
print(np.min(gt_label))

io.imsave('./gt16_fusion.tif',gt_label)


gt_label=io.imread(gt_path)
class_num = 20
for i in range(class_num):
    index = np.where(gt_label==i+1)
    if i == 10:
        gt_label[index]=1
    elif i == 11:
        gt_label[index]=2
    elif i == 12:
        gt_label[index]=3
    elif i == 13:
        gt_label[index]=4
    elif i == 14:
        gt_label[index]=5
    else:
        gt_label[index]=0

print(np.max(gt_label))
print(np.min(gt_label))
io.imsave('./gt5_road.tif',gt_label)

