import sys
import cv2
import numpy as np
from PIL import Image
from skimage import io
import random
import os
from matplotlib import pyplot
from collections import Counter
from tqdm import tqdm


predict_fusion_label = io.imread('./model/result/image_predict_fusion.tif') + 1
predict_senet_label = io.imread('./model/result/image_predict_senet.tif') + 1

other_cate = [11,12,13,14,15,16]
for i in range(len(other_cate)):
    cate = other_cate[i]
    index = np.where(predict_fusion_label==cate)
    predict_fusion_label[index]=cate+4
road_cate = 10
index0 = np.where(predict_fusion_label==road_cate)[0]
index1 = np.where(predict_fusion_label==road_cate)[1]
for j in range(len(index0)):
    x = index0[j]
    y = index1[j]
    predict_fusion_label[x,y]=predict_senet_label[x,y]+9

img = cv2.GaussianBlur(predict_fusion_label,(5,5),0)

io.imsave('./final_result.tif', img)

