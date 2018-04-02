import sys
import cv2
import numpy as np
from PIL import Image
from skimage import io
import random
import os
from matplotlib import pyplot
from collections import Counter

### convet hsi to tif
sbdd_dir = "/home/bidlc/Desktop/TestHSI.tif"  ### tif data, size=(2404,8344,50)
image_data_org = io.imread(sbdd_dir).astype('float32')
image_data_org = image_data_org[:, :, 0:48]
resize_c = image_data_org.shape[2]
resize_w = image_data_org.shape[0] * 2   # 2404
resize_h = image_data_org.shape[1] * 2   # 8344
dst = np.zeros([resize_w, resize_h, resize_c])
image_data = np.zeros([resize_w , resize_h , resize_c])
for i in range(resize_c):
    print(i)
    a = cv2.resize(image_data_org[:, :, i], (resize_h, resize_w), interpolation=cv2.INTER_LINEAR)
    min = np.min(a)
    max = np.max(a)
    a = (a - min) / (max - min)
    image_data[:, :, i] = a
channel = resize_c
image_w1 = resize_h*1/7
image_w2 = resize_h*5/7
image_h1 = resize_w*1/2
train_image_data=image_data[image_h1:,image_w1:image_w2,:]
io.imsave('./hsi_data/hsi_train.tif',train_image_data)
io.imsave('./hsi_data/hsi_test.tif',image_data)



