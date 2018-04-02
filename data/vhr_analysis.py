import sys
import cv2
import numpy as np
from PIL import Image
from skimage import io
import random
import os
from matplotlib import pyplot
from collections import Counter
### data path for Final RGB HR Imagery
sbdd_dir2_path = "/home/bidlc/Desktop/2018IEEE_Contest/Phase2/Final RGB HR Imagery/"


vhr_name1 = ['UH_NAD83_271460_3289689.tif', 'UH_NAD83_272056_3289689.tif', 'UH_NAD83_272652_3289689.tif',
                 'UH_NAD83_273248_3289689.tif','UH_NAD83_273844_3289689.tif','UH_NAD83_274440_3289689.tif','UH_NAD83_275036_3289689.tif']
vhr_name2 = ['UH_NAD83_271460_3290290.tif', 'UH_NAD83_272056_3290290.tif', 'UH_NAD83_272652_3290290.tif',
                 'UH_NAD83_273248_3290290.tif','UH_NAD83_273844_3290290.tif','UH_NAD83_274440_3290290.tif','UH_NAD83_275036_3290290.tif']
factor = 0.1
vhr_train_name = ['UH_NAD83_272056_3289689.tif', 'UH_NAD83_272652_3289689.tif',
                 'UH_NAD83_273248_3289689.tif','UH_NAD83_273844_3289689.tif',]

vhr_every_data = io.imread(sbdd_dir2_path + vhr_name1[0])

image_w = vhr_every_data.shape[1]
image_h = vhr_every_data.shape[0]
image_c = vhr_every_data.shape[2]

resize_h = int(image_h * factor)
resize_w = int(image_w * factor)
channel2 = 3
vhr_data = np.zeros([resize_h*2, resize_w*7, 3])
for i in range(len(vhr_name1)):
    print(i)
    vhr_every_data = io.imread(sbdd_dir2_path + vhr_name2[i])
    step2 = int(image_h * factor)
    for j in range(3):
        a = cv2.resize(vhr_every_data[:, :, j], (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        vhr_data[:step2,resize_w*i:resize_w*(i+1), j] = a
    vhr_every_data = io.imread(sbdd_dir2_path + vhr_name1[i])
    for j in range(3):
        a = cv2.resize(vhr_every_data[:, :, j], (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        vhr_data[step2:,resize_w*i:resize_w*(i+1), j] = a
save_name = './vhr_data/vhr_test.tif'
vhr_data = vhr_data.astype(np.uint8)
io.imsave(save_name, vhr_data)


vhr_train_data = np.zeros([resize_h, resize_w*4, 3])
for i in range(len(vhr_train_name)):
    print(i)
    vhr_every_data = io.imread(sbdd_dir2_path + vhr_train_name[i])
    for j in range(3):
        a = cv2.resize(vhr_every_data[:, :, j], (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        vhr_train_data[:, resize_w*i:resize_w*(i+1), j] = a
save_name = './vhr_data/vhr_train.tif'
vhr_train_data = vhr_train_data.astype(np.uint8)
io.imsave(save_name, vhr_train_data)
