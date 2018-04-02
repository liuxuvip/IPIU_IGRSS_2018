import sys
import cv2
import numpy as np
from PIL import Image
from skimage import io
import random
import os
from matplotlib import pyplot
from collections import Counter

#### path for lidar test data
lidar_dir = "/home/bidlc/Desktop/2018IEEE_Contest/Phase2/Lidar GeoTiff Rasters/"

lidar_dem1 = 'DEM+B_C123/UH17_GEM051.tif'
lidar_dem2 = 'DEM_C123_3msr/UH17_GEG051.tif'
lidar_dem3 = 'DEM_C123_TLI/UH17_GEG05.tif'
lidar_dsm = 'DSM_C12/UH17c_GEF051.tif'
lidar_int1 = 'Intensity_C1/UH17_GI1F051.tif'
lidar_int2 = 'Intensity_C2/UH17_GI2F051.tif'
lidar_int3 = 'Intensity_C3/UH17_GI3F051.tif'
files = [lidar_dem1,lidar_dem2,lidar_dem3,lidar_dsm,lidar_int1,lidar_int2,lidar_int3]
image_names = ['UH17_GEM051.tif','UH17_GEG051.tif','UH17_GEG05.tif','UH17c_GEF051.tif','UH17_GI1F051.tif','UH17_GI2F051.tif','UH17_GI3F051.tif']
new_dir = "./lidar_data/"
name_idx = 0
for image_name in files:
    image = io.imread(lidar_dir+image_name)
    image1 = image
    print("ok!!")
    min = np.min(image)
    max = np.max(image)
    print(min)
    print(max)
    index1 = np.where(image==max)[0]
    index2 = np.where(image==max)[1]
    image1[index1,index2]=0
    max_new = np.max(image1)
    print(max_new)

    for i in range(len(index1)):
        x = index1[i]
        y = index2[i]
        if x!=0 and y!=0:
            image[x,y] = image[x-1,y-1]
        else:
            image[x,y] = max_new
    image_name1 = image_names[name_idx]
    new_name = image_name1[:-4]+"_new.tif"
    print(np.max(image))
    print(np.min(image))
    io.imsave(new_dir+new_name,image)
    name_idx = name_idx +1


lidar_dir = "./lidar_data/UH17c_GEF051_new.tif"  ### dsm
image1 = io.imread(lidar_dir)
max = np.max(image1)
min = np.min(image1)
print(max)
print(min)
lidar_dir = "./lidar_data/UH17_GEG051_new.tif"   ### dem TLI
image2 = io.imread(lidar_dir)
max = np.max(image2)
min = np.min(image2)
print(max)
print(min)
image = image1-image2
print(np.max(image))
print(np.min(image))
index = np.where(image<0)
image[index]=0
print(np.max(image))
print(np.min(image))
save_path = "./lidar_data/dsmdem_new.tif"
io.imsave(save_path,image)

imagename = ['dsmdem_new.tif','UH17_GI1F051_new.tif','UH17_GI2F051_new.tif','UH17_GI3F051_new.tif']
lidar_image = np.zeros([2404,8344,4])
i = 0
for image_name in imagename:
    image = io.imread(new_dir+image_name).astype('float32')
    max = np.max(image)
    min = np.min(image)
    image = (image-min)/(max-min)
    lidar_image[:, :, i] = image
    i = i + 1
image_w1 = 8344*1/7
image_w2 = 8344*5/7
image_h1 = 2404*1/2
train_lidar_image=lidar_image[image_h1:,image_w1:image_w2,:]
io.imsave('./lidar_data/lidar_train.tif',train_lidar_image)
io.imsave('./lidar_data/lidar_test.tif',lidar_image)




