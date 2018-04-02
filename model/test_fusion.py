#coding=utf-8
import sys
caffe_root = '/home/bidlc/caffe/python'
sys.path.insert(0, caffe_root)
import caffe
import numpy as np
from skimage import io
import os
import scipy
import cv2
from tqdm import tqdm
import datetime
caffe.set_mode_gpu()
weights = './snapshot_fusion/fusion_iter_50000.caffemodel'
solver_name = './val_fusion.prototxt'
caffe.set_mode_gpu()
net = caffe.Net(solver_name, weights, caffe.TEST)

image_w = 8344   #4768
image_h = 2404  #1202
image_pdt = np.zeros((image_h,image_w))
image_pdt_score = np.zeros((image_h,image_w))
batchsize = 500
# solver.test_nets[0].share_with(solver.net)
all_num = int(np.ceil(image_h*1.0/batchsize)*image_w)
index = 0
col =0
# net = solver.test_nets[0]
for idx in tqdm(xrange(all_num)):
    # starttime1 = datetime.datetime.now()
    net.forward()
    # endtime1 = datetime.datetime.now()
    # print(endtime1-starttime1).total_seconds()
    n_data = net.blobs['classifier'].data
    class_num = n_data.argmax(axis = 1)
    class_score = n_data.max(axis = 1)
    # print class_num
    if index>image_h-batchsize:
        image_pdt[index:image_h,col] = class_num
        image_pdt_score[index:image_h, col] = class_score
    else:
        image_pdt[index:index+batchsize,col] = class_num
        image_pdt_score[index:index+batchsize, col] = class_score
    index = index + batchsize
    if index>image_h:
        index = 0
        col = col+1

io.imsave('./result/image_predict_fusion.tif',image_pdt)   ### 0~15
io.imsave('./result/image_scores_fusion.tif',image_pdt_score)


