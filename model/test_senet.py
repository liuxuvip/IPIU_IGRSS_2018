#coding=utf-8
import sys
caffe_root =  '/home/bidlc/caffe/python'
sys.path.insert(0, caffe_root)
import caffe
import numpy as np
from skimage import io
import os
import scipy
from tqdm import tqdm

caffe.set_mode_gpu()
weights = './snapshot_senet/senet_iter_20000.caffemodel'
solver_name = './val_senet.prototxt'
net = caffe.Net(solver_name, weights,caffe.TEST)

image_w = 8344
image_h = 2404
image_pdt = np.zeros((image_h,image_w))
image_pdt_score = np.zeros((image_h,image_w))
cut = 32
batchsize = 800
all_num = int(np.ceil(image_h*1.0/batchsize)*image_w)

index = 0
col =0
for idx in tqdm(range(all_num)):
    net.forward()
    n_data = net.blobs['classifier'].data
    class_num = n_data.argmax(axis = 1)
    class_score = n_data.max(axis = 1)
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

io.imsave('./result/image_predict_senet.tif',image_pdt)
io.imsave('./result/image_scores_senet.tif',image_pdt_score)


