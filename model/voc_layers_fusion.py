import sys
caffe_root = '/home/bidlc/caffe/python'
sys.path.insert(0, caffe_root)
import caffe
import cv2
import numpy as np
from PIL import Image
from skimage import io
import random
import datetime

class VOCSegDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # config PATH
        params = eval(self.param_str)
        self.sbdd_dir = "../data/lidar_data/lidar_test.tif"
        self.sbdd_dir2_path = "../data/hsi_data/hsi_test.tif"
        # config
        self.cuts = 32
        self.train = params.get('train', None)
        self.idx = 0
        self.seed = params.get('seed', None)
        self.random = params.get('randomize', False)
        self.col = 0
        self.batchsize = params.get('batchsize', None)
        # two tops: data and label
        if len(top) != 8:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        ##  data deal with 48
        self.image_data_org = io.imread(self.sbdd_dir).astype('float32')
        print("resize ok!!")
        self.channel = self.image_data_org.shape[2]
        self.image_w = self.image_data_org.shape[1]   # 8344
        self.image_h = self.image_data_org.shape[0]  # small 2404
        self.image_data = np.zeros([self.image_h + 2 * self.cuts, self.image_w + 2 * self.cuts, self.channel])
        self.image_data[self.cuts:self.image_h + self.cuts, self.cuts:self.image_w + self.cuts, :] = self.image_data_org
        print("hsi ok!!")

        #### vhr data
        self.vhr_data_org = io.imread(self.sbdd_dir2_path).astype('float32')
        resize_h = self.vhr_data_org.shape[0]
        resize_w = self.vhr_data_org.shape[1]
        self.channel2 = self.vhr_data_org.shape[2]
        self.vhr_data = np.zeros([resize_h + self.cuts * 2, resize_w + self.cuts * 2, 48])
        self.vhr_data[self.cuts:resize_h + self.cuts, self.cuts:resize_w + self.cuts, :] = self.vhr_data_org
        print("vhr ok!!")
        if not self.train:
            self.random = False

    def reshape(self, bottom, top):

        self.data32 = self.load_image(self.col,self.idx,self.batchsize,32)
        self.data_vhr32 = self.load_image2(self.col, self.idx, self.batchsize,32)
        self.data16 = self.data32[:,:,8:24,8:24]
        self.data_vhr16 = self.data_vhr32[:,:,8:24,8:24]
        self.data8 = self.data16[:,:,4:12,4:12]
        self.data_vhr8 = self.data_vhr16[:,:,4:12,4:12]
        self.data4 = self.data8[:,:,2:6,2:6]
        self.data_vhr4 = self.data_vhr8[:,:,2:6,2:6]

        top[0].reshape(*self.data32.shape)
        top[1].reshape(*self.data_vhr32.shape)
        top[2].reshape(*self.data16.shape)
        top[3].reshape(*self.data_vhr16.shape)
        top[4].reshape(*self.data8.shape)
        top[5].reshape(*self.data_vhr8.shape)
        top[6].reshape(*self.data4.shape)
        top[7].reshape(*self.data_vhr4.shape)


        # print("reshape end")
    def forward(self, bottom, top):

        top[0].data[...] = self.data32
        top[1].data[...] = self.data_vhr32
        top[2].data[...] = self.data16
        top[3].data[...] = self.data_vhr16
        top[4].data[...] = self.data8
        top[5].data[...] = self.data_vhr8
        top[6].data[...] = self.data4
        top[7].data[...] = self.data_vhr4

        self.idx = self.idx + self.batchsize
        if self.idx >= self.image_h:
            self.idx = 0
            self.col = self.col + 1

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, col, idx, batchsize, stage_size):
        if idx+batchsize>self.image_h:
            batchsize = self.image_h-idx
        data = np.zeros([batchsize,stage_size,stage_size,self.channel])
        size = self.cuts - stage_size / 2
        for every_i in xrange(batchsize):
            data[every_i,:,:,:] = self.image_data[idx + size:idx + size + stage_size,col + size:col + size + stage_size,:]
            idx = idx +1
        data = data.transpose((0, 3, 1, 2))
        return data

    def load_image2(self, col, idx, batchsize, stage_size):
        if idx+batchsize>self.image_h:
            batchsize = self.image_h-idx
        data = np.zeros([batchsize,stage_size,stage_size,self.channel2])
        size = self.cuts - stage_size / 2
        for every_i in xrange(batchsize):
            data[every_i,:,:,:] = self.vhr_data[idx+size:idx+size+stage_size,col+size:col+size+stage_size,:]
            idx = idx +1
        data = data.transpose((0, 3, 1, 2))
        return data

class SBDDSegDataLayer(caffe.Layer):

    def setup(self, bottom, top):

        # config PATH
        params = eval(self.param_str)

        #### label_path
        self.label_dir = "../gt16_fusion.tif"

        self.sbdd_dir = "../data/lidar_data/lidar_train.tif"
        self.sbdd_dir2_path = "../data/hsi_data/hsi_train.tif"
        self.patchtxt = "./trainidx_fusion20000.txt"

        # config
        self.cuts = 32
        self.factor=1
        self.batchsize =  params.get('batchsize', None)
        self.train = params.get('train', None)
        self.idx = 0
        self.seed = params.get('seed', None)
        self.random = params.get('randomize', True)
        self.patchtxt = np.loadtxt(self.patchtxt, dtype=int)
        # two tops: data and label
        if len(top) != 9:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        ##  data deal with 48
        self.image_data_org = io.imread(self.sbdd_dir).astype('float32')

        print("resize ok!!")
        self.channel = self.image_data_org.shape[2]
        self.image_w = self.image_data_org.shape[1]   # 4768
        self.image_h = self.image_data_org.shape[0]  # small 1202
        self.image_data = np.zeros([self.image_h + 2 * self.cuts, self.image_w + 2 * self.cuts, self.channel])
        self.image_data[self.cuts:self.image_h + self.cuts, self.cuts:self.image_w + self.cuts, :] = self.image_data_org
        print("hsi ok!!")

        #### vhr data
        self.vhr_data_org = io.imread(self.sbdd_dir2_path).astype('float32')
        resize_h = self.vhr_data_org.shape[0]
        resize_w = self.vhr_data_org.shape[1]
        self.channel2 = self.vhr_data_org.shape[2]

        self.vhr_data = np.zeros([resize_h + self.cuts * 2, resize_w + self.cuts * 2, 48])
        self.vhr_data[self.cuts:resize_h + self.cuts, self.cuts:resize_w + self.cuts, :] = self.vhr_data_org
        print("vhr ok!!")

        #### label data
        self.label_data = io.imread(self.label_dir).astype(np.uint8)
        self.label_data = self.label_data-1
        print("set ok")

        self.mini_index = random.sample(range(0, len(self.patchtxt)), self.batchsize)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.label = self.load_label(self.patchtxt, self.mini_index, self.batchsize)
        self.data32 = self.load_image(self.patchtxt,self.mini_index,self.batchsize,32)
        self.data_vhr32 = self.load_image2(self.patchtxt,self.mini_index,self.batchsize,32)
        self.data16 = self.data32[:,:,8:24,8:24]
        self.data_vhr16 = self.data_vhr32[:,:,8:24,8:24]
        self.data8 = self.data16[:,:,4:12,4:12]
        self.data_vhr8 = self.data_vhr16[:,:,4:12,4:12]
        self.data4 = self.data8[:,:,2:6,2:6]
        self.data_vhr4 = self.data_vhr8[:,:,2:6,2:6]


        top[0].reshape(*self.label.shape)
        top[1].reshape(*self.data32.shape)
        top[2].reshape(*self.data_vhr32.shape)
        top[3].reshape(*self.data16.shape)
        top[4].reshape(*self.data_vhr16.shape)
        top[5].reshape(*self.data8.shape)
        top[6].reshape(*self.data_vhr8.shape)
        top[7].reshape(*self.data4.shape)
        top[8].reshape(*self.data_vhr4.shape)



    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.label
        top[1].data[...] = self.data32
        top[2].data[...] = self.data_vhr32
        top[3].data[...] = self.data16
        top[4].data[...] = self.data_vhr16
        top[5].data[...] = self.data8
        top[6].data[...] = self.data_vhr8
        top[7].data[...] = self.data4
        top[8].data[...] = self.data_vhr4

        if self.random:
            self.mini_index=random.sample(range(0, len(self.patchtxt)), self.batchsize)


    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, patchtxt, mini_index, batchsize, stage_size):
        data = np.zeros([batchsize,stage_size,stage_size,self.channel])
        for every_i in range(batchsize):
            cx = patchtxt[mini_index[every_i]][0]
            cy = patchtxt[mini_index[every_i]][1]
            size = self.cuts - stage_size / 2
            data[every_i, :, :, :] = self.image_data[cy + size:cy + size + stage_size, cx + size:cx + size + stage_size, :]
        data = data.transpose((0, 3, 1, 2))
        return data


    def load_image2(self, patchtxt, mini_index, batchsize, stage_size):
        data = np.zeros([batchsize,stage_size,stage_size,self.channel2])
        for every_i in range(batchsize):
            cx = patchtxt[mini_index[every_i]][0]
            cy = patchtxt[mini_index[every_i]][1]
            size = self.cuts - stage_size / 2
            data[every_i, :, :, :] = self.vhr_data[cy + size:cy + size + stage_size, cx + size:cx + size + stage_size, :]
        data = data.transpose((0, 3, 1, 2))
        return data

    def load_label(self, patchtxt,mini_index,batchsize):
        label = np.zeros([batchsize])
        for every_i in range(batchsize):
            cx = patchtxt[mini_index[every_i]][0]
            cy = patchtxt[mini_index[every_i]][1]
            label[every_i] = self.label_data[cy, cx]
        return label


