import sys
caffe_root = '/home/bidlc/caffe/python'
sys.path.insert(0, caffe_root)
import caffe
import cv2
import numpy as np
from PIL import Image
from skimage import io
import random

class VOCSegDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        # config PATH
        params = eval(self.param_str)
        self.sbdd_dir = "../data/lidar_data/lidar_test.tif"
        # self.sbdd_dir2_path = "/home/asdf/chenglin/IGRSS_2018/FCN0201/vhr.tif"
        # config
        self.cuts = 32
        self.factor=1
        self.train = params.get('train', None)
        self.idx = 0
        self.seed = params.get('seed', None)
        self.random = params.get('randomize', False)
        self.col = 0
        self.batchsize = params.get('batchsize', None)

        # two tops: data and label
        if len(top) != 1:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        ##  data deal with 48
        self.image_data_org = io.imread(self.sbdd_dir).astype('float32')
        # self.image_data_org = self.image_data_org[800:1229,6060:6796]

        print("resize ok!!")
        self.channel = self.image_data_org.shape[2]
        self.image_w = self.image_data_org.shape[1]   # 4768
        self.image_h = self.image_data_org.shape[0]  # small 1202
        self.image_data = np.zeros([self.image_h + 2 * self.cuts, self.image_w + 2 * self.cuts, self.channel])
        self.image_data[self.cuts:self.image_h + self.cuts, self.cuts:self.image_w + self.cuts, :] = self.image_data_org
        print("hsi ok!!")


        if not self.train:
            self.random = False

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data32 = self.load_image(self.col,self.idx,self.batchsize,32)

        top[0].reshape(*self.data32.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data32

        self.idx = self.idx + self.batchsize
        if self.idx >= self.image_h:
            self.idx = 0
            self.col = self.col+1


    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, col, idx, batchsize, stage_size):
        if idx+batchsize>self.image_h:
            batchsize = self.image_h-idx
        data = np.zeros([batchsize,stage_size,stage_size,self.channel])
        size = self.cuts - stage_size / 2
        for every_i in range(batchsize):
            data[every_i,:,:,:] = self.image_data[idx + size:idx + size + stage_size,col + size:col + size + stage_size,:]
            idx = idx +1
        data = data.transpose((0, 3, 1, 2))
        return data

class SBDDSegDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        # config PATH
        params = eval(self.param_str)
        self.sbdd_dir = "../data/lidar_data/lidar_test.tif"
        # self.sbdd_dir2_path = "/home/asdf/chenglin/IGRSS_2018/FCN0201/vhr.tif"
        self.patchtxt = "./trainidx_senet15000.txt"
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
        if len(top) != 2:
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

        #### label data
        print("set ok")

        self.mini_index = random.sample(range(0, len(self.patchtxt)), self.batchsize)


    def reshape(self, bottom, top):
        # load image + label image pair
        self.label = self.load_label(self.patchtxt, self.mini_index, self.batchsize)
        self.data32 = self.load_image(self.patchtxt,self.mini_index,self.batchsize,32)

        top[0].reshape(*self.label.shape)
        top[1].reshape(*self.data32.shape)



    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.label
        top[1].data[...] = self.data32


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


    # def load_label(self, patchtxt,mini_index,batchsize):
    #     label = np.zeros([batchsize])
    #     for every_i in range(batchsize):
    #         label[every_i] = patchtxt[mini_index[every_i]][2]
    #     return label


    def load_label(self, patchtxt,mini_index,batchsize):
        label = np.zeros([batchsize])
        for every_i in range(batchsize):
            label[every_i] = patchtxt[mini_index[every_i]][2]
        return label