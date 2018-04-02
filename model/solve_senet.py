import sys
caffe_root = '/home/bidlc/caffe/python'
sys.path.insert(0, caffe_root)
import caffe
import numpy as np
import os

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

caffe.set_mode_gpu()
solver = caffe.SGDSolver('solver.prototxt')

max_iters = 20000
# loss = np.zeros([max_iters])
for i in range(max_iters):
    solver.step(1)
    # loss[i] = solver.net.blobs['loss'].data
# _,ax1 = plt.subplots()
# ax1.plot(np.arange(max_iters),loss)
# ax1.set_xlabel('iteration')
# ax1.set_ylabel('train loss')
# plt.show()


