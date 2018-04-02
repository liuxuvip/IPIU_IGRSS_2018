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
solver = caffe.SGDSolver('solver_fusion.prototxt')


for _ in range(10):
    solver.step(5000)
