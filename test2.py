

#import cltools
#import pyopencl as cl
#import pyopencl.clrandom
import numpy as np

print np.zeros((1,2))
#import time
#import logging

#logging.basicConfig(level=logging.DEBUG)

#tools = cltools.ClTools()
#gpu = tools[cltools.ClTools.INTEL]
#import numpy as np
#a = np.random.rand(5).astype(np.float32)
#c = a.copy()
#b = np.empty_like(a)
#a_buf = gpu.readwriteBuffer(a)
#gpu.call('doublify',a.shape, None, a_buf)
#gpu.enqueueCopy(b, a_buf)
#
#print c[:5]
#print b[:5]

#tools = cltools.ClTools()




#__global float *w, __const float eta,
#        __const int vHei, __global float *h,
#        __global float *v, __const int hWid
