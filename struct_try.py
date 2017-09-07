
import struct
import cltools
import pyopencl as cl
import pyopencl.reduction
import numpy as np

def make_stratumMeta_dtype(device):
    dtype = np.dtype([
        ("x", np.int32),
        ("y", np.int32),
        ("data", np.float32, (1,2))    
    ])
    name = "stratumMeta"
    from pyopencl.tools import get_or_register_dtype
    dtype = get_or_register_dtype(name, dtype)
    return dtype
#
#class stratumMeta:
#    
#    def __init__(self,x,y):
#        self.x = x
#        self.y = y
#    
#    def pack(self):
#        return struct.pack("=ii",self.x,self.y)

tools = cltools.ClTools()
intel = tools[tools.INTEL]
stradtype = make_stratumMeta_dtype(intel.device)
#'int32,int32,(1,2)float32'



r = np.array([(10,20,np.array([[30.0,40.0]],dtype=np.float32))], dtype=stradtype)

rbuf = cl.Buffer(intel.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=r)

w = np.empty_like(r)
wbuf = cl.Buffer(intel.context, cl.mem_flags.READ_WRITE, w.nbytes)

intel.call('test',(1,1),None,rbuf, wbuf)

m = np.array([(0,0,np.zeros((1,2),dtype=np.float32))], dtype=stradtype)

evt = cl.enqueue_copy(intel.queue,m,wbuf)

print m

