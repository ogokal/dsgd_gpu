import numpy as np
import pyopencl as cl
from kernels import kernels


a_np = np.random.rand(50).astype(np.float32)
b_np = np.random.rand(50).astype(np.float32)

platforms = cl.get_platforms()
gpu_devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=gpu_devices)
queue = cl.CommandQueue(ctx,device=gpu_devices[0])

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

#prg = cl.Program(ctx, """
#__kernel void sum(__global const float *a_g, __global const float *b_g, __global float *res_g) {
#  int gid = get_global_id(0);
#  res_g[gid] = a_g[gid] + b_g[gid];
#}
#""").build()

#res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
#prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)
#
#res_np = np.empty_like(a_np)
#cl.enqueue_copy(queue, res_np, res_g)
#
## Check on CPU with Numpy:
#print(res_np - (a_np + b_np))
#print(np.linalg.norm(res_np - (a_np + b_np)))

prg = cl.Program(ctx,kernels ).build()