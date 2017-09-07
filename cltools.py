import pyopencl as cl
from kernels import kernels_scr
import logging



class ClTool:
    
    def __init__(self,name, platform, device, context, queue,sharedMemo=False):
        self.name = name
        self.platform = platform
        self.device = device
        self.context = context
        self.queue = queue
        self.sharedMemo =  sharedMemo
        self.program = cl.Program(self.context, kernels_scr).build()
        self.methods = dict((kernel.function_name, kernel) for kernel in self.program.all_kernels())        
    
    def readonlyBuffer(self,hostObj):
        return cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=hostObj)

    def writeonlyBuffer(self,hostObj):
        if self.sharedMemo:
            return cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR, hostObj.nbytes)
        return cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, hostObj.nbytes)
        
    def readwriteBuffer(self, hostObj):
        if self.sharedMemo:
            return cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR, hostbuf=hostObj)
        return cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=hostObj)
    
    def call(self, methodName,*params):
        params = (self.queue,)+params
        return self.methods[methodName](*params)
        
    def enqueueCopy(self,dest,src):
        return cl.enqueue_copy(self.queue,dest,src)
        
class ClTools:
    INTEL = 'INTEL'
    NVIDIA = 'NVIDIA'
    AMD = 'AMD'
    
    def __init__(self):
        self.gpus = {}
        platforms = cl.get_platforms()
        for p in platforms:
            plat_devices = p.get_devices(device_type=cl.device_type.GPU)
            if plat_devices:
                if p.vendor.upper().find(ClTools.INTEL) != -1:
                    intelContext = cl.Context(devices=plat_devices)
                    intelQueue = cl.CommandQueue(intelContext,device=plat_devices[0])
                    self.gpus[ClTools.INTEL] = ClTool(ClTools.INTEL,p,plat_devices[0],intelContext,intelQueue,True)
                    logging.debug("Intel gpu found")
                    logging.debug("Intel Global Memory size {0}".format(plat_devices[0].get_info(cl.device_info.GLOBAL_MEM_SIZE) / 10**6))
                elif p.vendor.upper().find(ClTools.NVIDIA) != -1:
                    nvidiaContext = cl.Context(devices=plat_devices)
                    nvidiaQueue = cl.CommandQueue(nvidiaContext,device=plat_devices[0])
                    self.gpus[ClTools.NVIDIA] = ClTool(ClTools.NVIDIA,p,plat_devices[0],nvidiaContext,nvidiaQueue)
                    logging.debug("Nvidia gpu found")
                    logging.debug("Nvidia Global Memory size {0}".format(plat_devices[0].get_info(cl.device_info.GLOBAL_MEM_SIZE) / 10**6))
                elif p.vendor.upper().find(ClTools.AMD) != -1:
                    amdContext = cl.Context(devices=plat_devices)
                    amdQueue = cl.CommandQueue(amdContext,device=plat_devices[0])
                    self.gpus[ClTools.AMD] = ClTool(ClTools.AMD,p,plat_devices[0],amdContext,amdQueue)
                    logging.debug("Amd gpu found")
                    logging.debug("Amd Global Memory size {0}".format(plat_devices[0].get_info(cl.device_info.GLOBAL_MEM_SIZE) / 10**6))

            
    def __getitem__(self, idx):
        if idx == ClTools.AMD:
            return self.gpus[ClTools.AMD]
        elif idx == ClTools.INTEL:
            return self.gpus[ClTools.INTEL]
        elif idx == ClTools.NVIDIA:
            return self.gpus[ClTools.NVIDIA]


