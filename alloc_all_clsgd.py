import numpy as np
import cltools
import stratum
import kernels
from Queue import Queue
from multiprocessing.dummy import Pool

class GpuSgdContext:
    def __init__(self,wRSize, hCSize, commonDimSize, tool,customDtype):
        self.tool = tool
        self.vBuf = self.tool.readwriteBuffer(v)
        self.M = int(wRSize)
        self.N = int(hCSize)
        self.K = commonDimSize
        self.customDtype = customDtype
        
    def trainStratum(self, stratum):
        test = np.zeros((kernels.BLOCK_WID, kernels.BLOCK_HEI),dtype=np.float32)        
        testBuf = self.tool.readwriteBuffer(test)
        sbstratum = np.array(stratum.substratum,dtype=self.customDtype)
        sbstratumBuf = self.tool.readonlyBuffer(sbstratum)
        self.tool.call('trainAllocSubstratum',(int(sbstratum.shape[0])*kernels.BLOCK_WID,kernels.BLOCK_HEI), \
                        (kernels.BLOCK_WID, kernels.BLOCK_HEI), \
                        self.vBuf,
                        np.int32(self.N) 
                        ,sbstratumBuf,testBuf)
        self.tool.enqueueCopy(test,testBuf)
        print test

class GpuDsgd:
    
    def __init__(self, wRSize, hCSize, commonDimSize):
        self.tools = cltools.ClTools().gpus
        self.customDtype = self.make_stratumMeta_dtype()
        self.sqdContextDict = {}
        for t in self.tools.keys():
            self.sqdContextDict[t] = GpuSgdContext(v,self.tools[t],self.customDtype)
        self.straExtr = stratum.stratumExtractor(v,kernels.BLOCK_HEI, kernels.BLOCK_WID,preAllocedData=True)

    def make_stratumMeta_dtype(self):
        dtype = np.dtype([
            ('x', np.int32),
            ('y', np.int32),
            ('rStart',np.int32),
            ('cStart', np.int32)
        ])
        name = "stratumMeta"
        from pyopencl.tools import get_or_register_dtype
        dtype = get_or_register_dtype(name, dtype)
        return dtype        
    
    def __runStratum__(self,stratum):
        print stratum
    
    def run(self):
        gpuQue = Queue()
        for t in self.tools.keys():
            gpuQue.put(t)
        for stra in self.straExtr.nextStratum(asNamed=True):
            targetGpu = gpuQue.get()
            self.sqdContextDict[targetGpu].trainStratum(stra)
#            print targetGpu
#            print stra.straIdx
#            for s in stra.substratum:
#                print s
            for t in self.tools.keys():
                gpuQue.put(t)
    
    def run3(self):
        gpuQue = Queue()
        stratumQueue = Queue(2)
        for t in self.tools.keys():
            gpuQue.put(t)
            
        straIter = self.straExtr.nextStratum(asNamed=True, customDtype=self.customDtype)
        try:
            for t in self.tools.keys(): 
                stratumQueue.put(straIter.next())
        except:
            pass
        
        while not stratumQueue.empty():
            stra = stratumQueue.get()
            targetGpu = gpuQue.get()
#            self.sqdContextDict[targetGpu]
            print targetGpu
            print stra.straIdx
            for s in stra.substratum:
                print s
            gpuQue.put(targetGpu)
            try:
                stratumQueue.put(straIter.next())
            except:
                pass
                
                
    
        
class GpuSgd:
    def __init__(self, sgdContext):
        self.sgdContext = sgdContext
        