import stratum
import kernels
from Queue import Queue
import cltools
import stratum

class GpuDSgd:
    
    def __init__(self,v):
        self.straExtr = stratum.stratumExtractor(v,stratum.BLOCK_HEI,stratum.BLOCK_WID,preAllocedData=True)
        self.tools = cltools.ClTools().gpus
        self.sqdContextDict = {}
        for t in self.tools.keys():
            self.sqdContextDict[t] = GpuSgdContext(self.tools[t])
        self.straIter = self.straExtr.nextStratum(asNamed=True)
    
        
    def run(self):
        gpuQue = Queue()
        for t in self.tools.keys():
            gpuQue.put(t)
        for stra in self.straIter:
            targetGpu = gpuQue.get()
            self.sqdContextDict[targetGpu].testStra(stra)
            for t in self.tools.keys():
                gpuQue.put(t)

   
                 
 
class GpuSgdContext:
    def __init__(self, tool):
        self.tool = tool
    
    def trainStra(self, stra):
        pass
    
    def testStra(self, stra):
        outsb = stratum.substratum(1,1).toNamed()
        self.tool.call('testStratum', (len(stra.substratum),), None, stra.substratum, outsb)
    
    
        