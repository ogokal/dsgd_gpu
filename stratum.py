import numpy as np
from bitarray import bitarray
import random
import math
import logging
from bitmatrix import bitmatrix
import threading
from Queue import Queue

BLOCK_HEI = 3
BLOCK_WID = 3
COMMON_DIM = 2

class substratum:
    def __init__(self, x, y,isExcess=False, data=None):
        self.x = x
        self.y = y
        self.isExcess = isExcess
        self.data = data
        self.w = np.random.rand(BLOCK_HEI, COMMON_DIM).astype(np.float32)
        self.h = np.random.rand(COMMON_DIM, BLOCK_WID).astype(np.float32)
        self.customDtype = self.make_stratumMeta_dtype()
        self.rStart = -1
        self.cStart = -1
        
    def setDataCoor(self, calVSlice):
        self.rStart = calVSlice[0].start
        self.cStart = calVSlice[1].start
        
    def toNamed(self):
        namedTuple = (self.x,self.y)
        if self.data != None:
            namedTuple = namedTuple + (self.data,)
        namedTuple = namedTuple + (self.rStart, self.cStart,self.w, self.h,)
        return np.array([namedTuple], dtype=self.customDtype)
        
    def make_stratumMeta_dtype(self):
        dtype = np.dtype([
            ('x', np.int32),
            ('y', np.int32),
            ('rStart',np.int32),
            ('cStart', np.int32),
            ('w', np.float32, (int(self.w.shape[0]),int(self.w.shape[1]))),
            ('h', np.float32, (int(self.h.shape[0]),int(self.h.shape[1])))
        ])
        name = "stratumMeta"
        from pyopencl.tools import get_or_register_dtype
        dtype = get_or_register_dtype(name, dtype)
        return dtype 
    
        
class stratum:
    def __init__(self, straIdx, hasExcess=False):
        self.substratum = []
        self.straIdx = straIdx
        self.hasExcess = hasExcess
        
        
class stratumExtractor:
    
    def __init__(self, v, blockHei, blockWid, preAllocedData=False):
        self.v = v
        self.M = v.shape[0]
        self.N = v.shape[1]
        self.blockHei = blockHei
        self.blockWid = blockWid
        self.xBLockCount =  int(math.ceil(self.M / self.blockHei))
        self.yBlockCount = int(math.ceil(self.N / self.blockWid))
        self.hasExcessXBLock = (self.M % self.blockHei != 0)
        self.hasExcessYBlock = (self.N % self.blockWid != 0)
        self.check = bitmatrix(bArrayVal=bitarray(self.xBLockCount * self.yBlockCount), \
                               shape=(self.xBLockCount, self.yBlockCount), locked=True)
        self.check.setall(0)
        self.stratumIdx  = 0
        self.lock = threading.Lock()
        self.preAllocedData = preAllocedData

    def calculateVSlice(self, substrat):
        if substrat.isExcess:
            if substrat.x >= self.xBLockCount:
                rowStart = self.blockHei * self.xBLockCount - 1
            else:
                rowStart = substrat.x * self.blockHei
            if substrat.y >= self.yBlockCount:
                colStart = self.blockWid * self.yBlockCount - 1
            else:
                colStart = colStart = substrat.y * self.blockWid
        else:
            rowStart = substrat.x * self.blockHei 
            colStart = substrat.y * self.blockWid
        return (slice(rowStart, rowStart+self.blockHei), slice(colStart, colStart+self.blockWid))
            
        
    def nextStratum(self,includeExcessive=True,asNamed=False):
        with self.lock:        
            completed = False
            q = Queue()
            targetBlock = self.check.findfirst()
            if targetBlock:
                for s in self.extractCrossStratums(targetBlock,asNamed):
                    q.put(s)
            while not q.empty():
                targetBlock = self.check.findfirst()
                if targetBlock:
                    for s in self.extractCrossStratums(targetBlock,asNamed):
                        q.put(s)
                elif includeExcessive:
                    if not completed:
                        for s in self.extractExcessiveStratums(asNamed):
                            q.put(s)
                        completed = True
                yield q.get()
            
            
        
    def decideStartBlock(self):
        xIdx = random.randint(0, self.xBLockCount)        
        yIdx = random.randint(0, self.yBlockCount)
        logging.debug("decidedStartBlock: {0}".format((xIdx, yIdx)))
        return (xIdx, yIdx)
        
    def castStratumAsData(self, stratum):
        sMatList = []
        for s in stratum.substratum:
            slices = self.calculateVSlice(s)
            sMat = self.v[slices[0], slices[1]]
            sMatList.append(sMat)
        return sMatList
            
    def printstratum(self, stratum):
        sMatList = self.castStratumAsData(stratum)
        for s in sMatList:
            print s
            
    def extractExcessiveStratums(self, asNamed=False):
        stratums = []
        if self.hasExcessXBLock:
            for y in range(self.yBlockCount):
                s = stratum(self.stratumIdx,True)
                sbs = substratum(self.xBLockCount, y, True)
                slices = self.calculateVSlice(sbs)
                sbs.setDataCoor(slices)
                if not self.preAllocedData:    
                    sbs.data = self.v[slices[0], slices[1]]
                if asNamed:
                    s.substratum.append(sbs.toNamed())
                else:
                    s.substratum.append(sbs)
                stratums.append(s)
                self.stratumIdx = self.stratumIdx + 1
                if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                    self.printstratum(s)
        if self.hasExcessYBlock:
            for x in range(self.xBLockCount):
                s = stratum(self.stratumIdx, True)
                sbs = substratum(x, self.yBlockCount, True)
                sbs.setDataCoor(slices)                
                if not self.preAllocedData:
                    sbs.data = self.v[slices[0], slices[1]]
                if asNamed:
                    s.substratum.append(sbs.toNamed())
                else:
                    s.substratum.append(sbs)
                stratums.append(s)
                self.stratumIdx = self.stratumIdx + 1
                if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                    self.printstratum(s)
        return stratums
            
    def extractCrossStratums(self, startBlock, asNamed=False):
        stratums = []
        x,y = startBlock
        xVar = x
        yVar = y
        stra = stratum(self.stratumIdx)
        #to bottom right
#        if xVar+1 < self.xBLockCount and yVar+1 < self.yBlockCount: 
        while xVar < self.xBLockCount and yVar < self.yBlockCount:
            if not self.check[xVar,yVar]:
                s = substratum(xVar, yVar)
                slices = self.calculateVSlice(s)
                s.setDataCoor(slices)
                if not self.preAllocedData:
                    s.data = self.v[slices[0], slices[1]]
                if asNamed:
                    stra.substratum.append(s.toNamed())
                else:
                    stra.substratum.append(s)
                self.check[xVar,yVar] = True
            xVar = xVar + 1 
            yVar = yVar + 1
            
        xVar = x
        yVar = y
        #to top left
#        if xVar-1 >= 0  and yVar-1 >= 0:
        while xVar >= 0  and yVar >= 0:
            if not self.check[xVar,yVar]:
                s = substratum(xVar, yVar)
                slices = self.calculateVSlice(s)
                s.setDataCoor(slices) 
                if not self.preAllocedData:
                    s.data = self.v[slices[0], slices[1]]
                if asNamed:
                    stra.substratum.append(s.toNamed())
                else:
                    stra.substratum.append(s)
                self.check[xVar,yVar] = True
            xVar = xVar - 1
            yVar = yVar - 1
                
        if len(stra.substratum) > 0:
            stratums.append(stra)
            self.stratumIdx = self.stratumIdx + 1
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                self.printstratum(stra)        
        
        xVar = x
        yVar = y
        stra = stratum(self.stratumIdx)
        #to top right
#        if xVar-1 >= 0 and yVar+1 < self.yBlockCount:
        while xVar >= 0 and yVar < self.yBlockCount:
            if not self.check[xVar,yVar]:
                s = substratum(xVar, yVar)
                slices = self.calculateVSlice(s)
                s.setDataCoor(slices) 
                if not self.preAllocedData:
                    s.data = self.v[slices[0], slices[1]]
                if asNamed:
                    stra.substratum.append(s.toNamed())
                else:
                    stra.substratum.append(s)
                self.check[xVar,yVar] = True
            xVar = xVar - 1
            yVar = yVar + 1
            
        xVar = x
        yVar = y
        #to bottom left
#        if xVar+1 < self.xBLockCount  and yVar-1 >= 0 :
        while xVar < self.xBLockCount  and yVar >= 0 :
            if not self.check[xVar,yVar]:
                s = substratum(xVar, yVar)
                slices = self.calculateVSlice(s)
                s.setDataCoor(slices) 
                if not self.preAllocedData:
                    s.data = self.v[slices[0], slices[1]]
                if asNamed:
                    stra.substratum.append(s.toNamed())
                else:
                    stra.substratum.append(s)
                self.check[xVar,yVar] = True
            xVar = xVar + 1
            yVar = yVar - 1
        
        if len(stra.substratum) > 0:
            stratums.append(stra)
            self.stratumIdx = self.stratumIdx + 1
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                self.printstratum(stra)       
        return stratums