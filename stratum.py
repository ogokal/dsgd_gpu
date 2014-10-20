import numpy as np
from bitarray import bitarray
import random
import math
import logging
from bitmatrix import bitmatrix
import threading


class substratum:
    def __init__(self, x, y,isExcess=False):
        self.x = x
        self.y = y
        self.isExcess = isExcess

        
class stratum:
    def __init__(self, straIdx, hasExcess=False):
        self.substratum = []
        self.straIdx = straIdx
        self.hasExcess = hasExcess
        
        
class stratumExtractor:
    
    def __init__(self, v, h, w):
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            self.v = v
        self.M = v.shape[0]
        self.N = v.shape[1]
        self.h = h
        self.w = w
        self.xBLockCount =  int(math.ceil(self.M / self.h))
        self.yBlockCount = int(math.ceil(self.N / self.w))
        self.hasExcessXBLock = (self.M % self.h != 0)
        self.hasExcessYBlock = (self.N % self.w != 0)
        self.check = bitmatrix(bArrayVal=bitarray(self.xBLockCount * self.yBlockCount), \
                               shape=(self.xBLockCount, self.yBlockCount))
        self.check.setall(0)
        self.stratumIdx  = 0
        self.lock = threading.Lock()

    def calculateVSlice(self, substrat):
        if substrat.isExcess:
            if substrat.x >= self.xBLockCount:
                rowStart = self.h * self.xBLockCount - 1
            else:
                rowStart = substrat.x * self.h
            if substrat.y >= self.yBlockCount:
                colStart = self.w * self.yBlockCount - 1
            else:
                colStart = colStart = substrat.y * self.w
        else:
            rowStart = substrat.x * self.h 
            colStart = substrat.y * self.w
        return (slice(rowStart, rowStart+self.h), slice(colStart, colStart+self.w))
            
        
    def nextStratum(self,includeExcessive=True):        
        found = True
#        with self.lock:
        while found:
            targetBlock = self.check.findfirst()
            if targetBlock:
                yield self.extractCrossStratums(targetBlock)
            elif includeExcessive:
                found = False
                yield self.extractExcessiveStratums()
            
        
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
            
    def extractExcessiveStratums(self):
        stratums = []
        if self.hasExcessXBLock:
            for y in range(self.yBlockCount):
                s = stratum(self.stratumIdx,True)
                s.substratum.append(substratum(self.xBLockCount, y, True))
                stratums.append(s)
                self.stratumIdx = self.stratumIdx + 1
                if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                    self.printstratum(s)
        if self.hasExcessYBlock:
            for x in range(self.xBLockCount):
                s = stratum(self.stratumIdx, True)
                s.substratum.append(substratum(x, self.yBlockCount, True))
                stratums.append(s)
                self.stratumIdx = self.stratumIdx + 1
                if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                    self.printstratum(s)
        return stratums
            
    def extractCrossStratums(self, startBlock):
        stratums = []
        x,y = startBlock
        xVar = x
        yVar = y
        stra = stratum(self.stratumIdx)
        #to bottom right
#        if xVar+1 < self.xBLockCount and yVar+1 < self.yBlockCount: 
        while xVar < self.xBLockCount and yVar < self.yBlockCount:
            if not self.check[xVar,yVar]:
                stra.substratum.append(substratum(xVar, yVar))
                self.check[xVar,yVar] = True
            xVar = xVar + 1 
            yVar = yVar + 1
            
        xVar = x
        yVar = y
        #to top left
#        if xVar-1 >= 0  and yVar-1 >= 0:
        while xVar >= 0  and yVar >= 0:
            if not self.check[xVar,yVar]:
                stra.substratum.append(substratum(xVar, yVar))
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
                stra.substratum.append(substratum(xVar, yVar))
                self.check[xVar,yVar] = True
            xVar = xVar - 1
            yVar = yVar + 1
            
        xVar = x
        yVar = y
        #to bottom left
#        if xVar+1 < self.xBLockCount  and yVar-1 >= 0 :
        while xVar < self.xBLockCount  and yVar >= 0 :
            if not self.check[xVar,yVar]:
                stra.substratum.append(substratum(xVar, yVar))
                self.check[xVar,yVar] = True
            xVar = xVar + 1
            yVar = yVar - 1
        
        if len(stra.substratum) > 0:
            stratums.append(stra)
            self.stratumIdx = self.stratumIdx + 1
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                self.printstratum(stra)       
        return stratums