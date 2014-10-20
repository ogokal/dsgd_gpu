import numpy as np
from bitarray import *
import random
import math
import logging
from bitmatrix import bitmatrix


data = np.array(range(1,11)) * np.ones((10,10))


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

    def decideStartBlock(self):
        xIdx = random.randint(0, self.xBLockCount)        
        yIdx = random.randint(0, self.yBlockCount)
        logging.debug("decidedStartBlock: {0}".format((xIdx, yIdx)))
        return (xIdx, yIdx)
        
        
    def printstratum(self, stratum):
        for s in stratum.substratum:
            if s.isExcess:
                logging.debug("For excessive stratum: %d", stratum.straIdx)
                if s.x >= self.xBLockCount:
                    rowStart = self.h * self.xBLockCount - 1
                else:
                    rowStart = s.x * self.h
                if s.y >= self.yBlockCount:
                    colStart = self.w * self.yBlockCount - 1
                else:
                    colStart = colStart = s.y * self.w
            else:
                logging.debug("For stratum : %d", stratum.straIdx)
                rowStart = s.x * self.h 
                colStart = s.y * self.w
                
            sMat = self.v[rowStart:rowStart+self.h, colStart:colStart+self.w]
            logging.debug(sMat)
            
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
            
    def extractCrossStratums(self, testStartBlock=None):
        stratums = []
        if testStartBlock:
            x,y = testStartBlock
        else:
            x,y = self.decideStartBlock()
        xVar = x
        yVar = y
        stra = stratum(self.stratumIdx)
        #to bottom right
        if xVar+1 < self.xBLockCount and yVar+1 < self.yBlockCount: 
            while xVar < self.xBLockCount and yVar < self.yBlockCount:
                stra.substratum.append(substratum(xVar, yVar))
                xVar = xVar + 1 
                yVar = yVar + 1
            
        xVar = x
        yVar = y
        #to top left
        if xVar-1 >= 0  and yVar-1 >= 0:
            while xVar >= 0  and yVar >= 0:
                stra.substratum.append(substratum(xVar, yVar))
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
        if xVar-1 >= 0 and yVar+1 < self.yBlockCount:
            while xVar >= 0 and yVar < self.yBlockCount:
                stra.substratum.append(substratum(xVar, yVar))
                xVar = xVar - 1
                yVar = yVar + 1
            
        xVar = x
        yVar = y
        #to bottom left
        if xVar+1 < self.xBLockCount  and yVar-1 >= 0 :
            while xVar < self.xBLockCount  and yVar >= 0 :
                stra.substratum.append(substratum(xVar, yVar))
                xVar = xVar + 1
                yVar = yVar - 1
        
        if len(stra.substratum) > 0:
            stratums.append(stra)
            self.stratumIdx = self.stratumIdx + 1
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                self.printstratum(stra)       
        return stratums