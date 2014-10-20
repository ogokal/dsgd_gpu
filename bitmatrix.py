from bitarray import bitarray

class bitmatrix:
    def __init__(self,bArrayVal=None, bMatVal=None, shape=None):
        self.bArray = bArrayVal
        if bMatVal:
            self.bMat = bMatVal
        else:
            self.bMat = []
        if not bArrayVal and bMatVal:
            self.shape = (len(bMatVal), len(bMatVal[0]))
        if bArrayVal and shape:
            self.reshape(shape)
    
    def reshape(self, shape):
        step = shape[1]
        steplimit = shape[0] * shape[1]
        rCount = -1
        for s in xrange(steplimit):
            if s % step == 0:
                rCount = rCount+1
                self.bMat.append(bitarray())
            self.bMat[rCount].append(self.bArray[s])
        self.shape = shape
    
    def __str__(self):
        print_str = []
        for r in xrange(len(self.bMat)):
            for c in xrange (len(self.bMat[r])):
                print_str.append(str(self.bMat[r][c]))
                print_str.append(' ')
            print_str.append("\n")
        return ''.join(print_str)
    
    def __getitem__(self,sli):
        if not isinstance(sli[0],slice) and not isinstance(sli[0],slice):
            if sli[0] >= len(self.bMat):
                raise Exception('out of bound {0} > {1}'.format(sli[0], len(self.bMat)-1))
                
            if sli[1] >= len(self.bMat[0]):
                raise Exception('out of bound {0} > {1}'.format(sli[1], len(self.bMat[0])-1))
            return self.bMat[sli[0]][sli[1]]
        else:
            bMat = []
            boundries = self.__extractSliceBoundries__(sli)
            rStart = boundries[0][0]
            rEnd = boundries[0][1]
            cStart = boundries[1][0]
            cEnd = boundries[1][1]
            rCount = 0
            for r in xrange(rStart,rEnd+1):
                bMat.append(bitarray())
                for c in xrange(cStart,cEnd+1):
                    bMat[rCount].append(self.bMat[r][c])
                rCount = rCount + 1
            return bitmatrix(bMatVal = bMat)

    def __extractSliceBoundries__(self, sli):
        rStart = sli[0]
        rEnd = sli[0]
        cStart = sli[1]
        cEnd = sli[1]
        if isinstance(sli[0],slice):
                if sli[0].start:
                    rStart = sli[0].start
                else: 
                    rStart = 0
                if sli[0].stop:
                    rEnd = sli[0].stop
                else:
                    rEnd = len(self.bMat)-1
        if isinstance(sli[1],slice):
            if sli[1].start:
                cStart = sli[1].start 
            else : 
                cStart = 0
            if sli[1].stop:
                cEnd = sli[1].stop
            else: 
                cEnd = len(self.bMat[0])-1
        return ((rStart,rEnd),(cStart,cEnd))
    
    def __instancecheck__(self,inst):
        if type(inst) == type(bitmatrix):
            return True
    
    def setall(self, val):
        self.__setitem__((slice(None, None, None), slice(None, None, None)),val)
        
    def findfirst(self, val=False):
        found = None
        for r in xrange(len(self.bMat)):
            try:
                found = (r,self.bMat[r].index(val))
                break
            except ValueError:
                pass
        return found
    
    def __setitem__(self, sli, bMatNew):
        boundries = self.__extractSliceBoundries__(sli)
        if isinstance(bMatNew, bitmatrix):
            for r in xrange(bMatNew.shape[0]):
                for c in xrange(bMatNew.shape[1]):
                    if boundries[0][0] + r < self.shape[0] and \
                        boundries[1][0] + c < self.shape[1]:
                            self.bMat[boundries[0][0] + r][boundries[1][0] +c] = bMatNew[r,c]
        elif isinstance(sli, tuple) and not isinstance(sli[0], slice):
            self.bMat[sli[0]][sli[1]] = bMatNew
        else:
            rStart = boundries[0][0]
            rEnd = boundries[0][1]
            cStart = boundries[1][0]
            cEnd = boundries[1][1]
            for r in xrange(rStart, rEnd+1):
                for c in xrange(cStart, cEnd+1):
                    self.bMat[rStart + r][cStart +c] = bMatNew
                    