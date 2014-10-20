import cpusgd
import numpy as np
import logging
from bitarray import bitarray
from bitmatrix import bitmatrix
import dsgd_cpu

#logging.basicConfig(level=logging.DEBUG)

vMat = np.array([
    [5,3,0,1],
    [4,0,0,1],
    [1,1,0,5],
    [1,0,0,4],
    [0,1,5,4]
],dtype=np.float32)

vMat = np.concatenate((vMat,np.ones((5,2), dtype=np.int32)),axis=1)


#csgd = nmf_sgd_cpu.CSgd(vMat,2)
#csgd.setLossFuns(nmf_sgd_cpu.CSgd.NZSL)
#csgd.trainAll(0.0002,5000)
#print csgd.result()


#import stratum
#ext = stratum.stratumExtractor(vMat,2,2)


#for stra in ext.nextStratum():
#    for s in stra:
#        print s.straIdx
#        ext.printstratum(s)


        
#ext.extractExcessiveStratums()
#print vMat



#b = bitmatrix(bArrayVal=bitarray("1100"),shape=(2,2))
#print b[1,0]
#b[1,0] = True
#print b[1,0]
#c = bitmatrix(bArrayVal=bitarray("01"),shape=(1,2))
#print c
#b.setall(0)
#print b

#d = bitmatrix(bArrayVal=bitarray(2**20), shape=(2,524288))
#print d

dsg = dsgd_cpu.Cdsgd(vMat.copy(),3,3,2)
dsg.run()
print vMat
print ""
print dsg.v