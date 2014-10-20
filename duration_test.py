import timeit
t = timeit.Timer('''
import cpusgd
import numpy as np
import logging
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

dsg = dsgd_cpu.Cdsgd(vMat.copy(),3,3,2)
dsg.run()
''')

print t.timeit(10)

#print vMat
#print ""
#print dsg.v

