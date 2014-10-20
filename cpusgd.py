import numpy as np
import logging
import random



class CSgd:

    NZSL = 0
    L2 = 1
    NZL2 = 2
    KL = 3
    
    def __init__(self, vMat,K, early_stop_diff=0.001):
        self.vMat = vMat
        self.K = K
        self.M = self.vMat.shape[0]
        self.N = self.vMat.shape[1]
        self.wMat = np.random.rand(self.M, K)
        self.hMat = np.random.rand(K, self.N)
        self.early_stop_diff = early_stop_diff
    
    def lossNZSLDervW(self, vR, vC, k, beta=0):
        return -2 * (self.vMat[vR,vC] - np.dot(self.wMat[vR,:], self.hMat[:,vC])) * self.hMat[k,vC]

    def lossNZSLDervH(self, vR, vC, k, beta=0):
        return -2 * (self.vMat[vR,vC] - np.dot(self.wMat[vR,:], self.hMat[:,vC])) * self.wMat[vR,k]
    
    def lossL2DerW(self, vR, vC, k, beta=0.002):
        return self.lossNZSLDervW(vR, vC, k) + 2 * beta * self.wMat[vR,k]/self.N
        
    def lossL2DerH(self, vR, vC, k, beta=0.002):
        return self.lossNZSLDervW(vR, vC, k) + 2 * beta * self.hMat[k,vC]/self.M
        
    def lossNZL2DerW(self, vR, vC, k, beta=0.002):
        return self.lossNZSLDervW(vR, vC, k) + 2 * beta * self.wMat[vR,k]
        
    def lossNZL2DerH(self, vR, vC, k, beta=0.002):
        return self.lossNZSLDervW(vR, vC, k) + 2 * beta * self.hMat[k,vC]
    
    def lossKLDerW(self, vR, vC, k, beta=0):
        return - self.hMat[k,vC] * (self.vMat[vR,vC] / np.dot(self.wMat[vR,:], self.hMat[:,vC]))
    
    def lossKLDerH(self, vR, vC, k, beta=0):
        return - self.wMat[vR,k] * (self.vMat[vR,vC] / np.dot(self.wMat[vR,:], self.hMat[:,vC]))
    
    
    def setLossFuns(self, loss_fun_type):
        if loss_fun_type == CSgd.NZSL:
            self.lossDervW = self.lossNZSLDervW
            self.lossDervH = self.lossNZSLDervH
        elif loss_fun_type == CSgd.L2:
            self.lossDervW = self.lossL2DerW
            self.lossDervH = self.lossL2DerH
        elif loss_fun_type == CSgd.NZL2:
            self.lossDervW = self.lossNZL2DerW
            self.lossDervH = self.lossNZL2DerH
        elif loss_fun_type == CSgd.KL:
            self.lossDervW = self.lossKLDerW
            self.lossDervH = self.lossKLDerH
    
    def trainOne(self,vR, vC, eta, beta=0):
        v = self.vMat[vR,vC]
        if v > 0:
            for k in xrange(self.K):
                wRowHat = self.wMat[vR,k] - eta * self.M * self.hMat[k,vC] * self.lossDervW(vR,vC,k,beta)
                self.hMat[k,vC] = self.hMat[k,vC] - eta * self.M * self.wMat[vR,k] * self.lossDervH(vR, vC, k,beta)
                self.wMat[vR,k] = wRowHat
    
    def trainAll(self, eta, steps, beta=0):
        for s in xrange(steps):
            rShuf = range(self.M)
            random.shuffle(rShuf)
            cShuf = range(self.N)
            random.shuffle(cShuf)
            for r in rShuf:
                for c in cShuf:
                    self.trainOne(r,c,eta,beta)
                    if s % 100 == 0:
                        e = self.err()
                        logging.debug("Error: %f", e)
                        if e < self.early_stop_diff:
                            logging.debug("Early stopping for error %f at %d. step", e, s)
                            return
                    
    def result(self):
        return np.dot(self.wMat, self.hMat)
    
    def err(self):
        tErr = 0
        for r in xrange(self.M):
            for c in xrange(self.N):
                v = self.vMat[r,c]
                if v >0:
                    tErr = tErr + (v - np.dot(self.wMat[r,:], self.hMat[:,c])) **2
        return tErr