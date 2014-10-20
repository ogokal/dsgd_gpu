import stratum
from cpusgd import CSgd
from multiprocessing.dummy import Pool



class Cdsgd:
    
    def __init__(self, v, h, w, K, sgd_loss_fun_type=CSgd.NZSL, \
                eta=0.01, block_train_steps=1000):
        self.K = K
        self.v = v
        self.ext = stratum.stratumExtractor(v,h,w)
        self.loss_fun_type = sgd_loss_fun_type
        self.eta = eta
        self.block_train_steps = block_train_steps
        
    def sgd(self, substratum):
        slices = self.ext.calculateVSlice(substratum)
        vMat = self.v[slices[0], slices[1]]
        csgd = CSgd(vMat,self.K)
        csgd.setLossFuns(self.loss_fun_type)
        csgd.trainAll(self.eta,self.block_train_steps)
        self.v[slices[0], slices[1]] = csgd.result()
    
    def p(self,stratum):
        mat = self.ext.convertBlockToData(stratum)
        print mat
    
    def stratumMapper(self,strat):
        stratumPool = Pool()
        stratumPool.map(self.sgd, strat.substratum)
        stratumPool.close()
        stratumPool.join()
        
    def run(self):
        for stratums in self.ext.nextStratum():
            map(self.stratumMapper,stratums)
                
    