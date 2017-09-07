import struct
import cltools

class clstratum:
    def __init__(self, x,y,mat):
        self.x = x
        self.y = y
    
    def tostruct(self):
        return struct.pack('ii',self.x,self.x)
        

tools = cltools.ClTools()
intel = tools[cltools.ClTools.INTEL]
rbuf = intel.readonlyBuffer(clstratum(2,3,None).tostruct())
wbuf = intel.writeonlyBuffer(clstratum(2,3,None).tostruct())
intel.call('getstruct',(1,1),None,rbuf,wbuf)

print wbuf.get()
