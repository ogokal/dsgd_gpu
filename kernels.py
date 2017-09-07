# -*- coding: utf-8 -*-
"""
Created on Tue Aug 05 19:13:40 2014

@author: ozturk
"""
import os
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

import stratum
from mako.template import Template

kernels = Template("""
<% wLen = BLOCK_HEI * COMMON_DIM%>
<% hLen = BLOCK_WID * COMMON_DIM%>

typedef struct substratum{
    int x;
    int y;
    int rStart;
    int cStart;
    float w[${wLen}];
    float h[${hLen}];
} substratum;

__kernel
void
testStratum(__global substratum *sbstras,
             __global substratum *outsbstra
             ){
/*    
    int sbStraIdx = get_global_id(0);
    if(sbStraIdx == 0){
         outsbstra.x = sbstras[sbStraIdx].x;
         outsbstra.y = sbstras[sbStraIdx].y;
         outsbstra.rStart = sbstras[sbStraIdx].rStart;
         outsbstra.cStart = sbstras[sbStraIdx].cStart;
         for(int wc=0;wc<${wLen};wc++){
            outsbstra[sbStraIdx].w[wc] = sbstras[sbStraIdx].w[wc];
         }
         for(int hc=0;hc<${wLen};hc++){
            outsbstra[sbStraIdx].h[hc] = sbstras[sbStraIdx].h[hc];
         }
    }
*/
}
""")
kernels_scr = kernels.render(BLOCK_HEI = stratum.BLOCK_HEI, \
                            COMMON_DIM = stratum.COMMON_DIM, \
                            BLOCK_WID  = stratum.BLOCK_WID)

if __name__ == '__main__':                           
    
    import numpy as np
    
    vMat = np.array([
        [5,3,0,1],
        [4,0,0,1],
        [1,1,0,5],
        [1,0,0,4],
        [0,1,5,4]
    ],dtype=np.float32)

    vMat = np.concatenate((vMat,np.ones((5,2), dtype=np.int32)),axis=1)
    
    import allalloc_cl_sgd

    gpuDsgd = allalloc_cl_sgd.GpuDSgd(vMat)
    gpuDsgd.run()

 
