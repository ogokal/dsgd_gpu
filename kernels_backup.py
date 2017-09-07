# -*- coding: utf-8 -*-
"""
Created on Tue Aug 05 19:13:40 2014

@author: ozturk
"""
import os
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


kernels = """

#define K 3

typedef struct stratumMeta{
    int x;
    int y;
    float data[2];
}stratumMeta;



typedef struct substratumDataAlloc{
    int x;
    int y;
    int rStart;
    int cStart;
} substratumDataAlloc;

__kernel
void
trainAllocSubstratum(__global float *v, 
                     const int vWid ,
                    __global substratumDataAlloc *substratums,
                    __global float *test
                     ){
                     
    size_t k = get_group_id(0);
    if(k == 1){
        substratumDataAlloc targetStra = substratums[k];
        size_t x = get_local_id(0);
        size_t y = get_local_id(1);
        test[x + y*2] = targetStra.cStart;
        //v[(targetStra.cStart + x) + (targetStra.rStart+y)*vWid];
    }
}



__kernel void test(__global stratumMeta *r, __global stratumMeta *w){
    w->data[0] = r->x*10;
    w->data[1] = r->y*20;
    w->x = 32;
    w->y = 33;
}

__kernel void doublify(__global float *a){
    a[get_global_id(0)] *= 2; 
}
 
float 
dotProd(__local float *row, __local float *col){
    float total = 0.0f;    
    for(int l=0;l<K;l++){
        total += row[l] * col[l];
    }
    return total;
}

float
lossNZSLDervW(__const float vCell, __local float *wRow, 
              __local float *hCol, __const int k){
    return -2 * (vCell - dotProd(wRow, hCol)) * hCol[k];
}

float
lossNZSLDervH(__const float vCell, __local float *wRow, 
              __local float *hCol, __const int k){
    return -2 * (vCell - dotProd(wRow, hCol)) * wRow[k];
}

__kernel 
void 
trainOne(__global float *w, __const float eta,
        __const int vHei, __global float *h,
        __global float *v, __const int hWid)                                   
{
    size_t x = get_local_id(0);
    size_t y = get_local_id(1);    
    size_t k = get_local_id(2);
    
    __local float wLocRow[K];
    __local float hLocCol[K];
    
    for(int c=0; c<K; c++){
          wLocRow[c] = w[y*K+c];
          hLocCol[c] = h[c*hWid+x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);   
    
    float vCell = v[y*hWid+x];    
    float wCellHat = wLocRow[k] - eta * vHei * hLocCol[k]
        * lossNZSLDervW(vCell,wLocRow,hLocCol,k);
    hLocCol[k] = hLocCol[k] - eta * vHei * wLocRow[k]
        * lossNZSLDervH(vCell,wLocRow,hLocCol,k);
    wLocRow[k] = wCellHat;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int c=0; c<K; c++){
          w[y*K+c] = wLocRow[c];
          h[c*hWid+x] = hLocCol[c];
    }
}

"""

