# fundcycles.pyx -- Constructs fundamental cycle basis
#
# Copyright (C) <2016> <Kevin Deweese>
# All rights reserved.
#
# This software may be modified and distributed under the terms 
# of the BSD license.  See the LICENSE file for details.

import scipy
import math
import numpy
cimport numpy

cpdef fundcycles(numpy.ndarray[numpy.float64_t, ndim=1] Bdata,
                 numpy.ndarray[int, ndim=1] Bindptr,
                 numpy.ndarray[int, ndim=1] Bindices,
                 nontree,
                 numpy.ndarray[int, ndim=1] parent,
                 numpy.ndarray[int, ndim=1] depth,
                 numpy.ndarray[int, ndim=1] gedge,
                 int m):

    cdef int n=len(parent)
    #cdef int m=len(treemap)
    
    cdef int maxF=3*m
    cdef int mF=-1
    list=(range(m))
    #print list
    
    cdef int i
    cdef numpy.ndarray FI=scipy.zeros(maxF,dtype=int)
    cdef numpy.ndarray FJ=scipy.zeros(maxF,dtype=int)
    cdef numpy.ndarray FV=scipy.zeros(maxF,dtype=numpy.int8)
    cdef int v,w
    
    for i in range(0,len(nontree)):
        if maxF-mF < n:
            maxF=round(maxF*1.5)
            FI.resize(maxF,refcheck=False)
            FJ.resize(maxF,refcheck=False)
            FV.resize(maxF,refcheck=False)

        ptr=Bindptr[nontree[i]]
        if(Bdata[ptr]>0):
            v=Bindices[ptr]
            w=Bindices[ptr+1]
        else:
            v=Bindices[ptr+1]
            w=Bindices[ptr]
        
        if(depth[v]==-1 or depth[w]==-1):
            continue
        
        mF=mF+1
        FI[mF]=nontree[i]
        FJ[mF]=nontree[i]
        FV[mF]=-1
            
        while(v!=w):
            if(depth[v] >= depth[w]):
                if(depth[v]==0):
                    print "error tree is not spanning"
                mF=mF+1
                FI[mF]=gedge[v]
                FJ[mF]=nontree[i]
                vv=parent[v]
                FV[mF]=int(math.copysign(1,v-vv))
                v=vv
            
            if(depth[w] > depth[v]):
                mF=mF+1
                FI[mF]=gedge[w]
                FJ[mF]=nontree[i]
                ww=parent[w]
                FV[mF]=int(math.copysign(1,ww-w))
                w=ww

                
    FI.resize(mF+1,refcheck=False)
    FJ.resize(mF+1,refcheck=False)
    FV.resize(mF+1,refcheck=False)
            

    F=scipy.sparse.coo_matrix((FV,(FI,FJ)),shape=(m,m))
    
    return F
