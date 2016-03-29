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
from libc.stdlib cimport malloc, free

cdef extern from "../C/fundcycles.c":
    int fundcycles_c(int *FI,
                     int *FJ,
                     numpy.int8_t *FV,
                     numpy.float64_t *Bdata,
                     int *Bindices,
                     int *Bindptr,
                     int *parent,
                     int *depth,
                     int *gedge,
                     int *nontree,
                     int maxF,
                     int mF,
                     int nontreecount,
                     int n,
                     int *finished,
                     int *i)

cpdef fundcycles(numpy.ndarray[numpy.float64_t, ndim=1] Bdata,
                 numpy.ndarray[int, ndim=1] Bindptr,
                 numpy.ndarray[int, ndim=1] Bindices,
                 numpy.ndarray[int, ndim=1] tree_map,
                 numpy.ndarray[int, ndim=1] parent,
                 numpy.ndarray[int, ndim=1] depth,
                 numpy.ndarray[int, ndim=1] gedge,
                 int m):

    cdef int n=len(parent)

    cdef int i

    cdef numpy.ndarray[int, ndim=1] temp = numpy.zeros(m,dtype=numpy.int32)
    for i in xrange(0,len(tree_map)):
        temp[tree_map[i]]=-1

    cdef numpy.ndarray[int, ndim=1] nontree = numpy.zeros(m-len(tree_map),dtype=numpy.int32)
    cdef int idx=0
    for i in xrange(0,m):
        if(temp[i]==0):
            nontree[idx]=i
            idx+=1

    cdef int maxF=3*m
    cdef int mF=-1
    cdef int nontreecount = len(nontree)
    
    
    
    cdef int v,w
    

    cdef numpy.ndarray[int, ndim=1] FI=scipy.zeros(maxF,dtype=numpy.int32)
    cdef numpy.ndarray[int, ndim=1] FJ=scipy.zeros(maxF,dtype=numpy.int32)
    cdef numpy.ndarray[numpy.int8_t, ndim=1] FV=scipy.zeros(maxF,dtype=numpy.int8)
    
    cdef int finished=0
    cdef int reused=0
    #TODO This is silly, not sure how to resize these numpy arrays inside of C 
    # so this exists the C code to resize

    while(finished==0):
        mF=fundcycles_c(&FI[0],&FJ[0],&FV[0],&Bdata[0],&Bindices[0],&Bindptr[0],&parent[0],&depth[0],&gedge[0],&nontree[0],maxF,mF,nontreecount,n,&finished,&reused)

        if maxF-mF < n:
            maxF=round(maxF*1.5)
            FI=numpy.resize(FI,maxF)
            FJ=numpy.resize(FJ,maxF)
            FV=numpy.resize(FV,maxF)

            
    FI=numpy.resize(FI,mF+1)
    FJ=numpy.resize(FJ,mF+1)
    FV=numpy.resize(FV,mF+1)
    
    F=scipy.sparse.coo_matrix((FV,(FI,FJ)),shape=(m,m))
    
    return F
