# treeinfo.pyx -- Returns tree info
#                                                           
# Copyright (C) <2016> <Kevin Deweese>
# All rights reserved.                                                                    
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

import numpy
cimport numpy
import scipy

cdef extern from "../C/treeinfo.c":
    void treeinfo_c(int *index_indptr,
                    int *index_indices,
                    long long *index_data,
                    int *parent,
                    int *depth,
                    int *gedge,
                    int *vperm,
                    int m)

cpdef treeinfo(T, int minvtx, index):
    cdef int n=T.shape[0]
    cdef int m=T.nnz/2

    vpermtemp=scipy.sparse.csgraph.breadth_first_order(T,minvtx,directed=False)

    cdef numpy.ndarray[int, ndim=1] parent=vpermtemp[1]
    
    cdef numpy.ndarray[int, ndim=1] vperm=vpermtemp[0]
    
    cdef numpy.ndarray[int,ndim=1] depth=-1*numpy.ones(n,dtype='i4')
    cdef numpy.ndarray[int,ndim=1] gedge=-1*numpy.ones(n,dtype='i4')

    cdef numpy.ndarray[int,ndim=1] index_indptr=index.indptr
    cdef numpy.ndarray[int,ndim=1] index_indices=index.indices
    cdef numpy.ndarray[long long,ndim=1] index_data=index.data
    
    depth[minvtx]=0
    
    treeinfo_c(&index_indptr[0],&index_indices[0],&index_data[0],&parent[0],&depth[0],&gedge[0],&vperm[0],m)
    
    return {'parent':parent,'depth':depth,'gedge':gedge}


