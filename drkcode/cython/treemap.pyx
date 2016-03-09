# treemap.pyx -- Returns array containing labels of edges in tree
#
# Copyright (C) <2016> <Kevin Deweese>
# All rights reserved.
# 
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

import numpy
cimport numpy
import scipy

def treemap(L,
    numpy.ndarray[numpy.float64_t, ndim=1] data,
    numpy.ndarray[int, ndim=1] indices,
    numpy.ndarray[int, ndim=1] indptr):

    n=L.shape[0]
    mat=scipy.sparse.coo_matrix(scipy.sparse.tril(L,-1))
    row=mat.row
    col=mat.col
    m=len(row)
    
    treemap=scipy.zeros(n-1,dtype=int)
    
    cdef i,j
    cdef int temp=0
    for i in range (0,m):
        for j in xrange(indptr[row[i]],indptr[row[i]+1]):
            if(indices[j]==col[i]):
                treemap[temp]=i
                temp=temp+1
    
    return treemap
    
