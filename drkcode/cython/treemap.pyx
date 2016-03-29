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

cdef extern from "../C/treemap.c":
    void treemap_c(int *Tindptr,
                   int *Tindices,
                   int *row,
                   int *col,
                   int *treemap,
                   int m)

cpdef treemap(L,
              numpy.ndarray[numpy.float64_t, ndim=1] Tdata,
              numpy.ndarray[int, ndim=1] Tindices,
              numpy.ndarray[int, ndim=1] Tindptr):

    cdef int n=L.shape[0]
    mat=scipy.sparse.coo_matrix(scipy.sparse.tril(L,-1))
    cdef numpy.ndarray[int, ndim=1] row = mat.row
    cdef numpy.ndarray[int, ndim=1] col = mat.col
    cdef int m=len(row)
    
    cdef numpy.ndarray[int, ndim=1] treemap = numpy.zeros(n-1,dtype=numpy.int32)
    
    
    treemap_c(&Tindptr[0],&Tindices[0],&row[0],&col[0],&treemap[0],m)
    
    return treemap
    
