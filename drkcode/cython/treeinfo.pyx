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

def treeinfo(T, minvtx, index):
    n=T.shape[0]
    m=T.nnz/2

    vperm=scipy.sparse.csgraph.breadth_first_order(T,minvtx,directed=False)

    parent=vperm[1]
    
    vperm=vperm[0]
    
    cdef numpy.ndarray[int,ndim=1] depth=-1*numpy.ones(n,dtype='i4')
    cdef numpy.ndarray[int,ndim=1] gedge=-1*numpy.ones(n,dtype='i4')
    
    depth[minvtx]=0

    cdef int i
    for i in xrange(1,m+1):
        v=vperm[i]
        depth[v]=depth[parent[v]]+1
        gedge[v]=index[parent[v],v]
    
    return {'parent':parent,'depth':depth,'gedge':gedge}


