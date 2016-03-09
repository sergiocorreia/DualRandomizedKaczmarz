# edgenumbers.pyx -- Provide index for looking up edge ID
#
# Copyright (C) <2016> <Kevin Deweese>
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

import numpy
cimport numpy
import scipy

def edgenumbers(L):
    tempL=scipy.sparse.tril(L,-1)
    row=tempL.row
    col=tempL.col
    n = L.shape[0]
    m=len(row)

    Index=scipy.sparse.coo_matrix((scipy.arange(0,m),(row,col)),shape=(n,n))

    Index=Index+Index.transpose()
    Index=Index.tocsr()

    return Index
