#!/usr/bin/env python
# kktmat.py -- KKT matrix from Laplacian matrix
#
# Copyright (C) <2016> <Kevin Deweese>
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

import scipy

def kktmat(L):
    mat=scipy.sparse.coo_matrix(scipy.sparse.tril(L,-1))
    row=mat.row
    m=len(row)
    n=L.shape[0]
    col=mat.col
    val=mat.data
    
    #R=scipy.sparse.diags(-1/val,0)
    R=scipy.array(-1/val)
    i=scipy.concatenate([scipy.arange(0,m),scipy.arange(0,m)])
    j=scipy.concatenate([row,col])
    data=scipy.concatenate([scipy.ones(m),-scipy.ones(m)])
    B=scipy.sparse.coo_matrix((data,(i,j)))
    return {'R':R,'B':B} 
