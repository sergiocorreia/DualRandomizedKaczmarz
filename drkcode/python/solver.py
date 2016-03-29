# solver.py -- Example solver script
#
# Copyright (C) <2016> <Kevin Deweese>
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

import pyximport
#pyximport.install(reload_support=True)
import scipy
import scipy.io
import numpy
import kktmat
from drkcode.cython import treemap,cython_utils,fundcycles,local_greedy,edgenumbers,treeinfo
import random
import networkx
import os
from scipy.sparse import csgraph
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import spsolve


def solve(A, maxiters=-1, tolerance=-1,randomTree=False,userTree=None,greedy=0,
          userX=None,procs=1,useres=1,tracker=0,fixediters=-1):
    print "Running Setup"
    
    n=A.shape[0]

    L=scipy.sparse.csgraph.laplacian(A)
    
    
    L=L.tocsr()
    n=L.shape[0]
    index=edgenumbers.edgenumbers(L)
    RB=kktmat.kktmat(L)
    R=RB['R']
    B=RB['B']
    m=R.shape[0]
    B=B.tocsr()

    
    if(userX!=None):
        x=userX
    else:
        x=scipy.random.rand(n)
    x=x-scipy.mean(x)
    b=L*x
    
    Adata=A.data
    
    if(userTree!=None):
        T=userTree

    else:
        if(randomTree):
            for i in range(0,len(Adata)):
                Adata[i]=Adata[i]*(1+scipy.rand())
        
        T=-minimum_spanning_tree(-A)
        #print T
    Tdata=T.data
    for i in range(0,len(Tdata)):
        Tdata[i]=1

    T=T+T.transpose()

    T=T.tocsc()
    tree_map=treemap.treemap(L,T.data,T.indices,T.indptr)
    minidx=scipy.argwhere(T.sum(0)==1)[0][0][1]
    info=treeinfo.treeinfo(T,minidx,index)
    parent = info['parent']
    depth = info['depth']
    gedge = info['gedge']
    
    flow=scipy.zeros(m)
    cython_utils.initialize_flow(B.data,B.indptr,B.indices,b,parent,depth,gedge,m,flow)

    F=fundcycles.fundcycles(B.data,B.indptr,B.indices,tree_map,parent,depth,gedge,m)
    F=F.tocsc()
    
    if(greedy!=0):
        newF=local_greedy.find_basis(L,B,index,n,m,greedy)
    

    if(greedy!=0):
        F=scipy.sparse.hstack([F,newF])
    

    tau = cython_utils.find_tau(F.data,F.indptr,F.indices,R)

    print "Starting Solve"
    
    #itersguess=scipy.ceil(tau*scipy.log2((tau*(tau-m+2*n-2))/(tolerance*tolerance)))
        
    if(maxiters < 0 and tolerance <0):
        maxiters=100000
        
    results=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    
    
    probs = cython_utils.get_probs(F.data,F.indptr,F.indices,R,1)
        
    temp_results=cython_utils.solve_wrapper(F.data,F.indices,F.indptr,
                                            B.data,B.indices,B.indptr,
                                            R,1.0,
                                            L.data,L.indices,L.indptr,
                                            b,probs,flow,
                                            parent,depth,gedge,tolerance,
                                            procs,x,useres,fixediters,tracker)

    
            
    
    for i in range(0,10):
        results[i]=temp_results[i]
    results[10]=temp_results[10]
    results[11]=results[10]-scipy.mean(results[10])
    results[12]=tau
    results[13]=1-(1/tau)
    
    return results
    

    

