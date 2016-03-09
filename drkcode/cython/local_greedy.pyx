# local_greedy.pyx -- Greedy search for extra cycles
#
# Copyright (C) <2016> <Kevin Deweese>
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

import numpy
cimport numpy
import scipy

cpdef find_cycle(numpy.ndarray[numpy.float64_t, ndim=1] Ldata,
                  numpy.ndarray[int, ndim=1] Lindices,
                  numpy.ndarray[int, ndim=1] Lindptr,
                  B,
                  numpy.ndarray[long, ndim=1] used,
                  int vtx1, int vtx2, int n, int m, index, int searchsize):
    
    cdef int cyclefound=0
                                
    cdef numpy.ndarray[long, ndim=1] queue = numpy.zeros(searchsize,dtype=int)
    cdef numpy.ndarray[long, ndim=1] trace = -1*numpy.ones(n,dtype=int)
    cdef numpy.ndarray[long, ndim=1] depth= -1*numpy.ones(n,dtype=int)
    cdef int head = 0
    #queue[0] = vtx1
    cdef int tail = 0
    cdef int size = 0
    cdef int vtx,i
    cdef int idx
    depth[vtx1]=0
    for i in xrange(Lindptr[vtx1],Lindptr[vtx1+1]):
        if(Lindices[i]==vtx1 or Lindices[i]==vtx2):
                continue
        else:
                queue[tail]=Lindices[i]
                trace[Lindices[i]]=vtx1
                depth[Lindices[i]]=1
                tail = numpy.mod(tail+1,searchsize)
                size = size+1

    while(size!=0 and size < searchsize and cyclefound==False):
        vtx = queue[head]
        head = numpy.mod(head+1,searchsize)
        size = size-1
        
        for i in xrange(Lindptr[vtx],Lindptr[vtx+1]):
            if(Lindices[i]==vtx1 or trace[Lindices[i]]!=-1):
                continue
            if(Lindices[i]==vtx2):
                cyclefound=True
                trace[vtx2]=vtx
                depth[vtx2]=depth[vtx]+1
                break
            
            queue[tail]=Lindices[i]
            tail = numpy.mod(tail+1,searchsize)
            size = size+1
            trace[Lindices[i]]=vtx
            depth[Lindices[i]]=depth[vtx]+1

    if(cyclefound==0):
        return cyclefound,[],[],[]

    
    cdef numpy.ndarray[long, ndim=1] colptr = numpy.zeros(1,dtype=int)
    cdef numpy.ndarray[long, ndim=1] inds = numpy.zeros(depth[vtx2]+1,dtype=int)

    cdef numpy.ndarray[long, ndim=1] data = numpy.zeros(depth[vtx2]+1,dtype=int)
    
    cdef int edgesptr = 0

    cycle=[]
    
    vtx=vtx2
    while(vtx!=vtx1):
        cycle.append(vtx)

        
        idx=index[vtx,trace[vtx]]

        Bptr=B.indptr[idx]
        if(B.indices[Bptr]==vtx and B.indices[Bptr+1]==trace[vtx]):
            inds[edgesptr]=idx
            data[edgesptr]=1
            edgesptr=edgesptr+1
            used[idx]=1
            
        if(B.indices[Bptr]==trace[vtx] and B.indices[Bptr+1]==vtx):
            inds[edgesptr]=idx
            data[edgesptr]=-1
            edgesptr=edgesptr+1
            used[idx]=1
            
            

        vtx=trace[vtx]
    
    idx=index[vtx1,vtx2]
    
    Bptr=B.indptr[idx]
    if(B.indices[Bptr]==vtx1 and B.indices[Bptr+1]==vtx2):
        inds[edgesptr]=idx
        data[edgesptr]=1
        edgesptr=edgesptr+1
        used[idx]=1

    if(B.indices[Bptr]==vtx2 and B.indices[Bptr+1]==vtx1):
        inds[edgesptr]=idx
        data[edgesptr]=-1
        edgesptr=edgesptr+1
        used[idx]=1


    #if(nontreeused==1):
    #    return 0,[],[],[]
    #    #print vtx
    return cyclefound,inds,data,edgesptr,cycle





cpdef find_basis(L,B,index,n,m, int searchsize):
        cdef numpy.ndarray[long] used=numpy.zeros(m,dtype=int)
        greedyptr=[0]
        greedyinds=[]
        greedydata=[]
        cdef cols=0
        cdef int i,j
        for i in xrange(0,m):
            if(used[i]==0):
                ptr=B.indptr[i]
                cycleinfo=find_cycle(L.data,L.indices,L.indptr,
                                      B,used,B.indices[ptr],
                                      B.indices[ptr+1],n,m,index,searchsize)

                if(cycleinfo[0]==1):
                    for j in xrange(0,cycleinfo[3]):
                        greedyinds.append(cycleinfo[1][j])
                        greedydata.append(cycleinfo[2][j])
                    greedyptr.append(cycleinfo[3]+greedyptr[cols])
                    cols=cols+1

        greedyptr=scipy.array(greedyptr)
        greedyinds=scipy.array(greedyinds)
        greedydata=scipy.array(greedydata)


        newF=scipy.sparse.csc_matrix((greedydata,greedyinds,greedyptr),
                                     shape=(m,cols),dtype=numpy.int8)

        return newF
