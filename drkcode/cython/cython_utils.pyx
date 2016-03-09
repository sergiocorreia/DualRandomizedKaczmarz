#cython: profile=False
# cython_utils.pyx -- Utility file containing most of the DRK functions
# 
# Copyright (C) <2016> <Kevin Deweese>
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

import numpy
cimport numpy
import scipy
import random
from libc.stdlib cimport rand,RAND_MAX,srand

# updates flow around cycle
cdef update(numpy.ndarray[numpy.int8_t, ndim=1] Fdata,
            numpy.ndarray[int, ndim=1] Findices,
            numpy.ndarray[int, ndim=1] Findptr,
            numpy.ndarray[numpy.float64_t, ndim=1] R,
            numpy.float64_t resist_mod,
            numpy.ndarray[numpy.float64_t, ndim=1] probs, 
            numpy.ndarray[numpy.float64_t, ndim=1] flow,
            int m, int num_cycles, int cycle,
            numpy.ndarray[long, ndim=1] edges):


    cdef int fund_edges_updated = 0
    cdef int other_edges = 0
    edges[0]=0
    edges[1]=0
    cdef int i

    cdef int idx = Findptr[cycle]
    cdef numpy.float64_t inner = 0
    cdef numpy.float64_t dotproduct=0

    cdef int j
    if(cycle < m):
        for j in xrange(Findptr[cycle],Findptr[cycle+1]):
            if(Findices[j]==cycle):
                dotproduct += flow[cycle]*(R[cycle]/resist_mod)*Fdata[j]
                inner += (R[cycle]/resist_mod)
            else:
                dotproduct += flow[Findices[j]]*R[Findices[j]]*Fdata[j]
                inner += R[Findices[j]]
            fund_edges_updated += 1
    
    else:
        
        for j in xrange(Findptr[cycle],Findptr[cycle+1]):
            dotproduct += flow[Findices[j]]*(R[Findices[j]]/resist_mod)*Fdata[j]
            inner += (R[Findices[j]]/resist_mod)
            other_edges += 1

    cdef numpy.float64_t factor=dotproduct/inner
    
    for j in xrange(Findptr[cycle],Findptr[cycle+1]):
        flow[Findices[j]] = flow[Findices[j]] - factor*Fdata[j]

    edges[0]=fund_edges_updated
    edges[1]=other_edges
    

cdef get_random():
    cdef numpy.float64_t random_var = rand()
    return random_var/RAND_MAX

# calculated relative redisual
cdef relative_residual(numpy.ndarray[numpy.float64_t, ndim=1] v, 
            numpy.ndarray[numpy.float64_t, ndim=1] b, 
            numpy.ndarray[numpy.float64_t, ndim=1] Ldata,
            numpy.ndarray[int, ndim=1] Lindices,
            numpy.ndarray[int, ndim=1] Lindptr):

    cdef double relres = numpy.linalg.norm(matvec(Ldata,Lindices,Lindptr,v)-b)/numpy.linalg.norm(b)
    
    return relres

# incudes potentials to calculate primal lhs
cdef induce_voltage(numpy.ndarray[numpy.float64_t, ndim=1] flow,
                     numpy.ndarray[numpy.float64_t, ndim=1] b,
                     numpy.ndarray[numpy.float64_t, ndim=1] Bdata,
                     numpy.ndarray[int, ndim=1] Bindices,
                     numpy.ndarray[int, ndim=1] Bindptr,
                     numpy.ndarray[numpy.float64_t, ndim=1] R,
                     numpy.ndarray[int, ndim=1] parents, 
                     numpy.ndarray[int, ndim=1] gedge,
                     numpy.ndarray[long, ndim=1] sorteddepth,
                     numpy.ndarray[numpy.float64_t, ndim=1] v):

    cdef int i
    cdef int vtx
    cdef int n = len(parents)
    cdef int ptr

    for i in xrange(0,n):
        v[i]=0

    for i in xrange(1,n):
        vtx = sorteddepth[i]        
        ptr=Bindptr[gedge[vtx]]
        if(Bindices[ptr]!=vtx):
            ptr=ptr+1
        v[vtx]=flow[gedge[vtx]]*R[gedge[vtx]]/numpy.sign(Bdata[ptr])+v[parents[vtx]]
        
# matrix vector product
cdef matvec(numpy.ndarray[numpy.float64_t, ndim=1] Ldata,
            numpy.ndarray[int, ndim=1] Lindices,
            numpy.ndarray[int, ndim=1] Lindptr,
            numpy.ndarray[numpy.float64_t, ndim=1] v):

    cdef numpy.ndarray[numpy.float64_t, ndim=1] b = numpy.zeros(Lindptr.shape[0]-1)    

    cdef int i,j
    for i in xrange(0,Lindptr.shape[0]-1):
        for j in xrange(Lindptr[i], Lindptr[i+1]):
            b[i]=b[i]+Ldata[j]*v[Lindices[j]]
          
    return b


# Python wrapper to call C solve funtions
def solve_wrapper(numpy.ndarray[numpy.int8_t, ndim=1] Fdata,
                  numpy.ndarray[int, ndim=1] Findices,
                  numpy.ndarray[int, ndim=1] Findptr,
                  numpy.ndarray[numpy.float64_t, ndim=1] Bdata,
                  numpy.ndarray[int, ndim=1] Bindices,
                  numpy.ndarray[int, ndim=1] Bindptr,
                  numpy.ndarray[numpy.float64_t, ndim=1] R, numpy.float64_t resist_mod,
                  numpy.ndarray[numpy.float64_t, ndim=1] Ldata,
                  numpy.ndarray[int, ndim=1] Lindices,
                  numpy.ndarray[int, ndim=1] Lindptr,
                  numpy.ndarray[numpy.float64_t, ndim=1] b,
                  numpy.ndarray[numpy.float64_t, ndim=1] probs,
                  numpy.ndarray[numpy.float64_t, ndim=1] flow,
                  numpy.ndarray[int, ndim=1] parents,
                  numpy.ndarray[int, ndim=1] depth,
                  numpy.ndarray[int, ndim=1] gedge,
                  double tolerance, int processors,
                  numpy.ndarray[numpy.float64_t, ndim=1] known_solution,
                  int useres):
    cdef int n=len(b)
    cdef numpy.ndarray[numpy.float64_t] stats = numpy.zeros(10)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] v = numpy.zeros(n)
    solve_fixedtol(Fdata,Findices,Findptr,Bdata,Bindices,Bindptr,
                   R,resist_mod,Ldata,Lindices,Lindptr,b,v,probs,
                   flow,parents,depth,gedge,tolerance,processors,stats,known_solution,useres)

    return(stats[0],stats[1],stats[2],stats[3],stats[4],stats[5],stats[6],
           stats[7],stats[8],stats[9],v)

# solves dual system to a fixed tolerance
def solve_fixedtol(numpy.ndarray[numpy.int8_t, ndim=1] Fdata,
                    numpy.ndarray[int, ndim=1] Findices,
                    numpy.ndarray[int, ndim=1] Findptr,
                    numpy.ndarray[numpy.float64_t, ndim=1] Bdata,
                    numpy.ndarray[int, ndim=1] Bindices,
                    numpy.ndarray[int, ndim=1] Bindptr,
                    numpy.ndarray[numpy.float64_t, ndim=1] R, numpy.float64_t resist_mod,
                    numpy.ndarray[numpy.float64_t, ndim=1] Ldata,
                    numpy.ndarray[int, ndim=1] Lindices,
                    numpy.ndarray[int, ndim=1] Lindptr,
                    numpy.ndarray[numpy.float64_t, ndim=1] b,
                    numpy.ndarray[numpy.float64_t, ndim=1] v,
                    numpy.ndarray[numpy.float64_t, ndim=1] probs, 
                    numpy.ndarray[numpy.float64_t, ndim=1] flow,
                    numpy.ndarray[int, ndim=1] parents,
                    numpy.ndarray[int, ndim=1] depth,
                    numpy.ndarray[int, ndim=1] gedge,
                    double tolerance, int processors,
                    numpy.ndarray[numpy.float64_t, ndim=1] stats,
                    numpy.ndarray[numpy.float64_t, ndim=1] known_solution,
                    int useres):

    cdef int iteration = 0
    cdef int i,j
    cdef int m = len(flow)
    cdef int n = len(b)
    cdef double relres = 100000
    cdef edges_updated = 0
    cdef other_edges = 0
    cdef long logn_times_iters=0
    cdef long log_edges_updated=0
    cdef long projections = 0
    cdef numpy.ndarray[long, ndim=1] temp_edges =numpy.zeros(2,dtype=int)
    cdef numpy.ndarray[long, ndim=1] sorteddepth = numpy.argsort(depth)
    cdef numpy.ndarray[long, ndim=1] used = numpy.zeros(m,dtype=int)
    cdef numpy.float64_t randomvar=0
    cdef int cycle
    cdef int num_cycles = len(Findptr)-1
    cdef int clash = 0
    cdef long maxspan = 0
    cdef long totalspan=0
    cdef long maxlogspan = 0
    cdef long totallogspan=0
    cdef long totallogespan=0
    cdef long maxlogespan=0
    cdef numpy.float64_t err = 100000
    
    
    while(err > tolerance):

        maxspan=0
        maxlogspan=0
        maxlogespan=0
        used = numpy.zeros(m,dtype=int)
        
        cycle = binary_search(probs,get_random(),0,num_cycles-1)
        
        update(Fdata,Findices,Findptr,R,resist_mod,probs,flow,m,num_cycles,cycle,temp_edges)

        for i in xrange(Findptr[cycle],Findptr[cycle+1]):
            used[Findices[i]]=1
        #iteration+=1
        if(temp_edges[0]!=0):
            iteration+=1
            edges_updated+=temp_edges[0]
            log_edges_updated+=numpy.log2(temp_edges[0])
            maxspan=temp_edges[0]
            maxlogspan=numpy.log2(n)
            logn_times_iters+=numpy.log2(n)
            maxlogespan=numpy.log2(temp_edges[0])
            projections+=1
        if(temp_edges[1]!=0):
            iteration+=1
            other_edges+=temp_edges[1]
            log_edges_updated+=numpy.log2(temp_edges[1])
            maxspan=temp_edges[1]
            maxlogspan=temp_edges[1]
            maxlogespan=numpy.log2(temp_edges[1])
            projections+=1
        temp_edges[0]=0
        temp_edges[1]=0
        
        for i in xrange(1,processors):
            clash = 0
            cycle = binary_search(probs,get_random(),0,num_cycles-1)
            for j in xrange(Findptr[cycle],Findptr[cycle+1]):
                if(used[Findices[j]]==1):
                    clash = 1
                    break
                else:
                    used[Findices[j]]=1

            if clash:
                continue
            else:
                update(Fdata,Findices,Findptr,R,resist_mod,probs,flow,m,num_cycles,cycle,temp_edges)
                if(temp_edges[0]!=0):
                    edges_updated+=temp_edges[0]
                    log_edges_updated+=numpy.log2(temp_edges[0])
                    if(temp_edges[0] > maxspan):
                        maxspan=temp_edges[0]
                    if(numpy.log2(n) > maxlogspan):
                        maxlogspan = numpy.log2(n)
                    logn_times_iters+=numpy.log2(n)
                    projections+=1
                    if(numpy.log2(temp_edges[0]) > maxlogespan):
                        maxlogespan=numpy.log2(temp_edges[0])
                if(temp_edges[1]!=0):
                    other_edges+=temp_edges[1]
                    log_edges_updated+=numpy.log2(temp_edges[1])
                    if(temp_edges[1] > maxspan):
                        maxspan=temp_edges[1]
                    if(temp_edges[1] > maxlogspan):
                        maxlogspan=temp_edges[1]
                    projections+=1
                    if(numpy.log2(temp_edges[1]) > maxlogespan):
                        maxlogespan=numpy.log2(temp_edges[1])
                temp_edges[0]=0
                temp_edges[1]=0
                

        totalspan=totalspan+maxspan
        totallogspan=totallogspan+maxlogspan
        totallogespan=totallogespan+maxlogespan
        if(numpy.mod(iteration,n)==0):
            
            induce_voltage(flow,b,Bdata,Bindices,Bindptr,R,parents,gedge,sorteddepth,v)
            if(useres==1):
                err=relative_residual(v,b,Ldata,Lindices,Lindptr)
            else:
                err=numpy.linalg.norm(v-numpy.mean(v) - known_solution)/numpy.linalg.norm(known_solution)

    
    stats[0]=edges_updated
    stats[1]=logn_times_iters
    stats[2]=log_edges_updated
    stats[3]=projections

    stats[4]=totalspan
    stats[5]=totallogspan
    stats[6]=totallogespan
    stats[7]=iteration

    stats[8]=other_edges

    stats[9]=err
    
# Solve dual system to a fixed number of iterations
cdef solve_fixediters(numpy.ndarray[numpy.int8_t, ndim=1] Fdata,
                     numpy.ndarray[int, ndim=1] Findices,
                     numpy.ndarray[int, ndim=1] Findptr,
                     B,
                     numpy.ndarray[numpy.float64_t, ndim=1] R, 
                     numpy.float64_t resist_mod, 
                     numpy.ndarray[numpy.float64_t, ndim=1] Ldata,
                     numpy.ndarray[int, ndim=1] Lindices,
                     numpy.ndarray[int, ndim=1] Lindptr,b,v,
                     numpy.ndarray[numpy.float64_t, ndim=1] probs, 
                     numpy.ndarray[numpy.float64_t, ndim=1] flow,
                     numpy.ndarray[int, ndim=1] parents,
                     numpy.ndarray[int, ndim=1] depth,
                     numpy.ndarray[int, ndim=1] gedge,
                     int maxiters, int processors,
                     numpy.ndarray[numpy.float64_t, ndim=1] stats,
                     numpy.ndarray[numpy.float64_t, ndim=1] known_solution,
                     int useres):

    cdef int iterations=0
    cdef long edges_updated=0
    cdef long log_edges_updated=0
    cdef long logn_times_iters=0
    cdef int num_cycles = len(Findptr)-1

    cdef numpy.ndarray[long, ndim=1] sorteddepth = numpy.argsort(depth) 
    
    cdef numpy.ndarray[long, ndim=1] temp_edges =numpy.zeros(2,dtype=int) 
    cdef numpy.float64_t relres
    cdef int n=len(b)
    cdef int m=len(flow)
                
    cycle = binary_search(probs,get_random(),0,num_cycles-1)

    while(iteration < maxiters):
        maxspan=0
        maxlogspan=0
        maxlogespan=0
        used = numpy.zeros(m,dtype=int)
        
        cycle = binary_search(probs,get_random(),0,num_cycles-1)
        
        update(Fdata,Findices,Findptr,R,resist_mod,probs,flow,m,num_cycles,cycle,temp_edges)

        for i in xrange(Findptr[cycle],Findptr[cycle+1]):
            used[Findices[i]]=1

        if(temp_edges[0]!=0):
            iteration+=1
            edges_updated+=temp_edges[0]
            log_edges_updated+=numpy.log2(temp_edges[0])
            maxspan=temp_edges[0]
            maxlogspan=numpy.log2(n)
            logn_times_iters+=numpy.log2(n)
            maxlogespan=numpy.log2(temp_edges[0])
            projections+=1
        if(temp_edges[1]!=0):
            iteration+=1
            other_edges+=temp_edges[1]
            log_edges_updated+=numpy.log2(temp_edges[1])
            maxspan=temp_edges[1]
            maxlogspan=temp_edges[1]
            maxlogespan=numpy.log2(temp_edges[1])
            projections+=1
        temp_edges[0]=0
        temp_edges[1]=0
        
        for i in xrange(1,processors):
            clash = 0
            cycle = binary_search(probs,get_random(),0,num_cycles-1)
            for j in xrange(Findptr[cycle],Findptr[cycle+1]):
                if(used[Findices[j]]==1):
                    clash = 1
                    break
                else:
                    used[Findices[j]]=1

            if clash:
                continue
            else:
                update(Fdata,Findices,Findptr,R,resist_mod,probs,flow,m,num_cycles,cycle,temp_edges)
                if(temp_edges[0]!=0):
                    edges_updated+=temp_edges[0]
                    log_edges_updated+=numpy.log2(temp_edges[0])
                    if(temp_edges[0] > maxspan):
                        maxspan=temp_edges[0]
                    if(numpy.log2(n) > maxlogspan):
                        maxlogspan = numpy.log2(n)
                    logn_times_iters+=numpy.log2(n)
                    projections+=1
                    if(numpy.log2(temp_edges[0]) > maxlogespan):
                        maxlogespan=numpy.log2(temp_edges[0])
                if(temp_edges[1]!=0):
                    other_edges+=temp_edges[1]
                    log_edges_updated+=numpy.log2(temp_edges[1])
                    if(temp_edges[1] > maxspan):
                        maxspan=temp_edges[1]
                    if(temp_edges[1] > maxlogspan):
                        maxlogspan=temp_edges[1]
                    projections+=1
                    if(numpy.log2(temp_edges[1]) > maxlogespan):
                        maxlogespan=numpy.log2(temp_edges[1])
                temp_edges[0]=0
                temp_edges[1]=0
                

        totalspan=totalspan+maxspan
        totallogspan=totallogspan+maxlogspan
        totallogespan=totallogespan+maxlogespan
            
    
    induce_voltage(flow,b,B.data,B.indices,B.indptr,R,parents,gedge,sorteddepth,v)
    if(useres==1):
        err=relative_residual(v,b,Ldata,Lindices,Lindptr)
    else:
        err=numpy.linalg.norm(v-numpy.mean(v) - known_solution)/numpy.linalg.norm(known_solution)
    
    stats[0]=edges_updated
    stats[1]=logn_times_iters
    stats[2]=log_edges_updated
    stats[3]=projections

    stats[4]=totalspan
    stats[5]=totallogspan
    stats[6]=totallogespan
    stats[7]=iteration

    stats[8]=other_edges

    stats[9]=err


# search cycle distribution function for cycle
cdef binary_search(numpy.ndarray[numpy.float64_t, ndim=1] probs, 
    numpy.float64_t target,
    int start,
    int stop):    

    cdef int mid
    while(start < stop):
        mid = (stop + start)/2
    
        #search first half
        if(probs[mid] > target):
            stop = mid
        else:             
            start = mid + 1

    return start

# initialize the flow vector
def initialize_flow(numpy.ndarray[numpy.float64_t, ndim=1] Bdata,
                    numpy.ndarray[int, ndim=1] Bindptr,
                    numpy.ndarray[int, ndim=1] Bindices,
                    numpy.ndarray[numpy.float64_t,ndim=1] b, 
                    numpy.ndarray[int, ndim=1] parents,
                    numpy.ndarray[int, ndim=1] depth, 
                    numpy.ndarray[int, ndim=1] gedge,int m,
                    numpy.ndarray[numpy.float64_t, ndim=1] flow):
    
    cdef int n = len(depth)
    cdef numpy.ndarray[numpy.float64_t] bcopy = b.copy()
    cdef int vtx
    cdef int i,j
    cdef int ptr,otherptr

    for i in xrange(0,m):
        flow[i]=0

    cdef numpy.ndarray[long, ndim=1] sorteddepth = numpy.argsort(depth)


    for i in xrange(n-1,0,-1):
        vtx=sorteddepth[i]
        ptr=Bindptr[gedge[vtx]]
        if(Bindices[ptr]!=vtx):
            otherptr=ptr
            ptr+=1
        else:
            otherptr=ptr+1
                                        
        flow[gedge[vtx]]=bcopy[vtx]/Bdata[ptr]
        bcopy[parents[vtx]]-=Bdata[otherptr]*flow[gedge[vtx]]
        
# calculate the cycle probabilities
cpdef get_probs(numpy.ndarray[numpy.int8_t, ndim=1] Fdata,
                numpy.ndarray[int, ndim=1] Findptr,
                numpy.ndarray[int, ndim=1] Findices,
                numpy.ndarray[numpy.float64_t, ndim=1] R,
                numpy.float64_t resist_mod):

            
            cdef int m = len(R)
            cdef int rest=len(Findptr)-1
            
            cdef numpy.float64_t sum = 0
            cdef numpy.ndarray[numpy.float64_t] probs = numpy.zeros(rest)
            cdef int i,j
            for i in xrange(0,m):
                for j in xrange(Findptr[i],Findptr[i+1]):
                    if(Findices[j]==i):
                        probs[i] += 1
                    else:
                        probs[i] += R[Findices[j]]/(R[i]/resist_mod)
                sum += probs[i]

            for i in xrange(m,rest):
                for j in xrange(Findptr[i],Findptr[i+1]):
                    probs[i] += R[Findices[j]]
                sum += probs[i]
            
            probs[0]=probs[0]/sum
            
            for i in xrange(1,rest):
                probs[i]=probs[i-1]+(probs[i]/sum)
                

            return probs
            
# find tau of the tree
cpdef find_tau(numpy.ndarray[numpy.int8_t, ndim=1] Fdata,
          numpy.ndarray[int, ndim=1] Findptr,
          numpy.ndarray[int, ndim=1] Findices,
          numpy.ndarray[numpy.float64_t, ndim=1] R):

                cdef double tau = 0
                cdef int m = len(R)
                cdef int i,j

                for i in xrange(0,m):
                    for j in xrange(Findptr[i],Findptr[i+1]):
                        if(Findices[j]!=i):
                            tau += R[Findices[j]]/R[i]

                return tau
