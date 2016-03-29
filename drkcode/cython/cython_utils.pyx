# cython_utils.pyx -- Utility file containing most of the DRK functions
# 
# Copyright (C) <2016> <Kevin Deweese>
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

# cython: profile=False
# cython.boundscheck(False)
# cython.wraparound(False)
# cython.cdivision(True)

import numpy
cimport numpy
import scipy
import random
from libc.stdlib cimport rand,RAND_MAX,srand,malloc,free

cdef extern from "../C/utils.c":
    void update_c(numpy.int8_t *Fdata,
                  int *Findices,
                  int *Findptr,
                  numpy.float64_t *R,
                  numpy.float64_t *flow,
                  numpy.float64_t resist_mod,
                  int m, int num_cycles, int cycle,
                  int tracker, long *edges)

    int binary_search_c(numpy.float64_t *probs,
                        numpy.float64_t target,
                        int start,
                        int stop)

    void induce_voltage_c(numpy.float64_t *flow,
                          numpy.float64_t *Bdata,
                          int *Bindices,
                          int *Bindptr,
                          numpy.float64_t *R,
                          int *parents,
                          int *gedge,
                          long *sorteddepth,
                          numpy.float64_t *v,
                          int n)

    numpy.float64_t relative_residual_c(numpy.float64_t *v,
                               numpy.float64_t *b,
                               numpy.float64_t *Ldata,
                               int *Lindices,
                               int *Lindptr,
                               int n)

    void get_probs_c(numpy.int8_t *Fdata,
                     int *Findices,
                     int *Findptr,
                     numpy.float64_t *R,
                     numpy.float64_t *probs,
                     numpy.float64_t resist_mod,
                     int m,
                     int num_cycles)

    void initialize_flow_c(numpy.float64_t *Bdata,
                           int *Bindices,
                           int *Bindptr,
                           numpy.float64_t *b,
                           int *parents,
                           long *sorteddepth,
                           int *gedge,
                           numpy.float64_t *flow,
                           int n, int m)
                     
    

cdef get_random():
    cdef numpy.float64_t random_var = rand()
    return random_var/RAND_MAX



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
                  int useres, int maxiters, int tracker):
    cdef int n=len(b)
    cdef numpy.ndarray[numpy.float64_t] stats = numpy.zeros(10)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] v = numpy.zeros(n)
    if(maxiters > -1):
        solve_fixediters(Fdata,Findices,Findptr,Bdata,Bindices,Bindptr,R,resist_mod,Ldata,Lindices,Lindptr,b,v,probs,flow,parents,depth,gedge,maxiters,processors,stats,known_solution,useres,tracker)
    else:
        solve_fixedtol(Fdata,Findices,Findptr,Bdata,Bindices,Bindptr,
                   R,resist_mod,Ldata,Lindices,Lindptr,b,v,probs,
                   flow,parents,depth,gedge,tolerance,processors,stats,known_solution,useres,tracker)

    return(stats[0],stats[1],stats[2],stats[3],stats[4],stats[5],stats[6],
           stats[7],stats[8],stats[9],v)

# solves dual system to a fixed tolerance
cdef solve_fixedtol(numpy.ndarray[numpy.int8_t, ndim=1] Fdata,
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
                    int useres, int tracker):

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
        
        
        cycle = binary_search_c(&probs[0],get_random(),0,num_cycles-1)
        
        #update(Fdata,Findices,Findptr,R,resist_mod,probs,flow,m,num_cycles,cycle,temp_edges)
        update_c(&Fdata[0],&Findices[0],&Findptr[0],&R[0],&flow[0],resist_mod,m,num_cycles,cycle,tracker,&temp_edges[0])

        if(processors > 1):
            for i in xrange(0,m):
                used[i]=0
            for i in xrange(Findptr[cycle],Findptr[cycle+1]):
                used[Findices[i]]=1
        iteration+=1
        if(tracker==1):
            if(temp_edges[0]!=0):
                edges_updated+=temp_edges[0]
                log_edges_updated+=numpy.log2(temp_edges[0])
                maxspan=temp_edges[0]
                maxlogspan=numpy.log2(n)
                logn_times_iters+=numpy.log2(n)
                maxlogespan=numpy.log2(temp_edges[0])
                projections+=1
            if(temp_edges[1]!=0):
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
            cycle = binary_search_c(&probs[0],get_random(),0,num_cycles-1)
            for j in xrange(Findptr[cycle],Findptr[cycle+1]):
                if(used[Findices[j]]==1):
                    clash = 1
                    break
                else:
                    used[Findices[j]]=1

            if clash:
                continue
            else:
                update_c(&Fdata[0],&Findices[0],&Findptr[0],&R[0],&flow[0],resist_mod,m,num_cycles,cycle,tracker,&temp_edges[0])
                if(tracker==1):
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
            induce_voltage_c(&flow[0],&Bdata[0],&Bindices[0],&Bindptr[0],&R[0],&parents[0],&gedge[0],&sorteddepth[0],&v[0],n)
            if(useres==1):
                err=relative_residual_c(&v[0],&b[0],&Ldata[0],&Lindices[0],&Lindptr[0],n)
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
                      numpy.ndarray[numpy.float64_t, ndim=1] Bdata,
                      numpy.ndarray[int, ndim=1] Bindices,
                      numpy.ndarray[int, ndim=1] Bindptr,
                      numpy.ndarray[numpy.float64_t, ndim=1] R, 
                      numpy.float64_t resist_mod, 
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
                      int maxiters, int processors,
                      numpy.ndarray[numpy.float64_t, ndim=1] stats,
                      numpy.ndarray[numpy.float64_t, ndim=1] known_solution,
                      int useres, int tracker):

    cdef int iteration=0
    cdef long edges_updated=0
    cdef long log_edges_updated=0
    cdef long logn_times_iters=0
    cdef int num_cycles = len(Findptr)-1
    cdef int projections=0
    cdef int totalspan=0
    cdef long maxspan = 0
    cdef long maxlogspan = 0
    cdef long totallogspan=0
    cdef long totallogespan=0
    cdef long maxlogespan=0
    cdef long other_edges=0

    cdef numpy.ndarray[long, ndim=1] sorteddepth = numpy.argsort(depth) 
    
    cdef numpy.ndarray[long, ndim=1] temp_edges =numpy.zeros(2,dtype=int) 
    cdef numpy.float64_t relres
    cdef int n=len(b)
    cdef int m=len(flow)
                
    cycle = binary_search_c(&probs[0],get_random(),0,num_cycles-1)

    while(iteration < maxiters):
        maxspan=0
        maxlogspan=0
        maxlogespan=0

        if(processors>1):
            used = numpy.zeros(m,dtype=int)
        
        cycle = binary_search_c(&probs[0],get_random(),0,num_cycles-1)
        
        update_c(&Fdata[0],&Findices[0],&Findptr[0],&R[0],&flow[0],resist_mod,m,num_cycles,cycle,tracker,&temp_edges[0])

        iteration+=1
        if(tracker==1):
            for i in xrange(Findptr[cycle],Findptr[cycle+1]):
                used[Findices[i]]=1

            if(temp_edges[0]!=0):
                edges_updated+=temp_edges[0]
                log_edges_updated+=numpy.log2(temp_edges[0])
                maxspan=temp_edges[0]
                maxlogspan=numpy.log2(n)
                logn_times_iters+=numpy.log2(n)
                maxlogespan=numpy.log2(temp_edges[0])
                projections+=1
            if(temp_edges[1]!=0):
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
            cycle = binary_search_c(&probs[0],get_random(),0,num_cycles-1)
            for j in xrange(Findptr[cycle],Findptr[cycle+1]):
                if(used[Findices[j]]==1):
                    clash = 1
                    break
                else:
                    used[Findices[j]]=1

            if clash:
                continue
            else:
                update_c(&Fdata[0],&Findices[0],&Findptr[0],&R[0],&flow[0],resist_mod,m,num_cycles,cycle,tracker,&temp_edges[0])
                if(tracker==1):
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
            
    
    induce_voltage_c(&flow[0],&Bdata[0],&Bindices[0],&Bindptr[0],&R[0],&parents[0],&gedge[0],&sorteddepth[0],&v[0],n)
    if(useres==1):
        err=relative_residual_c(&v[0],&b[0],&Ldata[0],&Lindices[0],&Lindptr[0],n)
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



# initialize the flow vector
cpdef initialize_flow(numpy.ndarray[numpy.float64_t, ndim=1] Bdata,
                    numpy.ndarray[int, ndim=1] Bindptr,
                    numpy.ndarray[int, ndim=1] Bindices,
                    numpy.ndarray[numpy.float64_t,ndim=1] b, 
                    numpy.ndarray[int, ndim=1] parents,
                    numpy.ndarray[int, ndim=1] depth, 
                    numpy.ndarray[int, ndim=1] gedge,int m,
                    numpy.ndarray[numpy.float64_t, ndim=1] flow):
    
    cdef int n = len(depth)
    cdef numpy.ndarray[numpy.float64_t] bcopy = b.copy()

    cdef numpy.ndarray[long, ndim=1] sorteddepth = numpy.argsort(depth)

    initialize_flow_c(&Bdata[0],&Bindices[0],&Bindptr[0],&bcopy[0],&parents[0],&sorteddepth[0],&gedge[0],&flow[0],n,m)
    
# calculate the cycle probabilities
cpdef get_probs(numpy.ndarray[numpy.int8_t, ndim=1] Fdata,
                numpy.ndarray[int, ndim=1] Findptr,
                numpy.ndarray[int, ndim=1] Findices,
                numpy.ndarray[numpy.float64_t, ndim=1] R,
                numpy.float64_t resist_mod):

            
            cdef int m = len(R)
            cdef int num_cycles=len(Findptr)-1

            
            cdef numpy.float64_t sum = 0
            cdef numpy.ndarray[numpy.float64_t] probs = numpy.zeros(num_cycles)
            
            get_probs_c(&Fdata[0],&Findices[0],&Findptr[0],&R[0],&probs[0],resist_mod,m,num_cycles)
            
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
