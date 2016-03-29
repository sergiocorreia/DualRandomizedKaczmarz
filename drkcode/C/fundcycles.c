// fundcycles.c
//
// Copyright (C) <2016> <Kevin Deweese>
// All rights reserved.
//
// This software may be modified and distributed under the terms
// of the BSD license.  See the LICENSE file for details.


#include <stdint.h>
#include <math.h>

static int fundcycles_c(int *FI,
                        int *FJ,
                        int8_t *FV,
                        double *Bdata,
                        int *Bindices,
                        int *Bindptr,
                        int *parent,
                        int *depth,
                        int *gedge,
                        int *nontree,
                        int maxF,
                        int mF,
                        int nontreecount,
                        int n,
                        int *finished,
                        int *reused) {

  int i,v,w,ptr,vv,ww;

  for(i=*reused; i<nontreecount; ++i) {
    if(maxF-mF < n) {
      *reused=i;
      return mF;
    }

    ptr=Bindptr[nontree[i]];
    if(Bdata[ptr]>0) {
      v=Bindices[ptr];
      w=Bindices[ptr+1];
    }
    else {
      v=Bindices[ptr+1];
      w=Bindices[ptr];
    }

    if(depth[v]==-1 || depth[w]==-1)
      continue;

    mF=mF+1;
    FI[mF]=nontree[i];
    FJ[mF]=nontree[i];
    FV[mF]=-1;
    
    
    while(v!=w) {
      if(depth[v] >= depth[w]) {
        mF=mF+1;
        FI[mF]=gedge[v];
        FJ[mF]=nontree[i];
        vv=parent[v];
        FV[mF]= ((v-vv) > 0) - ((v-vv) < 0);
        //FV[mF]=1;
        v=vv;
      }
      
      if(depth[w] > depth[v]) {
        mF=mF+1;
        FI[mF]=gedge[w];
        FJ[mF]=nontree[i];
        ww=parent[w];
        FV[mF]=((ww-w) > 0) - ((ww-w) < 0);
        //FV[mF]=-1;
        w=ww;
      }
    }
  }

  *finished=1;
  return mF;
}
