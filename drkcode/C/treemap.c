// treemap.c
//
// Copyright (C) <2016> <Kevin Deweese>
// All rights reserved.
//
// This software may be modified and distributed under the terms
// of the BSD license.  See the LICENSE file for details.

static void treemap_c(int *Tindptr,
                    int *Tindices,
                    int *row,
                    int *col,
                    int *treemap,
                    int m) {
  int i,j,idx;
  idx=0;

  for(i=0; i<m; ++i) {
    for(j=Tindptr[row[i]]; j<Tindptr[row[i]+1]; ++j) {
      if(Tindices[j]==col[i]) {
        treemap[idx]=i;
        idx+=1;
      }
    }
  }
}
