// treeinfo.c
//
// Copyright (C) <2016> <Kevin Deweese>
// All rights reserved.
//
// This software may be modified and distributed under the terms
// of the BSD license.  See the LICENSE file for details.

static void treeinfo_c(int *index_indptr,
                       int *index_indices,
                       long long *index_data,
                       int *parent,
                       int *depth,
                       int *gedge,
                       int *vperm,
                       int m) {
  int i,j,v;

  for(i=1; i<m+1; ++i) {
    v=vperm[i];
    depth[v]=depth[parent[v]]+1;
    
    for(j=index_indptr[parent[v]]; j<index_indptr[parent[v]+1]; ++j) {
      if(index_indices[j]==v) {
        gedge[v]=index_data[j];
      }
    }
   
  }
}
