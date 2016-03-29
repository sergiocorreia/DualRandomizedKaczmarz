#include <stdint.h>
#include <math.h>

static void update_c(int8_t *Fdata,
                     int *Findices,
                     int *Findptr,
                     double *R,
                     double *flow,
                     double resist_mod,
                     int m,
                     int num_cycles,
                     int cycle,
                     int tracker,
                     long *edges) {

  double inner=0;
  double dotproduct=0;
  int fund_edges=0;
  int other_edges=0;
  int j;

  if(tracker==1) {
    edges[0]=0;
    edges[1]=0;
  }

  if(cycle < m) {
    for(j=Findptr[cycle]; j<Findptr[cycle+1]; ++j) {
      dotproduct += flow[Findices[j]]*(R[Findices[j]]/resist_mod)*Fdata[j];
      inner += R[Findices[j]]/resist_mod;
      fund_edges += 1;
    }
  }
  else {
    for(j=Findptr[cycle]; j<Findptr[cycle+1]; ++j) {
      dotproduct += flow[Findices[j]]*(R[Findices[j]]/resist_mod)*Fdata[j];
      inner += R[Findices[j]]/resist_mod;
      other_edges += 1;
    }
  }

  double factor = dotproduct/inner;

  for(j=Findptr[cycle]; j<Findptr[cycle+1]; ++j) {
    flow[Findices[j]] = flow[Findices[j]] - factor*Fdata[j];
  }
  
  if(tracker==1) {
    edges[0]=fund_edges;
    edges[1]=other_edges;
  }

}

// search cycle distribution function for cycle
static int binary_search_c(double *probs,
                           double target,
                           int start,
                           int stop) {
  int mid;
  while(start < stop) {
    mid= (stop+start)/2;

    //search first half
    if(probs[mid] > target) {
      stop = mid;
    }
    else {
      start = mid+1;
    }
  }
  
  return start;
}

static void induce_voltage_c(double *flow,
                             double *Bdata,
                             int *Bindices,
                             int *Bindptr,
                             double *R,
                             int *parents,
                             int *gedge,
                             long *sorteddepth,
                             double *v,
                             int n) {
  
  int i,vtx,ptr;

  for(i=0; i<n; ++i) {
    v[i]=0;
  }

  for(i=0; i<n; ++i) {
    vtx = sorteddepth[i];
    ptr = Bindptr[gedge[vtx]];

    if(Bindices[ptr]!=vtx)
      ptr+=1;

    v[vtx]=flow[gedge[vtx]]*R[gedge[vtx]]/Bdata[ptr]+v[parents[vtx]];
  }
}

static void matvec_c(double *Ldata,
                     int *Lindices,
                     int *Lindptr,
                     double *v,
                     double *b,
                     int n) {
  
  int i,j;

  for(i=0; i<n; ++i) {
    for(j=Lindptr[i]; j<Lindptr[i+1]; ++j) {
      b[i]=b[i]+Ldata[j]*v[Lindices[j]];
    }
  }
}

static double relative_residual_c(double *v,
                                double *b,
                                double *Ldata,
                                int *Lindices,
                                int *Lindptr,
                                int n) {

  int i;
  double *temp = (double*) malloc(sizeof(double) * n);
  
  for(i=0; i<n; ++i) {
    temp[i]=0;
  }

  matvec_c(Ldata,Lindices,Lindptr,v,temp,n);

  double normbv=0;
  for(i=0; i<n; ++i) {
    temp[i]=temp[i]-b[i];
  }
  for(i=0; i<n; ++i) {
    normbv+=temp[i]*temp[i];
  }

  normbv=sqrt(normbv);

  double normb=0;

  for(i=0; i<n; ++i) {
    normb+=b[i]*b[i];
  }
  normb=sqrt(normb);

  free(temp);
  
  return normbv/normb;
}

static void get_probs_c(int8_t *Fdata,
                        int *Findices,
                        int *Findptr,
                        double *R,
                        double *probs,
                        double resist_mod,
                        int m,
                        int num_cycles) {
  double sum=0;
  
  int i,j;
  
  for(i=0; i<m; ++i) {
    for(j=Findptr[i]; j<Findptr[i+1]; ++j) {
      if(Findices[j]==i) {
        probs[i]+=1;
      }
      else {
        probs[i]+=R[Findices[j]]/(R[i]/resist_mod);
      }
    }
    sum+=probs[i];
  }

  for(i=m; i<num_cycles; ++i) {
    for(j=Findptr[i]; j < Findptr[i+1]; ++j) {
      probs[i] += R[Findices[j]];
    }
    sum+=probs[i];
  }

  probs[0]=probs[0]/sum;

  for(i=1; i<num_cycles; ++i) {
    probs[i]=probs[i-1]+(probs[i]/sum);
  }

}


static void initialize_flow_c(double *Bdata,
                            int *Bindices,
                            int *Bindptr,
                            double *b,
                            int *parents,
                            long *sorteddepth,
                            int *gedge,
                            double *flow,
                            int n,
                            int m) {
  
  int i,vtx,ptr,otherptr;

  for(i=0; i<m; ++i) {
    flow[i]=0;
  }

  for(i=n-1;i>0;i--) {
    vtx=sorteddepth[i];
    ptr=Bindptr[gedge[vtx]];
    
    if(Bindices[ptr]!=vtx) {
      otherptr=ptr;
      ptr+=1;
    }
    else {
      otherptr=ptr+1;
    }
    
    flow[gedge[vtx]]=b[vtx]/Bdata[ptr];
    b[parents[vtx]]-=Bdata[otherptr]*flow[gedge[vtx]];
  }
}
                            
