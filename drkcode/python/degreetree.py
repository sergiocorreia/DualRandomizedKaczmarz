import scipy.io
import scipy
from scipy.sparse.csgraph import minimum_spanning_tree

def get_tree(A):
    A=A.tocsc()
    degree=A.sum(0).getA1()
    
    Adata=A.data
    Aindices=A.indices
    Aindptr=A.indptr
    n=A.shape[0]
    for i in range(0,n):
        for j in range(Aindptr[i],Aindptr[i+1]):
            Adata[j]=degree[Aindices[j]]+degree[i]

    
    T=-minimum_spanning_tree(-A)
    return T
