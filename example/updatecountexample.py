import drkcode.python.solver
import drkcode.python.degreetree
import os
import scipy.io


filename="email.mtx"
A=scipy.io.mmread(filename)
#if (A.nnz > 10000):
#    continue
        
T=drkcode.python.degreetree.get_tree(A)

results=drkcode.python.solver.solve(A,tolerance=1e-6,userTree=T,useres=1,tracker=1)

# O(length cycle) cost per cycle update
RK_edges=results[0]

# O(log(n)) cost per cycle update
RK_logn_times_iters=results[1]

# O(log(cycle length)) cost per cycle update
RK_log_edges=results[2]

# O(1) cost per cycle update
RK_projections=results[3]



print "O(length cycle) cost per cycle update"
print RK_edges
print "O(log(n)) cost per cycle update"
print RK_logn_times_iters
print "O(log(cycle length)) cost per cycle update"
print RK_log_edges
print "O(1) cost per cycle update"
print RK_projections




