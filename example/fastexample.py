import drkcode.python.solver
import drkcode.python.degreetree
import os
import scipy.io


filename="email.mtx"
A=scipy.io.mmread(filename)

        
T=drkcode.python.degreetree.get_tree(A)

results=drkcode.python.solver.solve(A,tolerance=1e-6,userTree=T,useres=1)
print "solution"
print results[11]
print "found to"
print results[9]
