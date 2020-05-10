import bolom
import numpy as np
import scipy.spatial


vec=np.array([[1,1,0],[2,1,0],[2,5,0],[1,5,0],[1,1,3],[2,1,3],[2,5,3],[1,5,3]])
valsF = lambda x,y,z:2*x+3*y+4*z
tri = scipy.spatial.Delaunay(vec)     
vals =  valsF(vec[:,0],vec[:,1], vec[:,2])
ii=bolom.Interpolator(tri, ['x'], [vals])

print (ii([[1.5,1.5,1.5],[1.5,1.5,1.5],[1,1,1]]))

