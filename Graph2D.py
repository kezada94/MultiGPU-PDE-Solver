from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

## Create a GL View widgimport numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as anim
from tqdm import tqdm
import time

import pyqtgraph as pg
import pyqtgraph.opengl as gl

import sys
# In[2]:
"""
M = sys.argv[1]
n = sys.argv[2]
q = sys.argv[3]
t = int(sys.argv[4])
_M = str(M)
if (t==0):
    Z = np.genfromtxt("result-"+str(n)+"-"+str(q)+"-"+_M+"-A.dat", delimiter="\n")
elif t==1:
    Z = np.genfromtxt("result-"+str(n)+"-"+str(q)+"-"+_M+"-F.dat", delimiter="\n")
else:
    Z = np.genfromtxt("result-"+str(n)+"-"+str(q)+"-"+_M+"-G.dat", delimiter="\n")


Z = Z.reshape((-1, 1, 20, 20))


L = Z.shape[0]
print(Z.shape)


data = Z[l, 0]#, 1:-1, 1:-1, 1:-1]
print(data.shape)

"""
"""
Demonstrates GLVolumeItem for displaying volumetric data.

"""
l = 0
C = 1290
k = 315
d = 19300

ad = k/(C*d)
print(ad)
cols = 20
rows = 20
x = np.linspace(0, 1, cols)
y = np.linspace(0, 1, rows)

X, Y = np.meshgrid(x, y)

u = lambda h, v, t : h*(1-h)*v*(1-v)*(1+0.5*0.1*t)

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

app = pg.mkQApp("GLSurfacePlot Example")
w = gl.GLViewWidget()
# Switch to 'nearly' orthographic projection.
w.show()
w.setWindowTitle('pyqtgraph example: GLVolumeItem')


#b = gl.GLBoxItem()
#w.addItem(b)
g = gl.GLGridItem()
w.addItem(g)

def u(x, y, t):
    global ad
    sum = 0
    t=t*0.001
    for n in range(1,10):
        for m in range(1,10):
            sum += (((1+(-1)**(m+1))*(1+(-1)**(n+1)))/(m**3*n**3))*np.sin(m*np.pi/2*x)*np.sin(n*np.pi/3*y)*np.cos(np.pi*np.sqrt(9*m**2+4*n**2)*t)
    return (576/np.pi**6)*sum

x = np.linspace(0, 2, 100)
y = np.linspace(0, 3, 100)

X, Y = np.meshgrid(x, y)
#print(u(X, Y, 1))
import numpy as np
sp = gl.GLSurfacePlotItem(x=x, y = y, shader='heightColor', computeNormals=False, smooth=False)
#sp2 = gl.GLSurfacePlotItem(x=x, y = y, shader='heightColor', computeNormals=False, smooth=False)
sp.shader()['colorMap'] = np.array([0.2, 2, 0.5, 0.2, 1, 1, 0.2, 0, 2])

w.addItem(sp)
#w.addItem(sp2)

ax = gl.GLAxisItem()
w.addItem(ax)


def update():
    global sp, l, Z, x, y, X, Y, u
    print(l)
    ## update volume colors
    #data = Z[l, :, :, :]
    data = u(X,Y,l)#, 1:-1, 1:-1, 1:-1]
    print(data.shape)
    sp.setData(x=x, y = y, z= data)
    #sp2.setData(x=x, y = y, z= u(X,Y,l))
    l = (l+1)


pop = QtCore.QTimer()
pop.timeout.connect(update)
pop.start(100)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    pg.mkQApp().exec_()

