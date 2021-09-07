from Graph import M
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

if len(sys.argv) != 5:
    print("Ejecutar como 'python Graph2.py <filename> <M> <N> <O>")
    exit()

name = sys.argv[1]
M = sys.argv[2]
N = sys.argv[3]
O = sys.argv[4]

Z = np.genfromtxt(name, delimiter="\n")
Z = Z.reshape((-1, O, N, M))
#Z = np.nan_to_num(Z, copy=False)
#Z = Z[:, 1:-1, 1:-1, 1:-1]

L = Z.shape[0]
print(Z.shape)

l = 0
data = Z[l]#, 1:-1, 1:-1, 1:-1]
print(data.shape)


min = data.min()
interr = (data - min)
scale = (interr )#np.clip(interr, 0, interr.max())
positive = (np.clip(scale, 0.0000, scale.max()))#np.clip(interr, 0, interr.max())
negative = (np.clip(-scale, 0.0000, -scale.min()))
d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
#d2[..., 0] = positive/positive.max() * (255.)
#d2[..., 1] = negative/negative.max() * (255.)
d2[..., 0] = scale/scale.max() * (255.)
d2[..., 1] = scale/scale.max() * (255.)
d2[..., 2] = 255# - scale/scale.max() * (255.)
d2[..., 3] = 30
d2[:, 0, 0] = [255,0,0,250]
d2[0, :, 0] = [0,255,0,250]
d2[0, 0, :] = [0,0,255,250]


"""
Demonstrates GLVolumeItem for displaying volumetric data.

"""


from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

app = QtGui.QApplication([])
w = gl.GLViewWidget()
# Switch to 'nearly' orthographic projection.
w.opts['distance'] = 2000
w.opts['fov'] = 1
w.show()
w.setWindowTitle('pyqtgraph example: GLVolumeItem')

#b = gl.GLBoxItem()
#w.addItem(b)
g = gl.GLGridItem()
g.translate(50,50,0)
g.scale(1, 1, 1)
w.addItem(g)

import numpy as np


v = gl.GLVolumeItem(d2)
w.addItem(v)

ax = gl.GLAxisItem()
w.addItem(ax)


def update():
    global v, d2, l, Z, t
    print(l)
    ## update volume colors
    #data = Z[l, :, :, :]
    data = Z[l]#, 1:-1, 1:-1, 1:-1]
    interr = (data)# - (data.min()))
    print("Valor maximo:", data.max())
    print("T00 minimo:", data.min())
    scale = (interr )#np.clip(interr, 0, interr.max())
    positive = (np.clip(scale, 0.0000, scale.max()))#np.clip(interr, 0, interr.max())
    negative = (np.clip(-scale, 0.0000, -scale.min()))
    d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
    d2[..., 0] = positive/positive.max() * (255.)
    d2[..., 1] = negative/negative.max() * (255.)
    #d2[..., 0] = scale/scale.max() * (255.)
    #d2[..., 1] = scale/scale.max() * (255.)
    d2[..., 2] = 255# - scale/scale.max() * (255.)
    d2[..., 3] = 30




    v.setData(d2)
    l = (l+1)
    if (l == Z.shape[0]):
        print("REINICIOOOOOOOOOOOOOOOOOOOOOOOOOOOOO!!")
        l=0


pop = QtCore.QTimer()
pop.timeout.connect(update)
pop.start(100)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



