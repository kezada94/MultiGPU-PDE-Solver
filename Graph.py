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

M = sys.argv[1]
n = sys.argv[2]
q = sys.argv[3]
_M = str(M)
Z = np.genfromtxt("result-"+str(n)+"-"+str(q)+"-"+_M+".dat", delimiter="\n")
Z = Z.reshape((-1, 3, 100, 100))

L = Z.shape[0]
print(Z.shape)

data = Z[0, :, :, :]
interr = (data)
print(data.max())
print(data.min())
positive = np.log(np.clip(interr, 0.0000, interr.max()))#np.clip(interr, 0, interr.max())
negative = np.log(np.clip(-interr, 0.0000, -interr.min()))
d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
d2[..., 0] = positive/positive.max() * (255.)
d2[..., 1] = negative/negative.max() * (255.)
d2[..., 2] = 255
d2[..., 3] = 1
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
g.scale(5, 5, 1)
w.addItem(g)

import numpy as np


v = gl.GLVolumeItem(d2)
v.scale(3,3,3)
w.addItem(v)

ax = gl.GLAxisItem()
w.addItem(ax)

index = 0
def update():
    global v, L, Z, index
    index += 1
    #print(index)
    if(L == 1):
        return
    if(index >= L):
        index=0
    data = Z[index, :, :, :]
    interr = (-data)
    positive = np.log(interr-interr.min()+1)#np.clip(interr, 0, interr.max())
    negative = np.clip(-interr, 0, -interr.min())
    d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
    d2[..., 0] = positive/positive.max() * (255.)
    d2[..., 1] = 0#negative/negative.max() * (255.)
    d2[..., 2] = 255
    d2[..., 3] = 100

    d2[:, 0, 0] = [255,0,0,250]
    d2[0, :, 0] = [0,255,0,250]
    d2[0, 0, :] = [0,0,255,250]
    v.setData(d2)



timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(17)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

