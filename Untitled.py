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


# In[2]:


M = 100
_M = str(M)
Z = np.genfromtxt("result-"+_M+"-"+_M+"-"+_M+".dat", delimiter="\n")
Z = Z.reshape((-1, M-1, M-1, M-1))


data = Z[0, :, :, :]
positive = np.clip(data, 0, data.max())
negative = np.clip(-data, 0, -data.min())
d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
d2[..., 0] = positive * (255./positive.max())
d2[..., 1] = negative * (255./negative.max())
d2[..., 2] = d2[...,1]
d2[..., 3] = 10

d2[:, 0, 0] = [255,0,0,100]
d2[0, :, 0] = [0,255,0,100]
d2[0, 0, :] = [0,0,255,100]


"""
Demonstrates GLVolumeItem for displaying volumetric data.

"""


from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 200
w.show()
w.setWindowTitle('pyqtgraph example: GLVolumeItem')

#b = gl.GLBoxItem()
#w.addItem(b)
g = gl.GLGridItem()
g.scale(10, 10, 1)
w.addItem(g)

import numpy as np


v = gl.GLVolumeItem(d2)
#v.translate(-50,-50,-100)
w.addItem(v)

ax = gl.GLAxisItem()
w.addItem(ax)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

