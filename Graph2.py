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
Z = np.genfromtxt("result2-"+str(n)+"-"+str(q)+"-"+_M+".dat", delimiter="\n")
Z = Z.reshape((-1, 1000, 1000, 3))

L = Z.shape[0]
print(Z.shape)

data = Z[0, 1:, 1:, :]
interr = (data - (data.min()))
print("T00 maximo:", data.max())
print("T00 minimo:", data.min())
scale = (interr )#np.clip(interr, 0, interr.max())
#positive = (np.clip(scale, 0.0000, scale.max()))#np.clip(interr, 0, interr.max())
#negative = (np.clip(-scale, 0.0000, -scale.min()))
d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
#d2[..., 0] = positive/positive.max() * (255.)
#d2[..., 1] = negative/negative.max() * (255.)
d2[..., 0] = scale/scale.max() * (255.)
d2[..., 1] = scale/scale.max() * (255.)
d2[..., 2] = 255 - scale/scale.max() * (255.)
d2[..., 3] = 255
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
w.addItem(v)

ax = gl.GLAxisItem()
w.addItem(ax)
## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

