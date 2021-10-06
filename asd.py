import numpy as np
def u(x, y, t):
    sum = 0
    for i in range(100):
        for j in range(100):
            sum += (1600.0/((2.0*j-1)*(2.0*i-1)*np.pi**2))*np.sin(((2*j-1)*np.pi*x)/10.0)*np.sin(((2*i-1)*np.pi*y)/20.0)*pow(2.71828, -(((2*i-1)*(2*i-1)/(400))+(((2*j-1)*(2*j-1))/(100)))*1*np.pi*np.pi*t*0.0001)
    return sum

x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)

X, Y = np.meshgrid(x, y)
#print(u(X, Y, 1))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import random

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Z = u(X, Y, 10000000)
from matplotlib import cm

print(Z.shape)
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
