import sage as sg
from sage.all import *
import numpy as np
import sys

if (len(sys.argv) != 4):
    print("Please execute with <n> <q> <SIZE>.")
    exit()

N = int(sys.argv[3])
_q = int(sys.argv[2])
n = int(sys.argv[1])
var("E0 A x l q A B C E Q Q1 r y D e")

interr = eval("sqrt((A+q**2*l*sin(x)**2)/(2*E0 + q**2*A*sin(x)**2))")

e = Integer(1)
val = 0.00001
while abs(e.numerical_approx(digits=30)) > 0.000000000001:
    integral = interr.subs({A:1, q:_q, l:1, E0: val})
    e = (2*pi) - numerical_integral(integral, 0, n*pi, max_points=n*1000)[0]
    val -= e*1/500
    print("Val: ", val)
    print("Err: ", e)
    print("Cur: ", numerical_integral(integral, 0, n*pi)[0])
val.numerical_approx(digits=30)

E0_hat = val.numerical_approx(digits=30)

der = eval("sqrt((A+q**2*l*sin(x)**2)/(2*E0 + q**2*A*sin(x)**2))")
der = der.subs({A:1, q:_q, l:1, E0: E0_hat})

integral = np.zeros(N, dtype=np.double)
arr = np.zeros(N, dtype=np.double)
delta = np.double(2*np.pi/(N-1))
alfa = np.double(0)
lastAlfa = np.double(0)
for i in range(1, N):
    alfa = integral[i-1]
    integral[i] = integral[i-1] + numerical_integral(der, lastAlfa, alfa, max_points=500)[0]
    rdesired = np.double(delta*i)
    robtained = integral[i]
    while abs(rdesired-robtained) > 0.000000000001:
        integral[i] = integral[i-1] + numerical_integral(der, lastAlfa, alfa, max_points=500)[0]
        robtained = integral[i]
        alfa+=(rdesired-robtained)*0.1
        #print(delta*i-robtained)
    print(alfa, "|", rdesired, "|", robtained, "|", i)
    lastAlfa = alfa
    arr[i] = alfa
np.savetxt("../alfa(r)-"+str(n)+"-"+str(_q)+"-"+str(N)+".csv", arr, delimiter=',', fmt='%.20f')

