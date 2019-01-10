import numpy as np
a = np.arange(100).reshape(5,20)    #0-99
np.savetxt('a.csv',a, fmt='%d', delimiter=',')
b = np.loadtxt('a.csv', delimiter=',')
#print(b)
np.savetxt('a.csv',a, fmt='%.1f', delimiter=',')
b = np.loadtxt('a.csv', dtype=np.int, delimiter=',')
#print(b)

a = np.arange(100).reshape(5,10,2)
a.tofile('b.dat', sep=',', format='%d')
c = np.fromfile('b.dat', dtype=np.int, sep=',')
#print(c)
c = np.fromfile('b.dat', dtype=np.int, sep=',').reshape(5,10,2)
#reshape获得原数组维度信息
print(c)
a.tofile('b.dat', format='%d')
