'''
数组的索引和切片
'''
import numpy as np

a = np.array([9,8,7,6,5])
print(a[2])
print(a[1 : 4 : 2])     #起始编号 : 终止编号 : 步长

a = np.arange(24).reshape((2,3,4))
print(a)
print(a[1,2,3])
print(a[0,1,2])
print(a[-1,-2,-3])
#每个维度一个索引值，逗号分隔

a = np.arange(24).reshape((2,3,4))
print(a)
print(a[:, 1, -3])  #选取一个维度用
print(a[:, 1:3, :]) #每个维度切片方法与一维数组相同
print(a[:, :, ::2]) #每个维度可以使用步长跳跃切片

