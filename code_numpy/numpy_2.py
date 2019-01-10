import numpy as np
a = np.array([[0,1,2,3,4],
              [9,8,7,6,5]])
print(a)
x = np.array([[0,1,2,3,4],
              [9,8,7,6]])
x.shape     #尺度。对于矩阵，n行m列
x.dtype     #元素类型
x
x.itemsize  #每个元素的大小
x.size      #元素的个数

#非同质对象可构成adarray数组但无法有效发挥numpy优势，尽量避免使用

