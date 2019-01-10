import numpy as np
from PIL import Image   #(R, G, B)
a = np.arange(15).reshape(3,5)
np.sum(a)
np.mean(a,axis=1)
np.mean(a,axis=0)
np.average(a,axis=0, weights=[10,5,1])  #加权平均
b=np.arange(15,0,-1).reshape(3,5)   #15-1
np.max(b)
np.argmax(b)     #扁平化后的下标
np.unravel_index(np.argmax(b), b.shape) #重塑成多维下标
np.ptp(b)   #最大值与最小值的差
np.median(b)    #中位数

a = np.random.randint(0, 20, (5))
np.gradient(a)  #返回元素梯度（一维）或者返回梯度函数（多维）
c = np.random.randint(0,50,(3,5))
np.gradient(c)  #返回两个数组，最外层和第二层维度的梯度

