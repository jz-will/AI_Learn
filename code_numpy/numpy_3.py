import numpy as np
import matplotlib.pyplot as plt


a=np.linspace(1,10,4)   #根据起步数据等间距填充数据，形成数组
b=np.linspace(1,10,4,endpoint=False)
c=np.concatenate((a,b)) #合拼数组
print(a,b,c)

a=np.ones((2,3,4),dtype=np.int32)   #2维3行4个数据

print(a.reshape(3,8))
print(a)

a=np.ones((2,3,4),dtype=np.int)
print(a)

b=a.astype(np.float)    #数组复制
print(b)

a=np.full((2,3,4),25,dtype=np.int32)
print(a)
print(a.tolist())   #数组向列表转换