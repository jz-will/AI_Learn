import numpy as np
a = np.arange(24).reshape((2,3,4))
print(a)
print(a.mean())
a=a/a.mean()
print(a)

a=np.arange(24).reshape((2,3,4))
print(np.square(a)) #计算平方
a = np.sqrt(a)      #计算平方根
print(a)
print(np.modf(a))   #将整数部分和小数部分全部返回
