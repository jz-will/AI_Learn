import numpy as np
a = np.arange(24).reshape((2,3,4))
b = np.sqrt(a)
print(a)
print(b)
print(np.maximum(a,b))  #运算结果是浮点类型
print(a>b)    #算术比较，产生布尔型数组