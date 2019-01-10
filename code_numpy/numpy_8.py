import numpy as np
a = np.arange(100).reshape(5,2,10)
np.save('a.npy', a) #文件的第一行是维度信息
b = np.load('a.npy')
#print(b)
a = np.random.rand(3,4,5)   #均匀分布
#print(a)
a = np.random.randn(3,4,5)  #正态分布
#print(a)
b = np.random.randint(100,200,(3,4))    #100-200每次不一样
#print(b)
np.random.seed(10)
np.random.randint(100,200,(3,4))    #种子使每次输出一样
#print(b)
np.random.shuffle(b)
#print(b)
np.random.shuffle(b)            #顺序与数都变
#print(b)
#print(np.random.permutation(b)) #顺序变数不变
#print(b)
b = np.random.randint(100,200,(8,))
print(b)
#print(np.random.choice(b,(3,2)))    #维度限定
print(np.random.choice(b,(3,2), replace=False))
print(np.random.choice(b,(3,2), p=b/np.sum(b)))
