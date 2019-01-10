import numpy as np
def pysum():
    a = [0,1,2,3,4]
    b = [9,8,7,6,5]
    c = []

    for i in range(len(a)):
        c.append(a[i]**2 + b[i]**3)
    return c

def npsum():
    a = np.array([0,1,2,3,4])
    b = np.array([9,8,7,6,5])

    c = a**2 + b**3
    return c

print(pysum())
print(npsum())