import cv2

import numpy as np
# imge = cv2.imread("G:\\photos\\1552745034335.jpg", 1)

imge = cv2.imread("./img/test2.jpg", 1)
arr = np.array(imge)

# row, cols  = arr.reshape
# for i in row:
#     for j in cols:
#         if(arr[i,j]<)
# [b, g, r] = imge[100, 100]
# print(b, g, r)
# print([b, g])
# for [a,b] in imge[a,b]:

cv2.imshow("imge", imge)
width = imge.shape[0]
height = imge.shape[1]
print(imge[108, 108])
# width = imge.width
# height = imge.height
for x in range(width):
    for y in range(height):
        pixel = imge[x, y]
        if pixel[-1] > pixel[0] and pixel[-1] > pixel[1] and pixel[1] < 100:
            # print(pixel)
            pixel[-1] = 0
            imge[x, y] = pixel

cv2.imshow('processed', imge)
cv2.waitKey(0)
