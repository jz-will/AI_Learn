# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 彩色图片直方图
# cv2.calcHist()、 line()

# (b,g,r) = cv2.split(imge)

# 直方图均衡化
# cv2.equalizeHist()

# 彩色直方图
# img = cv2.imread('./img/test2.jpg', 1)
# height, width, mode = img.shape
# count_b = np.zeros(256, np.float)
# count_g = np.zeros(256, np.float)
# count_r = np.zeros(256, np.float)
# for i in range(height):
#     for j in range(width):
#         (b, g, r) = img[i, j]
#         index_b = int(b)
#         index_g = int(g)
#         index_r = int(r)
#         count_b[index_b] = count_b[index_b] + 1
#         count_g[index_g] = count_g[index_g] + 1
#         count_r[index_r] = count_r[index_r] + 1
#
# for i in range(0, 256):
#     count_b[i] = count_b[i] / (height * width)
#     count_g[i] = count_g[i] / (height * width)
#     count_r[i] = count_r[i] / (height * width)
#
# x = np.linspace(0, 255, 256)
#
# y1 = count_b
# plt.figure()
# plt.bar(x, y1, 0.9, alpha=1, color='b')
#
# y2 = count_g
# plt.figure()
# plt.bar(x, y2, 0.9, alpha=1, color='g')
#
# y3 = count_r
# plt.figure()
# plt.bar(x, y3, 0.9, alpha=1, color='r')
#
# plt.show()
# # cv2.waitKey(0)

# 图片修补
# 1、坏图 2、array确定图片损坏的位置  3、cv2.inpaint

# 灰度直方图均衡化--统计颜色出现的概率--累计概率p--计算映射表(255*p)--映射--显示


# 灰度直方图均衡化--三通道都需要计算

# 亮度增强--bgr分别增加

# 磨皮美白 --双边滤波cv2.bilaterFilter()

# 高斯均值滤波（消除图片小点）（相当于边缘检测） cv2.GaussianBlur()
# 图像卷积： 矩阵对应点相乘并求和（图像剪辑）

# 中值滤波（效果一般）