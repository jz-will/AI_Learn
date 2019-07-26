# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# img = cv2.imread('./img/test2.jpg', 1)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# height, width, depth = img.shape

# 灰度处理：
# B=G=R; (B+G+R)/3;
# gray=r*0.299+g*0.578+b*0.114 优化:整数,(r+(g<<1)+b)>>2
# API：imread、cvtColor

# 颜色反转: 255-当前

# 马赛克效果:一定范围内的bgr值保持一致

# 毛玻璃: 随机像素(注意不要越界)

# 图片融合: 随机混合bgr值

# 边缘检测(卷积运算): canny: gray 高斯滤波 canny
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgG = cv2.GaussianBlur(gray, (3, 3), 0)
# dst = cv2.Canny(img, 50, 50)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)
# sobel: 算子模板, 图片卷积, 阈(yu)值判决
# 矩阵的 逆置矩阵
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# dst = np.zeros((height, width, 1), np.uint8)
# for i in range(0, height - 2):
#     for j in range(0, width - 2):
#         gy = gray[i, j] * 1 + gray[i, j + 1] * 2 + gray[i, j + 2] * 1 - gray[i + 2, j] * 1 - gray[i + 2, j + 1] * 2 - \
#              gray[i + 2, j + 2] * 1
#         gx = gray[i, j] + gray[i + 1, j] * 2 + gray[i + 2, j] - gray[i, j + 2] - gray[i + 1, j + 2] * 2 - gray[
#             i + 2, j + 2]
#         # 计算梯度
#         grad = math.sqrt(gx * gx + gy * gy)
#         if grad > 50:
#             dst[i, j] = 255
#         else:
#             dst[i, j] = 0
#
# cv2.imshow('dst', dst)
# cv2.waitKey(0)

# 浮雕效果:
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# dst = np.zeros((height, width, 1), np.uint8)
# for i in range(0, height):
#     for j in range(0, width - 1):
#         grayP0 = int(gray[i, j])
#         grayP1 = int(gray[i, j + 1])
#         newP = grayP0 - grayP1 + 150
#         if newP > 255:
#             newP = 255
#         elif newP < 0:
#             newP = 0
#         dst[i, j] = newP  # 单通道
#
# cv2.imshow('dst', dst)
# cv2.waitKey(0)

# 颜色映射: 变换公式(简单图片)
# 油画特效(减少细节)
