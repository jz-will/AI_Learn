# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 主要运用矩阵之间的运算来处理像素值

# img = cv2.imread('./img/test2.jpg', 1)
# cv2.imshow('image', img)
# height, width, depth = img.shape

# 图片移动
# matshift = np.float32([[1, 0, 10], [0, 1, 20]])   # 2*3
# dat = cv2.warpAffine(img, matshift, (height, width))    # move
# cv2.imshow('data', dat)
# cv2.waitKey(0)

# 图片镜像
# newImgInfo = (height * 2, width, depth)
# dst = np.zeros(newImgInfo, np.uint8)
# for i in range(0, height):
#     for j in range(0, width):
#         dst[i, j] = img[i, j]  # np - 像素点的赋值
#         dst[height * 2 - i - 1, j] = img[i, j]
#
# for i in range(0, width):
#     dst[height, i] = (0, 0, 255)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)

# 图片缩放-API
# dst = cv2.resize(img, (int(height / 2), int(width / 2)))
# cv2.imshow('dst', dst)
# cv2.waitKey(0)
# 算法结构
# dstheight = int(height / 2)
# dstwidth = int(width / 2)
# dstImage = np.zeros((dstheight, dstwidth, 3), np.uint8)  # 0-255
# for i in range(0, dstheight):
#     for j in range(0, dstwidth):
#         inew = int(i * (height * 1.0 / dstheight))
#         jnew = int(j * (width * 1.0 / dstwidth))
#         dstImage[i, j] = img[inew, jnew]
# cv2.imshow('dst', dstImage)
# cv2.waitKey(0)

# 图片缩小
# matScale = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
# dst = cv2.warpAffine(img, matScale,
#                      (int(height / 2), int(width / 2)))
# cv2.imshow('dst', dst)
# cv2.waitKey(0)

# 仿射变换--三点定面
# matSrc = np.float32([[0, 0], [0, height - 1], [width - 1, 0]])  # three points
# matDst = np.float32([[50, 50], [300, height - 200], [width - 300, 100]])
# matAffine = cv2.getAffineTransform(matSrc, matDst)
# dst = cv2.warpAffine(img, matAffine, (height, width))
# cv2.imshow('dst', dst)
# cv2.waitKey(0)

# 图片旋转：先要缩放，避免出界
# matRotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), 45, 0.7) # scale
# dst = cv2.warpAffine(img, matRotate, (height, width))
# cv2.imshow('dst', dst)
# cv2.waitKey(0)
