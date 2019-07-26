# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 绘制线段
# newImageInfo = (500, 500, 3)
# dst = np.zeros(newImageInfo, np.uint8)
# cv2.line(dst, (100, 100), (400, 400), (0, 0, 255), cv2.LINE_AA)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)

# 矩形圆形绘制
# rectangle(dst,begin, end, color, fill?)
# circle(dat, center, r, fill?>0 line)
# ellipse(dat, center, 轴, angle, begin, end, fill?)
# polylines
# points = np.array([[150,50],[140,140],[200,170],[250,250],[150,50]], np.uint8)
# print(points.shape)
# points = points.reshape((-1, 1, 2))
# print(points.shape)
# cv2.polylines(dst, [points], True, (0,255,255))   # failed!
# cv2.imshow('dst',dst)
# cv2.waitKey(0)

# 文字图片绘制
# img = cv2.imread('./img/test2.jpg', 1)
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.rectangle(img, (30, 30), (180, 180), (0, 255, 0), 3)
# cv2.putText(img, 'this is apple', (30, 50), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.imshow('img', img)
# cv2.waitKey(0)
