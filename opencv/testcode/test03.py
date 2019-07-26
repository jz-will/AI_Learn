# -*-coding: utf-8-*-
from PIL import Image
from pylab import *

from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"C:\windows\fonts\SimSun.ttc", size=14)

path = "G:\\python_workspace\\AI_Learn\\opencv\\data\\empire.jpg"
im = array(Image.open(path))
figure()

# 画有坐标轴
subplot(121)
imshow(im)
x = [100, 100, 400, 400]
y = [200, 500, 200, 500]
# plot绘图
plot(x, y, 'r*')    # r*: 红色星星
title(u'绘图："empire.jpg"', fontproperties=font)

# 不显示坐标轴
subplot(122)
imshow(im)
plot(x, y, 'r*')
plot(x[:2], y[:2])  #连线
axis('off')
title(u'绘图："empire.jpg"', fontproperties=font)

show()
