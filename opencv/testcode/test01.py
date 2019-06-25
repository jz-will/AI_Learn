# -*- coding: utf-8 -*-
from  PIL import Image
from pylab import *

#添加中文字体支持
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\windows\fonts\Simsun.ttc", size=14)

pil_im = Image.open("G:\\python_workspace\\AI_Learn\\opencv\\data\\empire.jpg")

subplot(121)
title(u'原图', fontproperties=font)
axis('off')
imshow(pil_im)

#灰度图
pil_im = Image.open("G:\\python_workspace\\AI_Learn\\opencv\\data\\empire.jpg").convert('L')
gray()
subplot(122)
title(u'灰度图',fontproperties = font)
axis('off')
imshow(pil_im)

show()