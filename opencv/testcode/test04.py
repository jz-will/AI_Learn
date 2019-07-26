from PIL import Image
from pylab import *

from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r'C:\windows\fonts\SimSun.ttc', size=14)
path = "G:\\python_workspace\\AI_Learn\\opencv\\data\\empire.jpg"
im = array(Image.open(path))

figure()
subplot(121)
gray()
# contour(im, origin='image') 运行错误
axis('equal')
axis('off')
title(u'图像轮廓', fontproperties=font)

subplot(122)
hist(im.flatten(), 128)
title(u'图像直方图', fontproperties=font)

plt.xlim([0, 260])
plt.ylim([0, 11000])

show()