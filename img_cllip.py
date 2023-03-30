# -*- coding: utf-8 -*-
import os
from PIL import Image

filename = r'FreezeD_output/vangogh_houses.png'
out = 'FreezeD_output'
if not os.path.exists(out):
    os.makedirs(out)
img = Image.open(filename)
size = img.size
print(size)

# 准备将图片切割成64张小图片
weight = int(size[0] // 2)
height = int(size[1] // 1)
# 切割后的小图的宽度和高度
print(weight, height)

for j in range(1):
    for i in range(2):
        box = (weight * i+2, height * j+2, weight * (i + 1) , height * (j + 1) )
        region = img.crop(box)
        region.save('FreezeD_output/{}_{}.png'.format(j, i))
# os.remove(os.path.join(out, '1.jpg'))