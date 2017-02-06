import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from io import StringIO
from io import BytesIO

import mnist_cnn
from PIL import Image

from recog.image3 import ImageParser

dir = '/home/mhkim/data/images'

if os.path.exists(dir) == False :
    os.mkdir('/home/mhkim/data/images')

param = b'testing'

'fewfe'



#im = Image.new("RGB", (512,512), "white")
img1 = Image.open(os.path.join(dir, 'number_font.png'))
# pix = img1.load()

img1Width = img1.size[0] - 40
img1Height = img1.size[1] - 20
#len(pix)

rate = 28 / img1Height

resizeWidth = int(img1Width * rate)
resizeHeight = 28

img2 = img1.crop((20, 10 , img1Width, img1Height))

img3 = img2.resize((int(resizeWidth), resizeHeight))

#plt.imshow(img3)
#plt.show()

position = 1
size = 28
begin = (position - 1) * size

test_data = img3.crop(((position - 1) * size, 0, begin + size, 28))

#plt.imshow(test_data)
#plt.show()

pix = test_data.load()

array = [ round(0.2126 * pix[i, j][0] + 0.7152 * pix[i, j][1] + 0.0722 * pix[i, j][2]) for j in range(28) for i in range(28) ]

_arr1 = []
_arr2 = []
index = 0

for i in range(len(array)) :
    num = array[i]
    if num > 0 :
        _arr2.append([0])
    else :
        _arr2.append([1])

    if (i+1) % 28 == 0:
        #print ( _arr2)
        _arr1.append(_arr2)
        _arr2 = []


#plt.imshow(_arr1)
#plt.show()

#print ( array )

imageDir = '/home/mhkim/data/images'

parser = ImageParser(os.path.join(imageDir, 'number_font.png'), channel=1 , resize=28)

images = parser.getImageArray()

mnistCnn = mnist_cnn.MnistCnn()

for i in range(len(images)):

    resultValue = mnistCnn.execute([images[i]])

    print ( resultValue)
#
