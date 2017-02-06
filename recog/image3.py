import matplotlib.pyplot as plt
import numpy as np
import os
import mnist_cnn
from PIL import Image

class ImageParser :
    _images = []
    _positions = []
    def __init__(self , path , channel=1 , resize=0):

        img1 = Image.open(path)

        self._channel = channel
        _width = img1.size[0]
        _height = img1.size[1]

        self._basis = np.min(img1.size)

        _basis = self._basis

        if self._basis < resize :
            resize = self._basis

        _widthPadding = 0
        _heightPadding = 0

        if _width % _basis != 0 :
            _widthPadding = _width // _basis
        if _height % _basis != 0 :
            _heightPadding = _height // _basis

        _im = Image.new("RGB", (_width + _widthPadding, _height + _heightPadding), "white")

        _im.paste(img1, (0,0))

        _pix = _im.load()

        _width = _im.size[0]
        _height = _im.size[1]

        _shiftWidth = int(_width / _basis)
        _shiftHeight = int(_height / _basis)

        _batchSize = _shiftWidth * _shiftHeight

        oneChannel = (lambda pix : round(0.2126 * pix[0] + 0.7152 * pix[1] + 0.0722 * pix[2]) )

        for row in range(_shiftHeight) :
            for cell in range(_shiftWidth):
                cropImage = _im.crop((cell * _basis, row * _basis , (cell+1)*_basis , (row + 1) * _basis))
                _rBasis = _basis
                if resize != 0:
                    cropImage = cropImage.resize((resize, resize))
                    _rBasis = resize
                pixel = cropImage.load()
                cropImage = []
                for x in range(_rBasis) :
                    if channel == 1 :
                        cropImage.append([[oneChannel(pixel[y, x])] for y in range(_rBasis)])
                    else :
                        cropImage.append([[pixel[y, x]] for y in range(_rBasis)])
                self._positions.append((row, cell))
                self._images.append(cropImage)

        img1.close()

    def getImageArray(self):
        return self._images

    def getImagePosition(self , index):
        return self._positions[index]

    def getChannel(self):
        return self._channel


if __name__ == '__main__' :

    imageDir = '/home/mhkim/data/images'

    parser = ImageParser(os.path.join(imageDir, 'number_font.png'), channel=1, resize=28)

    images = parser.getImageArray()

    mnistCnn = mnist_cnn.MnistCnn()
    for i in range(len(images)):

        y , x = parser.getImagePosition(i)

        #print(y, x , images[i])
        resultValue = mnistCnn.execute([images[i]])
        print(resultValue)
        #plt.imshow(images[i])
        #plt.show()


    #print ( images )