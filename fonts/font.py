from PIL import Image, ImageFont, ImageDraw

import numpy as np
import matplotlib.pyplot as plt

import sys , os

class FontUtil :

    def __init__(self, text="0123456789"):
        self._text = text

        #print ( fonts)
    def load(self, filePath):
        with open("fonts.txt", "r") as f :
            self._fonts = [ line.replace('\n', '') for line in f.readlines() ]

    def getFullImages (self) :
        list = []
        for font in self._fonts:
            list.append(self.getToImage(font, self._text))

    def getToImage (self, fontPath, strs) :
        returnValue = []
        for _char in strs :

            im = Image.new("RGB", (28, 28))

            draw = ImageDraw.Draw(im)

            font = ImageFont.truetype(fontPath, 30)

            draw.text((0, 0), _char, font=font)

            im = im.convert('L')

            im = np.array(im.getdata()).reshape(im.size[0], im.size[1])

            returnValue.append(im)

        return returnValue

    '''
    for font in fonts :
        print(font)
        try :
            fonta = ImageFont.truetype(font, 30)
            #print(font.size)
        except :
            print ('error ' , font)
    '''

def main() :

    fontUtil = FontUtil()

    test = '/home/mhkim/tools/android-studio/plugins/android/lib/layoutlib/data/fonts/DroidSans.ttf'

    # test = img.imread(test)

    # plt.imshow(test)
    # plt.show()

    result = fontUtil.getToImage(test, '123')
    #
    print(result)
    #
    for c in result:
        plt.imshow(c)
        plt.show()

if __name__ == '__main__' :
    main()