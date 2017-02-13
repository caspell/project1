from PIL import Image, ImageFont, ImageDraw

import numpy as np
import matplotlib.pyplot as plt

import sys , os


with open("fonts.txt", "r") as f :
    fonts = [ line.replace('\n', '') for line in f.readlines() ]


#print ( fonts)
nums = "0123456789"


def getToImage (fontPath, strs) :
    returnValue = []
    for _char in strs :

        im = Image.new("RGB", (28, 28))

        draw = ImageDraw.Draw(im)

        font = ImageFont.truetype(fontPath, 30)

        draw.text((0, 0), _char, font=font)

        im = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)

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


# remove unneccessory whitespaces if needed
# im2=palate.crop(palate.getbbox())

# im=palate





# im = im2.copy()


test = '/home/mhkim/tools/android-studio/plugins/android/lib/layoutlib/data/fonts/DroidSans.ttf'




result = getToImage(test, '123')

print(result)

for c in result :
    plt.imshow(c)
    plt.show()
