from PIL import Image , ImageFont, ImageDraw

import numpy as np
import matplotlib.pyplot as plt

nums = "0123456789"

fonts = [

]


im = Image.new("RGB", (28, 28))

draw = ImageDraw.Draw(im)

font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 30)

draw.text((0, 0), "9", font=font)

# remove unneccessory whitespaces if needed
#im2=palate.crop(palate.getbbox())

#im=palate




#im = im.save("img.png")

#im = im2.copy()














plt.imshow(im)
plt.show()