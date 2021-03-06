import os
import requests
from PIL import Image

maxwidth = 256
maxheight = 256

def wit_download_image(url, saveimages=False):
    foo = requests.get(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0'}, 
                stream=True, 
                timeout=3)
    if saveimages:
        with Image.open(foo) as fooimage:
            a = max(maxwidth/fooimage.size[0], maxheight/fooimage.size[1])
            fooimage = fooimage.resize((int(fooimage.size[0] * a), int(fooimage.size[1] * a)), Image.ANTIALIAS)
            with open(os.path.join('./wit_images/', + id + '.jpg'), 'wb') as file:
                fooimage.save(file, optimize=True, quality=85)
    return foo