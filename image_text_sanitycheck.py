# /home/robert/Developer/conceptualimages/images/008962747.jpg
from PIL import Image
from pathlib import Path
import os

img_folder = 'images'

fns = os.listdir(img_folder)

fileformats = ['jpg', 'jpeg', 'png', 'bmp']

fns = [x for x in fns if x.split('.')[1].lower() in fileformats]

badfiles = []

for fn in fns:
    try:
        img = Image.open(Path(img_folder + '/' + fn))
        img.verify()
        img.close()
        print('Good file {}'.format(fn))
    except (IOError, SyntaxError) as e:
        print('Bad file {}'.format(fn))
        badfiles.append(fn)

print(badfiles)

with open('badfiles.txt', 'w') as f:
    for fn in badfiles:
        f.write(fn + '\n')

