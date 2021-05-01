import pandas as pd
from pathlib import Path
from PIL import Image
import requests
import os
from pandarallel import pandarallel

pandarallel.initialize()

DATASETFOLDER = 'content'
DATASET = 'yfcc_filtered.csv'

PARENTPATH = '/ssd/f100m/text-image-data'
TEXTFOLDER = 'texts'
IMAGEFOLDER = 'images'
SKIPFOLDER = 'skips'
PREFIX = "F"

KEEPTHESECOLS = ['final_caption', 'url']
IMAGEFORMATS = ['jpg', 'jpeg', 'bmp', 'png']
MAXWIDTH = 320
MAXHEIGHT = 320
CHUNKS = 100000

def write_files(x, folderpath):
    id = PREFIX + "0"*(8-len(str(x.name))) + str(x.name)
    try:
        foo = Image.open(requests.get(x.url, stream=True, timeout=4).raw)
        a = max(MAXWIDTH/foo.size[0], MAXHEIGHT/foo.size[1])
        foo = foo.resize((int(foo.size[0] * a), int(foo.size[1] * a)), Image.ANTIALIAS)
        foo.save(Path(folderpath + '/' + id + '.jpg'), optimize=True, quality=85)
    except Exception:
        pass
    else:
        with open(Path(folderpath + '/' + id + '.txt'), 'w') as f:
            f.write(x.final_caption)

os.makedirs(Path(PARENTPATH), exist_ok=True)

keep_downloading = True
batch = len(os.listdir(Path(PARENTPATH))) - 1
batch = 0 if batch == -1 else batch

while keep_downloading:
    try:
        df = pd.read_csv(Path(DATASETFOLDER + '/' + DATASET), sep="|", skiprows=range(1, batch * CHUNKS + 1), nrows=CHUNKS, header=0, usecols=KEEPTHESECOLS)
        df.index = [x + batch * CHUNKS for x in list(df.index)]
        folderid = PREFIX + "0"*(4-len(str(batch))) + str(batch)
        folderpath = PARENTPATH + '/' + folderid
        print('Saving images to {}.'.format(folderpath))
        os.makedirs(folderpath, exist_ok=True)
        skip = list(set([int(x[1:-4]) for x in os.listdir(folderpath)]))
        df = df[~df.index.isin(skip)]
        print('Skipping {} urls.'.format(len(skip)))
        df.parallel_apply(lambda x: write_files(x, folderpath), axis=1)
        batch += 1
    except:
        print('Except: Failed reading dataframe...')
        keep_downloading = False
        pass
    else:
        print('DataFrame length: '.format(len(df)))
        if len(df) == 0:
            keep_downloading = False
            print('Finished downloading images!')

print('Finished downloading dataset.')
