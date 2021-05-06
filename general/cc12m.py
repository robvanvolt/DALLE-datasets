import pandas as pd
import os
import requests
from pathlib import Path
from PIL import Image
from pandarallel import pandarallel
pandarallel.initialize()
from tqdm import tqdm

# https://github.com/google-research-datasets/conceptual-12m
cc_url = 'https://storage.googleapis.com/conceptual_12m/cc12m.tsv'

def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

if not os.path.isfile('cc12m.tsv'):
    print('Missing cc12m url-caption-dataset. Downloading...')
else:
    print('cc12m.tsv already downloaded. Proceeding with downloading images!')

dfc = pd.read_csv("cc12m.tsv", sep='\t', names=["url", "caption"])

image_folder = 'images'
text_folder = 'texts'
output_folder = 'output'
skip_folder = 'skip'
paths = [image_folder, text_folder, output_folder, skip_folder]
for path in paths:
    os.makedirs(path, exist_ok=True)
imageformats = ['jpg', 'jpeg', 'bmp', 'png']
skips = os.listdir(skip_folder) + [x[:-4] for x in os.listdir(text_folder)]
total = 12423374
skiplist = [int(x) for x in skips]
remaining = total - len(skiplist)
percent_remaining = 100 * (total - remaining) / total
maxwidth = 256
maxheight = 256

df = dfc.loc[~dfc.index.isin(skiplist)]
print('Remaining {} images to be downloaded - {} ({:.5f} %) already downloaded.'.format(remaining, len(skiplist), percent_remaining))

def load_image_and_caption(x):
    id = "0"*(9-len(str(x.name))) + str(x.name)
    suffix = ''
    ending = x.url.split('.')[-1].lower()
    if ending.lower() in imageformats:
        suffix = '.' + ending
    try:
        foo = Image.open(requests.get(x.url, stream=True, timeout=3).raw)
        a = max(maxwidth/foo.size[0], maxheight/foo.size[1])
        foo = foo.resize((int(foo.size[0] * a), int(foo.size[1] * a)), Image.ANTIALIAS)
        foo.save(Path(image_folder + "/" + id + '.jpg'), optimize=True, quality=85)
    except Exception:
        open(Path(skip_folder + '/' + id), 'a').close
        pass
    else:
        with open(Path(text_folder + '/' + id + '.txt'), 'w') as f:
            f.write(x.caption)

df.parallel_apply(lambda x: load_image_and_caption(x), axis=1)
print('Finished downloading available images from conceptual images!')
