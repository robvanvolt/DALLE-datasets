import pandas as pd
import os
import requests
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool


# https://github.com/google-research-datasets/conceptual-12m
cc_url = 'https://storage.googleapis.com/conceptual_12m/cc12m.tsv'

root_folder = ''

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

if not os.path.isfile(root_folder + '/cc12m.tsv'):
    print('Missing cc12m url-caption-dataset. Downloading...')
else:
    print('cc12m.tsv already downloaded. Proceeding with downloading images!')

dfc = pd.read_csv(root_folder + "/cc12m.tsv", sep='\t', names=["url", "caption"])
image_folder = root_folder + '/images'
text_folder = root_folder + '/texts'
output_folder = root_folder + '/output'
skip_folder = root_folder + '/skip'
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
thread_count = 32

df = dfc.loc[~dfc.index.isin(skiplist)]
print('Remaining {} images to be downloaded - {} ({:.5f} %) already downloaded.'.format(remaining, len(skiplist), percent_remaining))

def load_image_and_caption(x):
    name, url, caption = x
    id = "0"*(9-len(str(name))) + str(name)
    suffix = ''
    ending = url.split('.')[-1].lower()
    if ending.lower() in imageformats:
        suffix = '.' + ending
    try:
        foo = Image.open(requests.get(url, stream=True, timeout=3).raw)
        a = max(maxwidth/foo.size[0], maxheight/foo.size[1])
        foo = foo.resize((int(foo.size[0] * a), int(foo.size[1] * a)), Image.ANTIALIAS)
        foo.save(Path(image_folder + "/" + id + '.jpg'), optimize=True, quality=85)
    except Exception:
        open(Path(skip_folder + '/' + id), 'a').close
        pass
    else:
        with open(Path(text_folder + '/' + id + '.txt'), 'w') as f:
            f.write(caption)

z = zip(df.index, df["url"], df["caption"])
pool = Pool(thread_count)
for _ in tqdm(pool.imap_unordered(load_image_and_caption, z), total=len(df)):
        pass
print('Finished downloading available images from conceptual images!')
