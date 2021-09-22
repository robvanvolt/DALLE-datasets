import pandas as pd
import os
import requests
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import gc
import glob

cc_url = 'https://storage.googleapis.com/conceptual_12m/cc12m.tsv'
root_folder = './'
total = 12423374
maxwidth = 256
maxheight = 256
thread_count = 16
batch = 10000

def load_caption(x):
    name, caption, text_folder = x
    fid = str(int(int(name) / 10000 )) 
    subdir = "0"*(5-len(fid)) + fid
    os.makedirs(Path(text_folder+"/"+subdir), exist_ok=True)
    fp = text_folder + '/' + subdir + "/" + "0"*(9-len(str(name))) + str(name) + '.txt'
    with open(fp, 'w') as f:
        f.write(caption)

def download_file(url):
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(Path(root_folder + '/cc12m.tsv'), 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("Error, something went wrong...")

def load_image(x):
    name, url, image_folder, skip_folder = x
    fid = str(int(int(name) / 10000 )) 
    subdir = "0"*(5-len(fid)) + fid
    os.makedirs(Path(image_folder+"/"+subdir), exist_ok=True)
    id = subdir + "/" + "0"*(9-len(str(name))) + str(name)
    try:
        with Image.open(requests.get(url,
            headers={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0'}, 
            stream=True, timeout=3).raw) as foo:
            a = max(maxwidth/foo.size[0], maxheight/foo.size[1])
            foo = foo.resize((int(foo.size[0] * a), int(foo.size[1] * a)), Image.ANTIALIAS)
            with open(Path(image_folder + "/" + id + '.jpg'), 'wb') as file:
                foo.save(file, optimize=True, quality=85)
    except Exception:
        os.makedirs(Path(skip_folder+"/"+subdir), exist_ok=True)
        open(Path(skip_folder + '/' + id), 'a').close
        pass

if __name__ == '__main__':
    if not os.path.isfile(Path(root_folder + '/cc12m.tsv')):
        print('Missing cc12m url-caption-dataset. Downloading...')
        download_file(cc_url)
    else:
        print('cc12m.tsv already downloaded. Proceeding with downloading images!')

    dfc = pd.read_csv(root_folder + "cc12m.tsv", sep='\t', names=["url", "caption"])

    image_folder = root_folder + '/images'
    text_folder = root_folder + '/texts'
    skip_folder = root_folder + '/skip'

    paths = [image_folder, text_folder, skip_folder]

    for path in paths:
        os.makedirs(path, exist_ok=True)

    def list_ids(path):
        return [int(os.path.splitext(os.path.basename(a))[0]) for a in glob.glob(path+"/**/*")]

    skiplist = list_ids(text_folder)
    remaining = total - len(skiplist)
    percent_remaining = 100 * (total - remaining) / total
    df = dfc.loc[~dfc.index.isin(skiplist)]

    print('Remaining {} captions to be written - {} ({:.5f} %) already written.'.format(remaining, len(skiplist), percent_remaining))

    if len(df) > 0:
        captions = zip(df.index, df["caption"], [text_folder]*len(df))
        pool = Pool(thread_count)
        for _ in tqdm(pool.imap_unordered(load_caption, captions), total=len(df)):
                pass
        pool.close()
    print('Done with captions!')

    skiplist = list_ids(skip_folder) + list_ids(image_folder)
    remaining = total - len(skiplist)
    percent_remaining = 100 * (total - remaining) / total

    df = dfc.loc[~dfc.index.isin(skiplist)]
    print('Remaining {} images to be downloaded - {} ({:.5f} %) already downloaded.'.format(remaining, len(skiplist), percent_remaining))
    images = list(zip(df.index, df["url"], [image_folder]*len(df), [skip_folder]*len(df)))

    for i in tqdm(range(0, len(df), batch)):
        pool = Pool(thread_count)
        for _ in tqdm(pool.imap_unordered(load_image, images[i:i+batch]), total=batch):
            pass
        pool.terminate()
        pool.join()
        del pool
        gc.collect()

    print('Finished downloading available images from conceptual images!')
