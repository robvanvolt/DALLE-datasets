import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import requests
import os
from pandarallel import pandarallel

### seperator = |

##### https://ai.google.com/research/ConceptualCaptions/download
##### download url-caption dataset from 
##### https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250

DATASETFOLDER = 'content'
DATASET = 'Train_GCC-training.tsv'
FILEID = '1edNr-GEYz69RWcsSgskNzjtM--Qxepdz'
URL = 'https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250'

##### download location of image-caption pairs
PARENTPATH = 'output'
TEXTFOLDER = 'texts'
IMAGEFOLDER = 'images'
PREFIX = ""
CHECKALLFOLDERS = True

KEEPTHESECOLS = ['caption', 'url']
IMAGEFORMATS = ['jpg', 'jpeg', 'bmp', 'png']
MAXWIDTH = 320
MAXHEIGHT = 320
CHUNKS = 500000
THREAD_COUNT = 16
HIDE_ERRORS = False

os.makedirs(Path(DATASETFOLDER), exist_ok=True)

#### Helper scripts to download url-caption dataset
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def download_file(url, root_folder):
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(Path(root_folder + '/cc3.tsv'), 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("Error, something went wrong...")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == '__main__':
    assert os.path.isfile(Path(DATASETFOLDER + '/' + DATASET)), print('''
        #################################################################################################################
        Missing cc3m url-caption-dataset. Automatic downloading not supported yet.
        Download https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250
        And put it into following folder: {}
        #################################################################################################################
        '''.format(DATASETFOLDER))

    pandarallel.initialize(nb_workers=THREAD_COUNT)

    ### downloading dataset and resizsing images in parallel
    def write_files(x, folderpath):
        id = PREFIX + "0"*(8-len(str(x.name))) + str(x.name)
        try:
            foo = Image.open(requests.get(x.url, stream=True, timeout=4).raw)
            a = max(MAXWIDTH/foo.size[0], MAXHEIGHT/foo.size[1])
            foo = foo.resize((int(foo.size[0] * a), int(foo.size[1] * a)), Image.ANTIALIAS)
            foo.save(Path(folderpath + '/' + id + '.jpg'), optimize=True, quality=85)
        except Exception as exc:
            if not HIDE_ERRORS:
                print('Failed downloading {} with url {}'.format(id, x.url))
                print(exc)
            pass
        else:
            with open(Path(folderpath + '/' + id + '.txt'), 'w') as f:
                f.write(x.caption)

    os.makedirs(Path(PARENTPATH), exist_ok=True)

    keep_downloading = True
    if CHECKALLFOLDERS:
        batch = 0
    else:
        batch = len(os.listdir(Path(PARENTPATH))) - 1
        batch = 0 if batch == -1 else batch

    while keep_downloading:
        try:
            df = pd.read_csv(Path(DATASETFOLDER + '/' + DATASET), sep="\t", skiprows=range(0, batch * CHUNKS), nrows=CHUNKS, names=KEEPTHESECOLS)
            # df = pd.read_csv(Path(DATASETFOLDER + '/' + DATASET), sep="\t", skiprows=range(0, batch * CHUNKS), nrows=CHUNKS, names=KEEPTHESECOLS)
            df.index = [x + batch * CHUNKS for x in list(df.index)]
            folderid = str(PREFIX) + "0"*(4-len(str(batch))) + str(batch)
            folderpath = PARENTPATH + '/' + folderid
            os.makedirs(folderpath, exist_ok=True)
            skip = list(set([int(x[1:-4]) for x in os.listdir(folderpath)]))
            df = df[~df.index.isin(skip)]
            print('Saving {} images to {}.'.format(len(df), folderpath))
            print('Skipping {} already downloaded urls.'.format(len(skip)))
            df.apply(lambda x: write_files(x, folderpath), axis=1)
            # df.parallel_apply(lambda x: write_files(x, folderpath), axis=1)
        except Exception as excp:
            print('An error occurred trying to download the filtered dataframe.')
            print(excp)
            keep_downloading = False
            pass
        else:
            if len(df) == 0:
                print('Alredy finished downloading images of batch {}!'.format(batch))
            batch += 1

    print('Finished downloading dataset to {}.'.format(PARENTPATH))
