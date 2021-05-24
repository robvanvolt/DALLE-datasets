##### WORK IN PROGRESS #####

import requests
from pathlib import Path
import tqdm
import json
import pandas as pd
import os

## DEVELOPMENTTESTING
DEBUGGING = True

DATASETFOLDER = 'datasets'
LABEL_URL = 'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_captions.jsonl'
# LABEL_URL = 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv'
# LABEL_URL = 'https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv'

if not os.path.exists(Path(DATASETFOLDER + '/' + LABEL_URL.split('/')[-1])):
    print('Downloading labeldataset...')
    # download(LABEL_URL, DATASETFOLDER + '/' + LABEL_URL.split('/')[-1])
    r = requests.get(LABEL_URL)
    with open(Path(DATASETFOLDER + '/' + LABEL_URL.split('/')[-1]), 'wb') as f:
        f.write(r.content)
    print('Finished downloading labeldataset!')
else:
    print('Found labeldataset!')

urls = []

os.makedirs(DATASETFOLDER, exist_ok=True)

for i in range(10):
    urls.append('https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train' + str(i) + '.tsv')

if DEBUGGING:
    urls = urls[:2]

for url in urls:
    if not os.path.exists(Path(DATASETFOLDER + '/' + url.split('/')[-1])):
        print('Downloading ' + url)
        r = requests.get(url)
        with open(Path(DATASETFOLDER + '/' + url.split('/')[-1]), 'wb') as f:
            f.write(r.content)
        print('Downloaded ' + url)
    else:
        print(url + ' already downloaded. Skipping.')

with open(Path(DATASETFOLDER + '/' + LABEL_URL.split('/')[-1]), encoding='utf-8') as narrative:
    results = [json.loads(jline) for jline in narrative.read().splitlines()]

# print(result[:2])

URL2ID = 'old_train-annotations-human-imagelabels-boxable.csv'

urldf = pd.read_csv(DATASETFOLDER + '/' + URL2ID)

input(urldf)

for result in results:
    input(result)

for url in urls:
    df = pd.read_csv(Path(DATASETFOLDER + '/' + url.split('/')[-1]))
    input(df)
# with open(Path(DATASETFOLDER + '/' + LABEL_URL.split('/')[-1]), encoding='utf-8') as narrative:
#     data = json.load(narrative)
# label_df = pd.read_json(data)
# input(label_df)


# requests.get(url, filename="open-images-dataset-train0.tsv")