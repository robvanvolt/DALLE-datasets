import pandas as pd
import json
import numpy as np
import shutil
from pathlib import Path
import os
from pandarallel import pandarallel

#############################################################################################
##### ATTENTION: run this file after openimages_labels.py !!! ###############################
##### you need to have captions_tran.json generated from the openimages_labels.py !!! #######
#############################################################################################

# Settings
CHUNKS = 500000
OUTPUTFOLDER = 'openimages'

################################################π#############################################
###### ATTENTION ############################################################################
###### You need to download the following 3 files from the conceptual captions website ######
###### https://google.github.io/localized-narratives/ #######################################
#############################################################################################

os.makedirs(OUTPUTFOLDER, exist_ok=True)

dicts = []

for d in open('downsampled-open-images-v4/open_images_validation_captions.jsonl'):
    dicts.append(json.loads(d))

for d in open('downsampled-open-images-v4/open_images_test_captions.jsonl'):
    dicts.append(json.loads(d))

for d in open('downsampled-open-images-v4/open_images_train_v6_captions.jsonl'):
    dicts.append(json.loads(d))

narrative = pd.DataFrame.from_dict(dicts)
narrative.index = narrative['image_id']
narrative = narrative.rename({'caption': 'narrative'})
narrative.columns = [x.replace('caption', 'narrative') for x in narrative.columns]

print('Found {} narrative image-text pairs.'.format(len(narrative)))

with open('captions_train.json') as json_file:
    data = json.load(json_file)

df = pd.DataFrame.from_dict(data, orient='index')
df.index = [x.split('_')[-1] for x in list(df.index)]

narrative_ids = list(narrative.index)

df = df.join(narrative)

df['caption'] = np.where(~df['narrative'].isna(),df['narrative'],df['caption'])

def save_files(x, folder_id):
    shutil.copyfile(x.path, Path('./' + OUTPUTFOLDER + '/' + folder_id + '/' + x.path.split('/')[-1]))
    with open(Path('./' + OUTPUTFOLDER + '/' + folder_id + '/' + x.name + '.txt'), 'w') as f:
        f.write(x.caption)

maxiter = int(len(df) / CHUNKS) + 1

pandarallel.initialize()

for batch in range(maxiter):
    print('Copying image-text-pairs into {}/{} folders.'.format(batch + 1, maxiter))
    folder_id = (4 - len(str(batch)))*'0' + str(batch)
    sdf = df[batch*CHUNKS:(batch+1)*CHUNKS]
    os.makedirs(Path(OUTPUTFOLDER, folder_id), exist_ok=True)
    sdf.parallel_apply(lambda x: save_files(x, folder_id), axis=1)

print('Done copying image-text-pairs to {} folders in outputfolder {}!'.format(maxiter, OUTPUTFOLDER))