import webdataset as wds
import os
from pathlib import Path
from collections import Counter
from PIL import Image

DATASETPATH = 'data'
OUTPUTFILENAME = 'dataset'

alldirs = os.walk(Path(DATASETPATH))
all_basepaths = []

############################################################################
########### Legacy wds_creator - better use wds_create_shards.py ###########
############################################################################

### (1) Find image-text pairs in all (sub)folders of the basepath

for dir in alldirs:
    fns = dir[2]
    if next((True for x in fns if '.txt' in x), False):
        basenames = [x.split('.')[0] for x in fns]
        basepaths = [dir[0] + '/' + k for k, v in dict(Counter(basenames)).items() if v == 2]
        all_basepaths.extend(basepaths)

all_basepaths = sorted(all_basepaths, key=lambda x: x.split('/')[-1])
curated_basepaths = []


### (2) Verify images exist and can be opened

for basepath in all_basepaths:
    img = Image.open(Path(basepath + '.jpg'))
    try:
        img.verify()
    except Exception:
        print('Invalid image on path {}'.format(basepath))
    else:
        curated_basepaths.append(basepath)


### (3) Create compressed Webdataset tar file

sink = wds.TarWriter(OUTPUTFILENAME + '.tar.gz', encoder=False)

sink.write

for basepath in curated_basepaths:
    with open(Path(basepath + '.jpg'), "rb") as imgstream:
        image = imgstream.read()
    with open(Path(basepath + '.txt'), "rb") as txtstream:
        text = txtstream.read()
    sample = {
        "__key__": basepath.split('/')[-1],
        "img": image,
        "cap": text
    }
    sink.write(sample)
sink.close()