import pandas as pd
import numpy as np
from pathlib import Path
from pandarallel import pandarallel
import os
import requests
from PIL import Image

pandarallel.initialize()

FILENAME = 'wit_url_captions/wit_v1.train.all-00000-of-00010.tsv.gz'
GROUP = FILENAME[38]
CHUNKS = 500000
TEXTFOLDER = 'texts'
IMAGEFOLDER = 'images'
SKIPFOLDER = 'skips'
IMAGEFORMATS = ['jpg', 'jpeg', 'bmp', 'png']
MAXWIDTH = 320
MAXHEIGHT = 320
REPORT = True
REPORTEVERY = 2500

LANGUAGEFILTER = True
LANGUAGES = ['en']

##### For a future version with auomatic tsv.gz folder reader
# os.listdir(Path('wit_url_captions'))
# MAINFOLDERNAME = 'wit_url_captions'
# FILENAME = 'wit1percent.tsv.gz'
# FILENAMES = os.listdir(Path(MAINFOLDERNAME))

os.makedirs(TEXTFOLDER, exist_ok=True)
os.makedirs(IMAGEFOLDER, exist_ok=True)
os.makedirs(SKIPFOLDER, exist_ok=True)

imagefolders = os.listdir(Path(IMAGEFOLDER))
textfolders = os.listdir(Path(TEXTFOLDER))
skipfiles = os.listdir(Path(SKIPFOLDER))

####### Progress calculation based on image files
def return_total_downloaded_images():
    images = []
    for subimagefolder in os.listdir(Path(IMAGEFOLDER)):
        images += os.listdir(Path(IMAGEFOLDER + '/' + subimagefolder))
    return len(images)

# def return_total_downloaded_texts():
#     texts = []
#     for subtextfolder in os.listdir(Path(TEXTFOLDER)):
#         texts += os.listdir(Path(TEXTFOLDER + '/' + subtextfolder))
#     return len(texts)

imagefiles = []
textfiles = []
skipfilenumbers = []

####### Extracting content of subfolders
for subtextfolder in textfolders:
    textfiles += os.listdir(Path(TEXTFOLDER + '/' + subtextfolder))

for subimagefolder in imagefolders:
    imagefiles += os.listdir(Path(IMAGEFOLDER + '/' + subimagefolder))

######## Calculating downloaded files
if len(imagefiles) > 0:
    imagefilenumbers = [int(x[1:-4]) for x in imagefiles]
else:
    imagefilenumbers = []

if len(textfiles) > 0:
    textfilenumbers = [int(x[1:-4]) for x in textfiles]
else:
    textfilenumbers = []

missing_images = [x for x in textfilenumbers if x not in imagefilenumbers]
missing_texts = [x for x in imagefilenumbers if x not in textfilenumbers]

print('Missing {:,} images and {:,} texts.'.format(len(missing_images), len(missing_texts)))

####### Note skip folder does not have subfolders, size does not matter
####### because it is not needed for training
# if len(skipfiles) > 0:
#     skipfilenumbers = [int(x[1:]) for x in skipfiles]

print('Already downloaded {:,} images and {:,} texts.'.format(len(imagefilenumbers), len(textfilenumbers)))

def load_missing_captions(x, textfolderpath, imagefolderpath, i, itotal, totallength):
    id = "w" + "0"*(9-len(str(x.name))) + str(x.name)
    with open(Path(textfolderpath + '/' + id + '.txt'), 'w') as f:
        f.write(x.caption)

def write_files(x, textfolderpath, imagefolderpath, i, itotal, totallength):
    id = "w" + "0"*(9-len(str(x.name))) + str(x.name)
    try:
        foo = Image.open(requests.get(x.image_url, stream=True, timeout=4).raw)
        a = max(MAXWIDTH/foo.size[0], MAXHEIGHT/foo.size[1])
        foo = foo.resize((int(foo.size[0] * a), int(foo.size[1] * a)), Image.ANTIALIAS)
        foo.save(Path(imagefolderpath + "/" + id + '.jpg'), optimize=True, quality=85)
    except Exception:
        # open(Path(SKIPFOLDER + '/' + id), 'a').close
        pass
    else:
        with open(Path(textfolderpath + '/' + id + '.txt'), 'w') as f:
            f.write(x.caption)
    if REPORT:
        if x.name % REPORTEVERY == 0:
            currentlength = return_total_downloaded_images()
            print('Folder {}/{} - Image {:,}/{:,} ({:.2f} %).'.format(i + 1, itotal, currentlength, totallength, currentlength*100/totallength))

        
    
print('Reading url-caption file...')
df = pd.read_csv(
    Path(FILENAME), 
    compression='gzip', 
    header=0, 
    sep='\t',
    quotechar='"', 
    dtype={
        'language': str,
        'page_url': str,
        'image_url': str,
        'page_title': str,
        'section_title': str,
        'hierarchical_section_title': str,
        'caption_reference_description': str,
        'caption_attribution_description': str,
        'caption_alt_text_description': str,
        'mime_type': str,
        'original_height': int,
        'original_width': int,
        'is_main_image': bool,
        'attribution_passes_lang_id': bool,
        'page_changed_recently': str,
        'context_page_description': str,
        'context_section_description': str
    },
    # sep='\t', 
    error_bad_lines=False)

if LANGUAGEFILTER:
    df = df[df['language'].isin(LANGUAGES)]

print('Preprocessing captions...')
df = df.replace(np.nan, '')
df['caption'] = df['caption_reference_description'] + '\n' + df['caption_attribution_description'] + '\n' + df['caption_alt_text_description'] + '\n' + df['hierarchical_section_title']
df['caption'] = '\n' + df['caption_reference_description'] + '\n' + df['caption_attribution_description'] + '\n' + df['caption_alt_text_description'] + '\n'
df['caption'] = df['caption'].str.replace(r'^\.+\s+', '', regex=True)
df['caption'] = df['caption'].str.replace(r'\.\.+', '.', regex=True)
df['caption'] = df['caption'].str.replace(r'\s\s+', ' ', regex=True)
df['caption'] = df['caption'].str.replace(r'\s+\.+', '', regex=True)
df['caption'] = df['caption'].str.replace(r'&amp;', 'and', regex=True)
df['caption'] = df['caption'].str.strip()
df['index'] = df.index

totallength = len(df)

parts = totallength / CHUNKS

splits = np.array_split(df, parts)

# print(totallength)
# print(len(splits))
itotal = len(splits)
print('The whole dataframe was divided into {} part(s).'.format(itotal))

for i, splitdf in enumerate(splits):
    foldername = "w" + str(GROUP) + '_' + "0"*(3-len(str(i))) + str(i)
    textfolderpath = TEXTFOLDER + '/' + foldername
    imagefolderpath = IMAGEFOLDER + '/' + foldername
    os.makedirs(Path(textfolderpath), exist_ok=True)
    os.makedirs(Path(imagefolderpath), exist_ok=True)
    textfolderfiles = os.listdir(textfolderpath)
    folderfiles = len(os.listdir(imagefolderpath))
    if len(missing_texts) > 0:
        print('Trying do generate missing text files...')
        missingdf = df[df.index.isin(missing_texts)]
        missingdf.parallel_apply(lambda x: load_missing_captions(x, textfolderpath, imagefolderpath, i, itotal, totallength), axis=1)
        textfiles = []
        for subtextfolder in textfolders:
            textfiles += os.listdir(Path(TEXTFOLDER + '/' + subtextfolder))
        textfilestotal = len(textfiles)
        print('Successfully generated {:,} missing text file(s).'.format(len(missingdf)))
    print('Downloading texts into {} and images into {}.'.format(textfolderpath, imagefolderpath))
    filteredsplitdf = splitdf[~splitdf.index.isin(textfilenumbers)]
    # filteredsplitdf = filteredsplitdf[~filteredsplitdf.index.isin(skipfilenumbers)]
    dflength = len(filteredsplitdf)
    downloadedimages = return_total_downloaded_images()
    print('Total length: {:,}'.format(totallength))
    print('Downloaded images: {:,}'.format(downloadedimages))
    print('Remaining images: {:,}\n'.format(totallength - downloadedimages))
    print(dflength)
    if dflength > 0:
        while return_total_downloaded_images() < totallength:
            print('##############')
            print('Remaining {:,} images to get downloaded.'.format(totallength - return_total_downloaded_images()))
            print('##############')
            filteredsplitdf.parallel_apply(lambda x: write_files(x, textfolderpath, imagefolderpath, i, itotal, totallength), axis=1)

textfolders = os.listdir(Path(TEXTFOLDER))
textfiles = []
for subtextfolder in textfolders:
    textfiles += os.listdir(Path(TEXTFOLDER + '/' + subtextfolder))
textfilestotal = len(textfiles)

imagefolders = os.listdir(Path(IMAGEFOLDER))
imagefiles = []
for subimagefolder in imagefolders:
    imagefiles += os.listdir(Path(IMAGEFOLDER + '/' + subimagefolder))
imagefilestotal = len(imagefiles)

print('Finished downloading {:,} images and {:,} texts.'.format(imagefilestotal, textfilestotal))
