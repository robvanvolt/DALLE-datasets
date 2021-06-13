import pandas as pd
import numpy as np
from pathlib import Path
from pandarallel import pandarallel
import os
import requests
from io import BytesIO
from PIL import Image
from cairosvg import svg2png

pandarallel.initialize()

### download urls from here https://github.com/google-research-datasets/wit/blob/main/DATA.md
# FILENAME = 'wit_url_captions/wit_v1.train.all-00000-of-00010.tsv.gz'
URL_FOLDER = 'wit_urls'
URLS = sorted(os.listdir(URL_FOLDER))
# FILENAME = 'wit_v1.train.all-1percent_sample.tsv.gz'
# FILENAME = 'wit_v1.train.all-1percent_sample1.tsv.gz'
CHUNKS = 50000
DATAPARENTFOLDER = 'wit'
MAXWIDTH = 320
MAXHEIGHT = 320

LANGUAGEFILTER = True
LANGUAGES = ['en']

##### For a future version with auomatic tsv.gz folder reader
# os.listdir(Path('wit_url_captions'))
# MAINFOLDERNAME = 'wit_url_captions'
# FILENAME = 'wit1percent.tsv.gz'
# FILENAMES = os.listdir(Path(MAINFOLDERNAME))

def eng_cap(x):
    if 'English:' in x:
        eng = [y.replace('English:', '').strip() for y in x.split('\n') if 'English:' in y]
        return '\n'.join(eng)
    else:
        return x


print('Found {} files containing Image-URLs.'.format(len(URLS)))

for i, FILENAME in enumerate(URLS):
    DATAFOLDER = DATAPARENTFOLDER + '/' + FILENAME
    print('{} - Starting downloading image-text-pairs from {} to {}'.format(i + 1, FILENAME, DATAFOLDER))
    os.makedirs(DATAFOLDER, exist_ok=True)

    path = Path(DATAFOLDER)
    text_files = [*path.glob('**/*.txt')]
    text_files = {text_file.stem: text_file for text_file in text_files} # str(text_file.parents[0]) + 
    text_total = len(text_files)

    image_files = [
        *path.glob('**/*.png'), *path.glob('**/*.jpg'),
        *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
    ]
    image_files = {image_file.stem: image_file for image_file in image_files} # str(image_file.parents[0]) +
    image_total = len(image_files)

    print('Found {:,} textfiles and {:,} images already downloaded.'.format(text_total, image_total))

    keys = (image_files.keys() & text_files.keys())

    def write_files(x, datafolderpath, i, itotal, totallength):
        id = "%09d" % x.name
        try:
            if '.svg' in x.image_url.lower():
                output = svg2png(url=x.image_url)
                foo_t = Image.open(BytesIO(output)).convert("RGBA")
                foo = Image.new("RGBA", foo_t.size, "WHITE")
                foo.paste(foo_t, (0, 0), foo_t)
                a = max(MAXWIDTH/foo.size[0], MAXHEIGHT/foo.size[1])
                foo = foo.resize((int(foo.size[0] * a), int(foo.size[1] * a)), Image.ANTIALIAS).convert('RGB')
                foo.save(Path(datafolderpath + "/" + id + '.jpg'), optimize=True, quality=85)
            else:
                with Image.open(requests.get(x.image_url, stream=True, timeout=4).raw) as foo:
                    if foo.mode == "RGBA":
                        foo = Image.new("RGBA", foo.size, "WHITE")
                        foo.paste(foo, (0, 0), foo)
                    a = max(MAXWIDTH/foo.size[0], MAXHEIGHT/foo.size[1])
                    foo = foo.resize((int(foo.size[0] * a), int(foo.size[1] * a)), Image.ANTIALIAS).convert('RGB')
                    foo.save(Path(datafolderpath + "/" + id + '.jpg'), optimize=True, quality=85)
        except Exception as e:
            print(e)
            print(x.image_url)
            pass
        else:
            with open(Path(datafolderpath + "/" + id + '.txt'), 'w') as f:
                f.write(x.caption)
            
        
    print('Reading url-caption file...')
    df = pd.read_csv(
        Path(URL_FOLDER + '/' + FILENAME), 
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
    df['caption'] = df['caption'].parallel_apply(lambda x: eng_cap(x))
    df['index'] = df.index

    totallength = len(df)

    parts = totallength / CHUNKS

    splits = np.array_split(df, parts)

    # print(totallength)
    # print(len(splits))

    itotal = len(splits)
    print('The whole dataframe was divided into {} part(s).'.format(itotal))

    for i, splitdf in enumerate(splits):
        foldername = "%05d" % i
        datafolderpath = DATAFOLDER + '/' + foldername
        os.makedirs(Path(datafolderpath), exist_ok=True)
        print('Downloading texts and images into {}.'.format(datafolderpath))
        filteredsplitdf = splitdf[~splitdf.index.isin(keys)]
        # filteredsplitdf = filteredsplitdf[~filteredsplitdf.index.isin(skipfilenumbers)]
        dflength = len(filteredsplitdf)
        print('Total length: {:,}'.format(totallength))
        print('Downloading split length: {:,}'.format(dflength))
        filteredsplitdf.parallel_apply(lambda x: write_files(x, datafolderpath, i, itotal, totallength), axis=1)

    text_files = [*path.glob('**/*.txt')]
    text_files = {text_file.stem: text_file for text_file in text_files} # str(text_file.parents[0]) + 
    text_total = len(text_files)

    image_files = [
        *path.glob('**/*.png'), *path.glob('**/*.jpg'),
        *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
    ]
    image_files = {image_file.stem: image_file for image_file in image_files} # str(image_file.parents[0]) +
    image_total = len(image_files)

    print('Found {:,} textfiles and {:,} images already downloaded.'.format(text_total, image_total))

    print('Finished downloading {:,} images and {:,} texts.'.format(image_total, text_total))
