import os
import argparse
import time
import pickle
import csv
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, get_context
from helper_scripts.wit_url_downloader import download_wit_urls
from helper_scripts.wit_clip_class import CLIP
from helper_scripts.wit_dtype import DTYPE, DFLENGTH, DFLENGTH_ENGLISH
from helper_scripts.wit_image_downloader import wit_download_image
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK']='True'

ONLYENGLISH = True
MULTIPROCESSING = True
THREAD_COUNT = multiprocessing.cpu_count()
SIMILARITIESFOLDER = './wit/witsimilarities'
EMBEDDINGSFOLDER = './wit/witembeddings'
WITURLFOLDER = './wit/witurls'
EMBEDDINGS_PER_PICKLE = 5000
CALCULATE_EMBEDDINGS_EVERY = 100

dflengths = DFLENGTH_ENGLISH if ONLYENGLISH else DFLENGTH

parser = argparse.ArgumentParser()

parser.add_argument('--wit_url_folder', type=str,
                    help='Download location for WIT urls.')

parser.add_argument('--onepercentsample', 
                    dest='onepercentsample', 
                    action='store_true',
                    help='Only download 1% sample file.')

parser.add_argument('--saveimages', 
                    dest='saveimages', 
                    action='store_true',
                    help='Save the images on the local drive.')

parser.add_argument('--saveembeddings', 
                    dest='saveembeddings', 
                    action='store_true',
                    help='Save the image embeddings on the local drive.')

parser.add_argument('--savewds', 
                    dest='savewds', 
                    action='store_true',
                    help='Save the images and best matching caption as WebDataset')

args = parser.parse_args()

wit_url_folder = args.wit_url_folder if args.wit_url_folder else WITURLFOLDER

clipper = CLIP()

os.makedirs(SIMILARITIESFOLDER, exist_ok=True)
if args.saveembeddings:
    os.makedirs(EMBEDDINGSFOLDER, exist_ok=True)

dtv = list(DTYPE.keys())
caption_dict = {0:dtv[4], 1:dtv[5], 2:dtv[6], 3:dtv[7], 4:dtv[8], 5:dtv[15], 6:dtv[16]}

def download_image(row):
    image_url = row[3]
    saveimages = row[-2]
    valid, image_request = wit_download_image(image_url, saveimages)
    return valid, row[-1], image_request

def calculate_embeddings(row): # , image_request):
    saveembeddings = row[-4]
    captions = [
        row[4],  # row.page_title,
        row[5],  # row.section_title,
        row[6],  # row.hierarchical_section_title,
        row[7],  # row.caption_attribution_description,
        row[6],  # row.caption_alt_text_description,
        row[15], # row.context_page_description,
        row[16]  # row.context_section_description
    ]
    available_captions = [True if isinstance(x, str) else False for x in captions]
    caption_tuples = [(i, x) for i, x in enumerate(captions) if available_captions[i]]
    available_ids, captions = list(zip(*caption_tuples))
    similarities, embeddings = clipper.return_similarities(row[-1], captions, row[2])
    similarities = {caption_dict[j]: round(similarities[i], 4) for i, j in enumerate(available_ids) }
    if not saveembeddings:
        embeddings = None
    return row[-2], similarities, embeddings


if __name__ == '__main__':
    start = time.time()
    download_wit_urls(urlfolder=wit_url_folder, onepercentsample=args.onepercentsample)
    fns = [x for x in os.listdir(wit_url_folder) if x[0] != '.' and '.tsv.gz' in x]

    for i, wit_filename in enumerate(fns):
        print('Processing {}. file: {}...'.format(i+1, wit_filename))
        global_counter = 0
        similarities_dict = {}
        embeddings_dict_counter = 0
        if args.saveembeddings:
            embeddings_dict = {}
            if '1percent' in wit_filename:
                prefix = "onepercent"
            else:
                prefix = 'main' + (wit_filename[-17])
        df = pd.read_csv(
            os.path.join(wit_url_folder, wit_filename), 
            sep="\t", 
            compression="gzip", 
            quotechar='"', 
            dtype=DTYPE,
            error_bad_lines=False)
        if ONLYENGLISH:
            df = df[df['language'] == 'en']
            dflen = DFLENGTH_ENGLISH[wit_filename]
        else:
            dflen = DFLENGTH[wit_filename]
    
        df['saveembeddings'] = args.saveembeddings
        df['saveimages'] = args.saveimages
        df['index'] = df.index

        embeddings_dict = {}
        results = []
        images = []
        index_list = []
        exception_list = []

        if MULTIPROCESSING:
            with get_context("spawn").Pool(THREAD_COUNT) as p:
                for valid, index, image in tqdm(p.imap_unordered(download_image, df.itertuples(name=None)), total=dflen):
                    if valid:
                        index_list.append(index)
                        images.append(image)
                    else:
                        exception_list.append((index, image))

                    if len(index_list) > CALCULATE_EMBEDDINGS_EVERY:

                        df_filtered = df[df['index'].isin(index_list)]
                        df_filtered.loc[:, 'images'] = images
                        df_filtered.to_excel('test.xlsx')

                        for r in zip(*df_filtered.to_dict("list").values()):
                            result = calculate_embeddings(r)
                            results.append(result)

                        with open('./wit/exceptions_{}.csv'.format(wit_filename[:-6]),'a') as f:
                            writer=csv.writer(f)
                            for line in exception_list:
                                writer.writerow([])
                                writer.writerow(list(line))

                        images = []
                        index_list = []
                        exception_list = []
                        index_image_dict = {}   
                p.close()

        else:
            with get_context("spawn").Pool(THREAD_COUNT) as p:
                for row in tqdm(df.itertuples(name=None), total=dflen):
                    valid, index, image = download_image(row)
                # for valid, index, image in tqdm(p.imap_unordered(download_image, df.itertuples(name=None)), total=dflen):
                    if valid:
                        index_list.append(index)
                        images.append(image)
                    else:
                        exception_list.append((index, image))

                    if len(index_list) > CALCULATE_EMBEDDINGS_EVERY:

                        df_filtered = df[df['index'].isin(index_list)]
                        df_filtered.loc[:, 'images'] = images
                        df_filtered.to_excel('test.xlsx')

                        for r in zip(*df_filtered.to_dict("list").values()):
                            iresult = calculate_embeddings(r)
                            results.append(result)

                        with open('./wit/exceptions_{}.csv'.format(wit_filename[:-6]),'a') as f:
                            writer=csv.writer(f)
                            for line in exception_list:
                                writer.writerow([])
                                writer.writerow(list(line))

                        images = []
                        index_list = []
                        exception_list = []
                        index_image_dict = {}   

        for result in results:
            if result[0] != False:
                index, sim, emb = result
                similarities_dict[index] = sim
                if args.saveembeddings:
                    embeddings_dict[index] = emb
                    if len(embeddings_dict.keys()) >= EMBEDDINGS_PER_PICKLE:
                        with open(os.path.join(
                            EMBEDDINGSFOLDER, 
                            '{}_{:05d}_image_embeddings.pkl'.format(prefix, embeddings_dict_counter)
                        ), 'wb') as f:
                                pickle.dump(embeddings_dict, f)
                                embeddings_dict_counter += 1

        global_counter += dflen

        similarity_df = pd.DataFrame.from_dict(similarities_dict, orient='index')
        similarity_df.index.name = 'index'
        similarity_df.index = similarity_df.index.astype(int)
        similarity_df = similarity_df.sort_index()
        similarity_df.to_csv(
            os.path.join(
                SIMILARITIESFOLDER, 
                wit_filename.replace('.tsv.gz', '') + '_with_similarities' + '.tsv'
            ), sep="\t")

    end = time.time()
    elapsed = end - start
    print('Finished processing {} WIT-rows in {} hours!'.format(global_counter, elapsed/60))