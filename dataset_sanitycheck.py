from PIL import Image
from pathlib import Path
import pandas as pd
import argparse
import os
import collections

parser = argparse.ArgumentParser(description='A script to sanitycheck your image-text pairs for DALLE-pytorch training.')
parser.add_argument("-D", "--dataset_folder", help="Add the folder containing image-text pairs for DALLE-training.", required=True)
parser.add_argument("-DEL", "--delete_incomplete_files", help="Decide if the incomplete/corrupt files shall be removed.", default=False)
parser.add_argument("-O", "--output_file", help="Incomplete files get saved in a textfile.", default='incomplete_files.csv')
parser.add_argument("-M", "--min_characters", help="Text files with less than the specified character length get deleted.", default=5)
args = parser.parse_args()

DATASETFOLDER = args.dataset_folder
OUTPUTFILE = args.output_file
MINCHARACTERS = args.min_characters
DELETE = args.delete_incomplete_files
FILEFORMATS = ['jpg', 'jpeg', 'png', 'bmp']

filenames_and_folders = os.listdir(DATASETFOLDER)
folders = [x for x in filenames_and_folders if os.path.isdir(Path(DATASETFOLDER + '/' + x)) == True]
files = [x for x in filenames_and_folders if x not in folders]
folders = [''] + folders
faulty_data = {}
    
def return_incomplete_and_paired_ids(files):
    ids = [x[:-4] for x in files]
    d = collections.defaultdict(int)
    for x in ids: d[x] += 1

    incomplete_ids = [x for x in ids if d[x] == 1]
    paired_ids = list(set(ids) - set(incomplete_ids))
    incomplete_files = [x for x in files if x[:-4] in incomplete_ids]

    return {'paired_ids': paired_ids, 'incomplete_files': incomplete_files}

def true_if_image_corrupt_and_fileformat(parent, id):
    for fileformat in FILEFORMATS:
        filepath = Path(parent + '/' + id + '.' + fileformat)
        if os.path.isfile(filepath):
            try:
                img = Image.open(Path(filepath))
                img.verify()
                img.close()
            except (IOError, SyntaxError) as _:
                return True, fileformat
            else:
                return False, fileformat

def return_empty_text_and_corrupt_images(parent, paired_ids):
    empty_texts = []
    corrupt_images = []
    delete = []

    short_text = False

    for id in paired_ids:
        with open(Path(parent + '/' + id + '.txt'), 'r') as f:
            if len(f.read()) < MINCHARACTERS:
                short_text = True

        corrupt, fileformat = true_if_image_corrupt_and_fileformat(parent, id)

        if corrupt:
            corrupt_images.append(id + '.' + fileformat)
        
        if short_text:
            empty_texts.append(id + '.txt')
        
        if corrupt or short_text:
            delete.append(parent + '/' + id + '.txt')
            delete.append(parent + '/' + id + '.' + fileformat)

    return {'empty': empty_texts, 'corrupt': corrupt_images, 'delete': delete}

for folder in folders:
    sep = '/' if folder != '' else ''
    parent = DATASETFOLDER + sep + folder
    files = os.listdir(parent)

    if folder == '':
        files = [x for x in files if '.' in x]
    
    incomplete_and_paired_files_in_parent_folder = return_incomplete_and_paired_ids(files)
    empty_and_corrupt_files_in_parent_folder = \
        return_empty_text_and_corrupt_images(
            parent, 
            incomplete_and_paired_files_in_parent_folder['paired_ids'])

    if len(files) > 0:
        faulty_data[parent] = {
            'incomplete': incomplete_and_paired_files_in_parent_folder['incomplete_files'],
            'corrupt': empty_and_corrupt_files_in_parent_folder['corrupt'],
            'empty': empty_and_corrupt_files_in_parent_folder['empty'],
            'delete': empty_and_corrupt_files_in_parent_folder['delete'] \
                + [parent + '/' + x for x in incomplete_and_paired_files_in_parent_folder['incomplete_files']]
        }

df = pd.DataFrame.from_dict(faulty_data).T
df.index.name = 'folderpath'
df.to_csv(OUTPUTFILE, sep='|')

if DELETE:
    print('Deleting incomplete and corrupt files...')
    delete_lists = list(df['delete'])
    count_deleted_files = 0
    for delete_list in delete_lists:
        count_deleted_files += len(delete_list)
        for delete_file in delete_list:
            if os.path.isfile(Path(delete_file)):
                os.remove(Path(delete_file))
    print('Finished deleting {:,}'.format(count_deleted_files))