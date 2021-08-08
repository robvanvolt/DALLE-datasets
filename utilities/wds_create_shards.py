import sys
import os
import os.path
import random
import argparse

from pathlib import Path
from PIL import Image

import webdataset as wds

parser = argparse.ArgumentParser("""Generate sharded dataset from image-text-datasets.""")
parser.add_argument("--maxsize", type=float, default=1e9)
parser.add_argument("--maxcount", type=float, default=100000)
parser.add_argument(
    "--compression", 
    dest="compression", 
    action="store_true",
    help="Creates compressed .tar.gz files instead of uncompressed .tar files."
    )
parser.add_argument(
    "--image_text_keys", 
    type=str, 
    default="img,cap",
    help="Comma separated WebDataset dictionary keys for images (first argument) and texts (second argument). \
          The exact argument has to be provided to train_dalle.py, e.g. python train_dalle.py --wds img,cp --image_text_folder ../shards"
    )
parser.add_argument(
    "--shards", 
    default="./shards", 
    help="directory where shards are written"
)
parser.add_argument(
    "--shard_prefix", 
    default="ds_", 
    help="prefix of shards' filenames created in the shards-folder"
)
parser.add_argument(
    "--data",
    default="./data",
    help="directory path containing data suitable for DALLE-pytorch training",
)
args = parser.parse_args()

assert len(args.image_text_keys.split(',')) == 2, 'Too many arguments provided'
assert args.maxsize > 10000000
assert args.maxcount < 1000000

image_key, caption_key = tuple(args.image_text_keys.split(','))

if not os.path.isdir(os.path.join(args.data)):
    print(f"{args.data}: should be directory containing image-text pairs", file=sys.stderr)
    print(f"or subfolders containing image-text-pairs", file=sys.stderr)
    sys.exit(1)

os.makedirs(Path(args.shards), exist_ok=True)

def readfile(fname):
    "Read a binary file from disk."
    with open(fname, "rb") as stream:
        return stream.read()

path = Path(args.data)
text_files = [*path.glob('**/*.txt', recursive=True)]
text_files = {text_file.stem: text_file for text_file in text_files} # str(text_file.parents[0]) + 
text_total = len(text_files)

image_files = [
    *path.glob('**/*.png', recursive=True), *path.glob('**/*.jpg', recursive=True),
    *path.glob('**/*.jpeg', recursive=True), *path.glob('**/*.bmp', recursive=True)
]
image_files = {image_file.stem: image_file for image_file in image_files} # str(image_file.parents[0]) +
image_total = len(image_files)

print('Found {:,} textfiles and {:,} images.'.format(text_total, image_total))

keys = (image_files.keys() & text_files.keys())

text_files = {k: v for k, v in text_files.items() if k in keys}
image_files = {k: v for k, v in image_files.items() if k in keys}

for key in image_files:
    img = Image.open(image_files[key])
    try:
        img.verify()
    except Exception:
        print('Invalid image on path {}'.format(key))
        keys.remove(key)

print("Remaining keys after image sanity check: {:,}".format(len(keys)))

total_pairs = len(keys)
keys = list(keys)

indexes = list(range(total_pairs))
random.shuffle(indexes)

# This is the output pattern under which we write shards.
pattern = os.path.join(args.shards, args.shard_prefix + f"%06d.tar" + (".gz" if args.compression else ''))

with wds.ShardWriter(pattern, maxsize=int(args.maxsize), maxcount=int(args.maxcount)) as sink:
    for i in indexes:
        with open(image_files[keys[i]], "rb") as imgstream:
            image = imgstream.read()
        with open(text_files[keys[i]], "rb") as txtstream:
            text = txtstream.read()

        ds_key = "%09d" % i

        sample = {
            "__key__": ds_key,
            image_key: image,
            caption_key: text
        }
        sink.write(sample)
