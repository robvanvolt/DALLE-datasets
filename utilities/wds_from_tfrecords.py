import tensorflow as tf
import webdataset as wds
from pathlib import Path
import argparse
import os
import timeit
import hashlib

parser = argparse.ArgumentParser("""Generate sharded dataset from tfrecord-files.""")
parser.add_argument("--maxsize", type=float, default=1e9)
parser.add_argument("--maxcount", type=float, default=100000)
parser.add_argument(
    "--compression", 
    dest="compression", 
    action="store_true",
    help="Creates compressed .tar.gz files instead of uncompressed .tar files."
    )
parser.add_argument(
    "--use_encoder", 
    dest="use_encoder", 
    action="store_true",
    help="Uses encoder on unknown filetimes (the suffix in the keep_keys argument)."
    )
parser.add_argument(
    "--keep_keys", 
    type=str, 
    default="image.pyd,label.cls",
    help="Only keep the columns from the comma separated keys from that argument. The dot separated suffix is the filetype."
    )
parser.add_argument(
    "--remove_duplicates",
    dest="remove_duplicates",
    default="",
    help="Remove duplicates from given column name. (e.g. --remove_duplicates image)"
    )
parser.add_argument(
    "--report_every", 
    type=int, 
    default="1000",
    help="Report every n iterations."
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
    default="./tfr",
    help="directory path containing tfrecord files",
)
args = parser.parse_args()

KEEP_KEYS = []
if args.keep_keys != '':
  KEEP_KEYS = {x.split('.')[0]: x.split('.')[1] for x in args.keep_keys.split(',')}

assert args.maxsize > 10000000
assert args.maxcount < 1000000
assert os.path.isdir(os.path.join(args.data)), '{} does not exist.'.format(args.data)

os.makedirs(Path(args.shards), exist_ok=True)

tfrecord_files = [args.data + '/' + x for x in os.listdir(args.data) if x.split('.')[-1] == 'tfrecord']
total_files = len(tfrecord_files)

###### Example of a feature description to a tfrecord dataset
FEATURE_DESCRIPTION = {
  ###### Please provide your tfrecord feature description
}
# FEATURE_DESCRIPTION = {
#     'sampleID': tf.io.FixedLenFeature([], tf.string),
#     'image': tf.io.FixedLenFeature([], tf.string),
#     'format': tf.io.FixedLenFeature([], tf.string),
#     'label': tf.io.FixedLenFeature([], tf.string),
#     'height': tf.io.FixedLenFeature([], tf.int64),
#     'width': tf.io.FixedLenFeature([], tf.int64),
# }

assert len(FEATURE_DESCRIPTION) > 0, 'Please provide the feature description to your tfrecord dataset.'

def wrapper(gen):
  while True:
    try:
      yield next(gen)
    except StopIteration:
      break
    except Exception as e:
      print(e)

def _parse_example(example_proto):
    example = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    return example

pattern = os.path.join(args.shards, args.shard_prefix + f"%06d.tar" + (".gz" if args.compression else ''))
count = 0

duplicate_count = 0
duplicate_md5 = set()
skip = False

start = timeit.default_timer()
with wds.ShardWriter(pattern, maxsize=int(args.maxsize), maxcount=int(args.maxcount), encoder=args.use_encoder) as sink:
  for tfrecord_file in tfrecord_files:
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = raw_dataset.map(_parse_example)
    for item in wrapper(dataset.as_numpy_iterator()):
        ds_key = "%09d" % count
        sample = {
            "__key__": ds_key,
        }
        if args.remove_duplicates != '':
          valuehash = hashlib.md5(item[args.remove_duplicates]).hexdigest()
          if valuehash in duplicate_md5:
            duplicate_count += 1
            skip = True
          else:
            duplicate_md5.add(valuehash)
        if not skip:
          for key in KEEP_KEYS:
              sample[key + '.' + KEEP_KEYS[key] if args.use_encoder else key] = item[key]
          sink.write(sample)
        else:
          skip = False
        if count % args.report_every == 0:
          print('   {:.2f}'.format(count), end='\r')
        count += 1
stop = timeit.default_timer()

print('###################################################################')  
print('Finished processing {:,} samples from tfrecord files.'.format(count))
print('Process took {:.2f} seconds to finish.'.format(stop - start))
if (args.remove_duplicates != ''):
  print('Skipped {} duplicates from a total of {} items.'.format(duplicate_count, count))
print('The WebDataset files can be found in {}.'.format(args.shards))
print('###################################################################')