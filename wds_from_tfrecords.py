import torch
from tfrecord.torch.dataset import MultiTFRecordDataset
from tfrecord.tools.tfrecord2idx import create_index
import tensorflow as tf
import webdataset as wds
from pathlib import Path
import argparse
import os

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
    "--keep_keys", 
    type=str, 
    default="",
    help="Only keep the columns from the comma separated keys from that argument."
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
  KEEP_KEYS = args.keep_keys.split(',')

assert args.maxsize > 10000000
assert args.maxcount < 1000000
assert os.path.isdir(os.path.join(args.data)), '{} does not exist.'.format(args.data)

os.makedirs(Path(args.shards), exist_ok=True)

index_path = args.data
tfrecord_pattern = args.data + '/{}.tfrecord'
index_pattern = index_path + '/{}.index'

os.makedirs(index_path, exist_ok=True)

tfrecord_files = [x[:-9] for x in os.listdir(args.data) if x.split('.')[-1] == 'tfrecord']
total_files = len(tfrecord_files)
splits = {k: 1/total_files for k in tfrecord_files}

tfrecord_index_files = [x[:-6] for x in os.listdir(index_path) if x.split('.')[-1] == 'index']
total_index_files = len(tfrecord_index_files)

TFR_MATCH_INDEX = True if len([x for x in tfrecord_files if x not in tfrecord_index_files]) == 0 else False

if not TFR_MATCH_INDEX:
  print('Index files must be provided when using multiple workers, otherwise the loader may return duplicate records.')
  print('Generating index files in {}...'.format(index_path))
  for tfrecord_file in tfrecord_files:
    create_index(args.data + '/' + tfrecord_file + '.tfrecord', index_path + '/' + tfrecord_file + '.index')
  print('Finished generating index files!')
else:
  print('Found matching number of index and tfrecord files.')


raw_dataset = tf.data.TFRecordDataset(args.data + '/' + [x for x in os.listdir(args.data) if x.split('.')[-1] == 'tfrecord'][0])
keys = {}
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    for key, value in example.features.feature.items():
      keys[key] = True if value.WhichOneof('kind') == 'bytes_list' else False

if len(KEEP_KEYS) > 0:
  keys = {k: v for k, v in keys.items() if k in KEEP_KEYS}
  assert len(keys.items()) > 0, 'No keys left to convert to WebDataset.'


def _parse_example(example_proto):
    """Return the example_proto as a tuple of the image and its label."""
    return {key: example_proto[key].tobytes() for key in keys}
    # return {key: example_proto[key].tobytes() if keys[key] else example_proto[key] for key in keys}

def _collate_fn(batch):
    return batch[0]

dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits, transform=_parse_example, infinite=False)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=_collate_fn, drop_last=False)

# This is the output pattern under which we write shards.
pattern = os.path.join(args.shards, args.shard_prefix + f"%06d.tar" + (".gz" if args.compression else ''))
count = 0

with wds.ShardWriter(pattern, maxsize=int(args.maxsize), maxcount=int(args.maxcount)) as sink:
    for i, item in enumerate(iter(loader)):
        count = i
        ds_key = "%09d" % i
        sample = {
            "__key__": ds_key,
        }
        for key in keys:
          sample[key] = item[key]
        sink.write(sample)
        if count % args.report_every == 0:
          print('   {:,}'.format(count), end='\r')

print('Finished converting samples {:,} from tfrecord files to webdataset.'.format(count))
