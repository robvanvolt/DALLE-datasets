from pathlib import Path

import tensorflow as tf


FEATURE_DESCRIPTION = {
    'sampleID': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
    'format': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
}
'''
"sampleID": bytes_feature(sampleID),
"image": bytes_feature(image_data),
"format": bytes_feature(image_format),
"label": bytes_feature(caption),#int64_feature(class_id),
"height": int64_feature(height),
"width": int64_feature(width),
'''
    
def _parse_example(example_proto):
    """Return the serialized `example_proto` as a tuple of the image and
    its label.
    """
    example = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    print(example)
    image = tf.io.decode_jpeg(example['image'])
    print(example['label'])
    return image, example['label'], example['sampleID'] , example['width'] ,example['height'] 


def from_tfrecords(path):
    """Return the TFRecords in `path` as a `tf.data.Dataset`."""
    if isinstance(path, Path):
        path = str(path.expanduser())

    raw_dataset = tf.data.TFRecordDataset([path])
    return raw_dataset.map(_parse_example)


dataset = from_tfrecords("/content/crawling_at_home_FIRST_SAMPLE_ID_IN_SHARD_3795000001_LAST_SAMPLE_ID_IN_SHARD_3796000000_1__00000-of-00001.tfrecord")
print(dataset)

def wrapper(gen):
  while True:
    try:
      yield next(gen)
    except StopIteration:
      break
    except Exception as e:
      print(e) # or whatever kind of logging you want



c= 0
for ex in wrapper(dataset.as_numpy_iterator() ):
    print(c)
    
    print(ex[1])
    c +=1