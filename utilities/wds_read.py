import webdataset as wds
from PIL import Image
import io
import matplotlib.pyplot as plt

# dataset = wds.WebDataset("dataset.tar.gz")
dataset = wds.WebDataset("./shards/ds_000000.tar")

for i, d in enumerate(dataset):
    # print(i)
    # print(d)
    # input()

    print(d.keys())

    # print(d['__key__'])
    # print(d['cap'])
    # if 'json' in d.keys():
    #     print(d['json'])

    # print(d['label'].decode('utf-8'))
    # image = Image.open(io.BytesIO(d['image']))
    # plt.show()
    # plt.imshow(image)
    # plt.show()
    input()