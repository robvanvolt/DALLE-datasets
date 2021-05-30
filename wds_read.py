import webdataset as wds
from PIL import Image
import io
import matplotlib.pyplot as plt

dataset = wds.WebDataset("dataset.tar.gz")

for i, d in enumerate(dataset):
    print(i)
    print(d['__key__'])
    print(d['cap'].decode('utf-8'))
    image = Image.open(io.BytesIO(d['img']))
    plt.imshow(image)
    plt.show()
    input()