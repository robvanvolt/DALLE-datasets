import webdataset as wds
import torch

# dataset = wds.WebDataset('dataset.tar.gz').shuffle(8).decode().to_tuple("cap", "img")
dataset = wds.WebDataset('dataset.tar.gz').shuffle(8).decode()

# for d in dataset:
#     input(d)

# for val in dataset:
#     for item in val:
#         print(item)
#         input()

# dataloader = torch.utils.data.DataLoader(dataset)

# for val in dataloader:
#     input(val)

image_files = {d['__key__']: d['img'] for d in dataset}
text_files = {d['__key__']: d['cap'] for d in dataset}

keys = list(image_files.keys() & text_files.keys())

print(keys)

for key in keys:
    input(text_files[key])

# for inp in dataloader:
#     input(inp)
    # for key in inp:
    #     print(inp[key])
    #     input()