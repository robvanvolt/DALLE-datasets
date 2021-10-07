# pip install git+https://github.com/openai/CLIP.git
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import clip
import torch
import torch.nn as nn
from PIL import Image

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


### TODO: clip image / text
### TODO: WDS support output
### TODO: Multiprocessing
### TODO: GPU support

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

DATAFOLDER = 'data'

path = Path(DATAFOLDER)

text_files = [*path.glob('**/*.txt')]
text_files = {text_file.stem: text_file for text_file in text_files} # str(text_file.parents[0]) + 
text_total = len(text_files)

image_files = [
    *path.glob('**/*.png'), *path.glob('**/*.jpg'),
    *path.glob('**/*.jpeg'), *path.glob('**/*.bmp'),
    *path.glob('**/*.PNG'), *path.glob('**/*.JPG'),
    *path.glob('**/*.JPEG'), *path.glob('**/*.BMP')
]
image_files = {image_file.stem: image_file for image_file in image_files} # str(image_file.parents[0]) +
image_total = len(image_files)

print('Found {:,} textfiles and {:,} images.'.format(text_total, image_total))

keys = (image_files.keys() & text_files.keys())

for key in keys:
    print(text_files[key])
    cap = ''
    with open(text_files[key], 'r') as t:
        for line in t:
            cap += line + ' '

    image = preprocess(Image.open(image_files[key])).unsqueeze(0).to(device)
    text = clip.tokenize(cap.split(' ')).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text) # self.model.encode_text(text_tokens.to(device)).float()
        
        logits_per_image, logits_per_text = model(image, text)

        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
        # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # print(similarity)

        # cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        # image_features = model.encode_image(image.to(device)).float()
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # text_features = model.encode_text(text.to(device)).float()
        # similarity = cosine_similarity(image_features, text_features).tolist()

        # print(float(cosine_similarity(text_features , image_features)))
        # print(float(cosine_similarity(torch.reshape(text_features, (1, 512)) , image_features)))
        # print(max(similarity))

        print("Label probs:", probs) 
        print('Cossimilarity > 0.3?')
        print(max(probs[0]))

    input()
    img = mpimg.imread(image_files[key])
    imgplot = plt.imshow(img)
    plt.title(cap)
    plt.show()

