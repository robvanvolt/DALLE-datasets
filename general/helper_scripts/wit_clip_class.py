import os
import clip
import torch
from PIL import Image
from multiprocessing import cpu_count
from multiprocessing.queues import JoinableQueue
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
use_jit = False # torch.cuda.is_available()

class CLIP:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=use_jit)
        self.tokenizer = clip.tokenize

    def return_similarities(self, image, captions, image_url):
        if '.svg' in image_url:
            svgname = image_url.split('/')[-1]
            pngname = svgname[:-4] + '.png'
            with open(svgname, 'wb') as f:
                f.write(image.content)
            svg_image = svg2rlg(svgname)
            renderPM.drawToFile(svg_image, pngname, fmt="PNG")
            openedImage = Image.open(pngname)
            image_tokens = self.preprocess(openedImage).unsqueeze(0).to(device)
            os.remove(svgname)
            os.remove(pngname)
        else:
            openedImage = Image.open(image.raw)
            image_tokens = self.preprocess(openedImage).unsqueeze(0).to(device)
        openedImage.close()
        logits = []
        for caption in captions:
            text_tokens = self.tokenizer(caption, context_length=77, truncate=True).to(device)
            with torch.no_grad():
                logits_per_image, _ = self.model(image_tokens, text_tokens)
                logits.append(list(torch.flatten(logits_per_image))[0].item())
        return logits, image_tokens