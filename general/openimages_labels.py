
import os
import re
import h5py
import json
from tqdm import trange
import numpy as np
import pandas as pd
from tabulate import tabulate

#############################################################################################
###### ATTENTION ############################################################################
###### You need to download class-descriptions-boxable.csv from the following website #######
###### https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv ##########
#############################################################################################

def get_open_images_label_names():
  with open("./downsampled-open-images-v4/class-descriptions-boxable.csv", "r") as f:
    open_image_labels = {x.split(",")[0]: x.split(",")[1] for x in f.read().split("\n") if len(x)}
  return open_image_labels

def get_open_images_labels(annotations_path):
  open_image_labels = get_open_images_label_names()
  df = pd.read_csv(annotations_path)
  image_to_labels = {}
  dropped = []
  pbar = trange(len(df.ImageID.unique()))
  path_f = "./downsampled-open-images-v4/256px/"
  if "validation" in annotations_path:
    path_f += "validation/"
  elif "train" in annotations_path:
    path_f += "train-256/"
  elif "test" in annotations_path:
    path_f += "test/"
  for _, (img_id, df_sub) in zip(pbar, df.groupby("ImageID")):
    path = f"{path_f}{img_id}.jpg"
    pbar.set_description(f"Loading {path[::-1][:40][::-1]}")
    high_conf = df_sub[df_sub.Confidence == 1].LabelName.values.tolist()
    low_conf = df_sub[df_sub.Confidence != 1].LabelName.values.tolist()
    if not high_conf or not os.path.exists(path):
      dropped.append(img_id)
    image_to_labels["open_images_" + img_id] = {
      "label": [
          [open_image_labels[x] for x in high_conf],
          [open_image_labels[x] for x in low_conf]
      ],
      "path": path
    }
  return image_to_labels, dropped

# ---- Captions are generated using CaptionsGenerator

class CaptionGenerator():
  templates_labels = [
    "a picture of {}",
    "a photo that has {}",
    "photo consisting of {}",
    "a low resolution photo of {}",
    "small photo of {}",
    "high resolution picture of {}",
    "low resolution picture of {}",
    "high res photo that has {}",
    "low res photo of {}",
    "{} in a photo",
    "{} in a picture",
    "rendered picture of {}",
    "jpeg photo of {}",
    "a cool photo of {}",
    "{} rendered in a picture",
  ]

  templates_maybe = [
    *[x + " and maybe containing {}" for x in templates_labels],
    *[x + " and possibly containing {}" for x in templates_labels],
    *[x + " and {} but not sure" for x in templates_labels],
    *[x + " also roughly {}" for x in templates_labels],
  ]

  captions_templates = {
    "open_images": [templates_labels, templates_maybe],
  }
  
  def __init__(self):
    self.ds_names = list(self.captions_templates.keys())

  def generate_open_images_caption(self, ds):
    temps_high, temps_low = self.captions_templates["open_images"]
    captions = {}
    for i,k in enumerate(ds):
      high_conf = ", ".join(ds[k]["label"][0])
      if np.random.random() > 0.5:
        low_conf = ", ".join(ds[k]["label"][1])
        temp = np.random.choice(temps_low, size=1)[0]
        cap = temp.format(high_conf, low_conf)
      else:
        temp = np.random.choice(temps_high, size = 1)[0]
        cap = temp.format(high_conf)
      cap = re.sub(r"\s+", " ", cap).strip().lower()
      captions["open_images_" + str(k)] = {
          "path": ds[k]["path"],
          "caption": cap
      }
    return captions
  
  def generate_captions(self, ds, ds_name):
    print("Generating captions for", ds_name)
    if ds_name not in self.ds_names:
      raise ValueError(f"{ds_name} not in {self.ds_names}")

    if ds_name == "open_images":
      return self.generate_open_images_caption(ds)

    temps = []
    for temp in self.captions_templates[ds_name]:
      temps.extend(temp)
    
    # each ds: {<id>: {"path": <path>, "label": [<label(s)>]}}
    captions = {}
    temps_ordered = np.random.randint(low = 0, high = len(temps), size = (len(ds)))
    for i,k in enumerate(ds):
      lbs_string = ", ".join(ds[k]["label"])
      cap = temps[temps_ordered[i]].format(lbs_string)
      cap = re.sub(r"\s+", " ", cap).strip().lower()
      captions[ds_name + "_" + str(k)] = {
        "path": ds[k]["path"],
        "caption": cap
      }
    return captions


# ---- Script
if __name__ == "__main__":
  print("-"*70 + "\n:: Loading OpenImages Dataset")
  open_images_img2lab_val, oi_dropped_val = get_open_images_labels(
    "./downsampled-open-images-v4/validation-annotations-human-imagelabels-boxable.csv"
  )
  open_images_img2lab_train, oi_dropped_train = get_open_images_labels(
      "./downsampled-open-images-v4/train-annotations-human-imagelabels-boxable.csv"
  )
  open_images_img2lab_test, oi_dropped_test = get_open_images_labels(
      "./downsampled-open-images-v4/test-annotations-human-imagelabels-boxable.csv"
  )

  # define table for tabulate
  headers = ["name", "num_samples", "dropped"]
  table = [
    ["open images (train)", len(open_images_img2lab_train), len(oi_dropped_train)],
    ["open images (val)", len(open_images_img2lab_val), len(oi_dropped_val)],
    ["open images (test)", len(open_images_img2lab_test), len(oi_dropped_test)],
  ]
  table_arr = np.asarray(table)
  total_samples = sum([
    len(open_images_img2lab_train),
    len(open_images_img2lab_val),
    len(open_images_img2lab_test),
  ])
  total_dropped = sum([
    len(oi_dropped_train),
    len(oi_dropped_val),
    len(oi_dropped_test),
  ])
  table.append(["total", total_samples, total_dropped])
  print("\n", "-"*70, "\n")
  print(tabulate(table, headers, tablefmt="psql"))

  print("\n:: Generating captions for labels")

  capgen = CaptionGenerator()
  capgen_oi_val   = capgen.generate_captions(open_images_img2lab_val, "open_images")
  capgen_oi_train = capgen.generate_captions(open_images_img2lab_train, "open_images")
  capgen_oi_test  = capgen.generate_captions(open_images_img2lab_test, "open_images")

  # make the master captions list
  common_captions = {}
  common_captions.update(capgen_oi_val)
  common_captions.update(capgen_oi_train)
  common_captions.update(capgen_oi_test)

  print(len(common_captions), table[-1][1])
  with open("captions_train.json", "w") as f:
    f.write(json.dumps(common_captions))