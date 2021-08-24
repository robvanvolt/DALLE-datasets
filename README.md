## DALLE-datasets
This is a summary of easily available, high-quality datasets consisiting of captioned image files for generalized DALLE-pytorch training (https://github.com/lucidrains/DALLE-pytorch).

The scripts help you download and resize the files from the given sources.

* general datasets
  * Conceptual Images 12m
  * Wikipedia
  * Filtered yfcc100m
  * Open Images
* specific datasets
  * None yet


## Helper scripts

All helper scripts can be found in the utilities folder now:
* TFrecords to WebDataset converter
* Image-Text-Folder to WebDataset converter
* Dataset sanitycheck for image-text-files
* Example reader for WebDataset files


### Sanitycheck for downloaded datasets

The following command will look for image-text-pairs (.jpg / .png / .bmp) and return a csv table with incomplete data.
When you add the optional argument -DEL, the incomplete files get deleted. The python scripts checks one folder and the first subdirectories.

```python sanity_check.py --dataset_folder my-dataset-folder```


## Pretrained models

If you want to continue training on pretrained models or even upload your own Dall-E model, head over to https://github.com/robvanvolt/DALLE-models

## Credits

Special thanks go to <a href="https://github.com/rom1504">Romaine</a>, who improved the download scripts and made the great WebDataset format more accessible with his continuous coding efforts! 🙏 

A lot of inspiration was taken from https://github.com/yashbonde/dall-e-baby - unfortunately that repo does not get updated anymore...
Also, the shard creator was inspired by https://github.com/tmbdev-archive/webdataset-examples/blob/master/makeshards.py.
The custom tokenizer was inspired by afiaka87, who showed a simple way to generate custom tokenizers with youtokentome.
