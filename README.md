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


## Sanitycheck for downloaded datasets

The following command will look for image-text-pairs (.jpg / .png / .bmp) and return a csv table with incomplete data.
When you add the optional argument -DEL, the incomplete files get deleted. The python scripts checks one folder and the first subdirectories.

```python sanity_check.py --dataset_folder my-dataset-folder```


## Pretrained models

If you want to continue training on pretrained models or even upload your own Dall-E model, head over to https://github.com/robvanvolt/DALLE-models

## Credits

A lot of inspiration was taken from https://github.com/yashbonde/dall-e-baby - unfortunately that repo does not get updated anymore...
