import urllib, os
from tqdm import tqdm
import urllib.request

def download_wit_urls(urlfolder='../wit_urls', onepercentsample=True):
    links = ["https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-0000{}-of-00010.tsv.gz".format(i) for i in range(9)]
    if onepercentsample:
        links = ["https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-1percent_sample.tsv.gz"]
    filenames = [link.split('/')[-1] for link in links]
    os.makedirs(urlfolder, exist_ok=True)

    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            return self.update(b * bsize - self.n)

    for witurl, filename in zip(links, filenames):
        filepath = os.path.join(urlfolder, filename)
        if not os.path.exists(filepath):
            with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                        desc=witurl.split('/')[-1]) as t:  # all optional kwargs
                urllib.request.urlretrieve(witurl, filename=filepath,
                                reporthook=t.update_to, data=None)
                t.total = t.n
        else:
            print('{} already downloaded.'.format(filename))