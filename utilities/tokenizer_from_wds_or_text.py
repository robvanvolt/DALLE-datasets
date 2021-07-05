import youtokentome as yttm
import webdataset as wds
from pathlib import Path
import argparse
import html
import os

# yttm bpe --vocab_size=4096 --coverage=1.0 --model=blogoimixer_4096.bpe --data=blogoi_allcaps.txt
parser = argparse.ArgumentParser("""Generate a custom tokenizer for your WebDataset files.""")

parser.add_argument(
    "--source", 
    type=str, 
    default="./shards",
    help="Specify the vocab size you want to use for your custom tokenizer."
    )
parser.add_argument(
    "--text_key", 
    type=str, 
    default="label",
    help="Specify the text column in your WebDataset file(s)."    
)
parser.add_argument(
    "--output_model_name", 
    type=str, 
    default="custom_tokenizer.bpe",
    help="Specify the output file of your tokenizer."
    )
parser.add_argument(
    "--coverage", 
    type=float, 
    default=0.9999,
    help="Specify the coverage for your custom tokenizer."
    )
parser.add_argument(
    "--output_folder", 
    type=str, 
    default='./output',
    help="Specify the output folder for the generated files."
    )
parser.add_argument(
    "--vocab_size", 
    type=int, 
    default=4096,
    help="Specify the vocab size for your custom tokenizer."
    )
args = parser.parse_args()

os.makedirs(args.output_folder, exist_ok=True)

path_to_textfile = args.output_folder + "/text_for_tokenizer.txt"

if args.source[-4:].lower() == '.txt':
    print('--------------------------------------------------------')
    print('----> Creating custom tokenizer from provided text file.')
    print('--------------------------------------------------------')
    path_to_textfile = args.source
else:
    assert os.path.isdir(args.source), 'The source path has to be a directory containing text files.'
    print('---------------------------------------------------------------')
    print('----> Generating a singe text file for WebDataset folder first.')
    print('---------------------------------------------------------------')
    path = Path(args.source)

    wds_files = image_files = [
        *path.glob('**/*.tar'), *path.glob('**/*.tar.gz')
    ]

    assert len(wds_files) > 0, 'No WebDataset files (.tar/.tar.gz) found in {} found'.format(args.source)

    wds_files = [str(x) for x in wds_files]

    dataset = wds.WebDataset(wds_files)

    c = 0

    with open(path_to_textfile, "w") as f:
        for item in dataset:
            f.write(html.unescape(item[args.text_key].decode('utf-8')))
            if c % 10000 == 0:
                print('   {:.2f}'.format(c), end='\r')
            c += 1

yttm.BPE.train(
    data=path_to_textfile, 
    model=args.output_folder + '/' + args.output_model_name, 
    vocab_size=args.vocab_size, 
    coverage=args.coverage, # or 1.0
    n_threads=-1, 
)