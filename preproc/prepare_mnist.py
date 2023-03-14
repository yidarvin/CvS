from __future__ import print_function
from __future__ import division

import argparse
import shutil
import sys
import wget
import gzip

import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile


def main(args):
    """
    Main function to parse arguments.
    """
    # Reading command line arguments into parser.
    parser = argparse.ArgumentParser(description = "Prepare MNIST data.")

    # Filepaths
    parser.add_argument("--pData", dest="path_data", type=str, default=None)

    # Creating Parser Object
    opts = parser.parse_args(args[1:])

    if not isdir(opts.path_data):
        mkdir(opts.path_data)

    urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    for url in urls:
        print(url)
        print(opts.path_data)
        wget.download(url, opts.path_data)
    filenames = ['train-images-idx3-ubyte.gz',
                 'train-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz',
                 't10k-labels-idx1-ubyte.gz']
    for filename in filenames:
        with gzip.open(join(opts.path_data, filename),'rb') as f_in:
            with open(join(opts.path_data, filename[:-3]),'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


if __name__ == "__main__":
    main(sys.argv)
