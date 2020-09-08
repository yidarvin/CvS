from __future__ import print_function
from __future__ import division

import argparse
import shutil
import sys
import wget
import tarfile
import zipfile

import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile


def main(args):
    """
    Main function to parse arguments.
    """
    # Reading command line arguments into parser.
    parser = argparse.ArgumentParser(description = "Prepare CIFAR10 data.")

    # Filepaths
    parser.add_argument("--pData", dest="path_data", type=str, default=None)

    # Creating Parser Object
    opts = parser.parse_args(args[1:])

    urls = [('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 'cifar-10-python.tar.gz'),
            ('https://uofi.box.com/shared/static/8sw0gj6d35zgw1z0isi6jy816r6x0g95.zip', 'cifar-10-smalldata-manualseg.zip')]
    #for url,filename in urls:
    #    wget.download(url, join(opts.path_data,filename))
    for url,filename in urls:
        if filename[-1] == 'z':
            with tarfile.open(join(opts.path_data, filename),'r:gz') as f:
                f.extractall(opts.path_data)
        else:
            with zipfile.ZipFile(join(opts.path_data, filename),'r') as f:
                f.extractall(opts.path_data)


if __name__ == "__main__":
    main(sys.argv)
