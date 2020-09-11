from __future__ import print_function
from __future__ import division

import copy
import glob
import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile

import cv2
import numpy as np
from scipy.ndimage.morphology import binary_dilation,binary_erosion
from skimage import exposure

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import datasets, models, transforms,utils
from torchvision.transforms import functional as func
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

def apply_model_to_img(model,img):
    model = model.cuda().eval()
    seg = F.softmax(model(torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0).cuda()),dim=1)
    seg = seg[0,:,:,:].detach().cpu().numpy()
    return seg

def main(args):
    """
    Main function to parse arguments.
    """
    # Reading command line arguments into parser.
    parser = argparse.ArgumentParser(description = "Prepare CIFAR100 data.")

    # Filepaths
    parser.add_argument("--pData", dest="path_data", type=str, default=None)
    parser.add_argument("--pSave", dest="path_save", type=str, default=None)
    parser.add_argument("--pModel", dest="path_model", type=str, default=None)

    # Creating Parser Object
    opts = parser.parse_args(args[1:])

    if not isdir(opts.path_save):
        mkdir(opts.path_save)
    for ii in range(100):
        path_dir = join(opts.path_save, str(ii))
        if not isdir(path_dir):
            mkdir(path_dir)

    model = densenet101(3, 11, pretrained=False)
    best_state_dict = torch.load(opts.path_model)
    model.load_state_dict(best_state_dict)

    X,Y,_,_,_ = load_CIFAR100_data(path_data=opts.path_data)

    for ind in range(X.shape[0]):
        img = np.array(X[ind,:,:,:]).transpose(1,2,0) #+ 10*np.random.randn(28,28)
        img = cv2.resize(img.astype(np.float32), (128,128))
        img = img.astype(np.float32) / 128.0 - 1.0 #+ np.random.randn(112,112)/10

        seg = apply_model_to_img(model,img)

        seg_bw = (np.argmax(seg,axis=0) != 0)
        seg_bw = cv2.resize(seg_bw.astype(np.float32), (32,32))

        cv2.imwrite(join(path_save_full,str(Y[ind]),str(ind)+'.png'), X[ind,:,:,:].transpose(1,2,0))
        cv2.imwrite(join(path_save_full,str(Y[ind]),str(ind)+'seg.png'), (seg_bw*255).astype(np.uint8))


if __name__ == "__main__":
    main(sys.argv)
