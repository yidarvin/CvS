from __future__ import print_function
from __future__ import division

import copy
import glob
import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile
import time

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

def take_one_step(model,X,Y,criterion,phase=None,alpha=1):
    alpha = np.clip(alpha,0,1)
    with torch.set_grad_enabled(phase == 'train'):
        output = model(X)
        loss = alpha*criterion(output,Y)
        loss += (1-alpha)*criterion(output,torch.argmax(output,dim=1))
        Y_copy = Y.detach().cpu().numpy()
        weight = np.ones_like(Y_copy)
        for ii in range(Y.size(0)):
            Y_slice = Y_copy[ii,:,:]
            boundary = Y_slice > 0
            boundary = binary_dilation(boundary, iterations=2)
            weight[ii,:,:] += boundary * 0
        loss = torch.mean(loss * torch.from_numpy(weight).float().cuda())
    pred = torch.sum(output,dim=(2,3))
    pred = torch.argmax(pred[:,1:],dim=1) + 1
    gt,_ = torch.max(Y.float().view(Y.shape[0],-1),dim=1)
    acc = torch.mean((pred == gt).float())
    return acc,loss

def trainer_CvS(model, dataloaders, path_save=None, name_exp='experiment', learning_rate=1e-4, num_epochs=25, verbose=True):
    since = time.time()
    criterion = nn.CrossEntropyLoss(reduction='none')
    model = nn.DataParallel(model.cuda())
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,5,eta_min=1e-10)
    best_acc = 0.0

    for epoch in range(num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
        for phase in dataloaders.keys():
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_acc = 0.0
            running_loss = 0.0

            #Iterate over data.
            optimizer.zero_grad()
            for sample_batch in dataloaders[phase]:
                X = sample_batch['X'].cuda()
                Y = sample_batch['Y'].cuda()
                acc,loss = take_one_step(model,X,Y,criterion,phase=phase)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                running_acc += acc.item() * X.size(0)
                running_loss += loss.item() * X.size(0)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_acc / len(dataloaders[phase].dataset)

            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'train':
                scheduler.step()
            if phase == 'val':
                if epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    if path_save != None:
                        torch.save(model.state_dict(), join(path_save, name_exp + '_best.pth'))
        print()

    if path_save != None:
        torch.save(model.state_dict(), join(path_save, name_exp + '_late.pth'))
    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    return model
