#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform classes for MRI preprocessing

@author: GM
@edited: CD
"""
from __future__ import print_function, division
import nibabel
from scipy import ndimage
import numpy as np
import torch

class LoadNifti(object):
    def __call__(self,file):
        a = nibabel.load(file) 
        a = nibabel.as_closest_canonical(a) # transform into RAS orientation
        pixdim = a.header.get('pixdim')[1:4]
        a = np.array(a.dataobj)
        a = np.float32(a)
        return {'data':a,'pixdim':pixdim,'affine':[]}

class ReturnImageData(object):
    def __call__(self,image):
        return image['data']

class ToTensor(object):
    """Convert np arrays in sample to Tensors."""
    def __call__(self,image):
        image = torch.from_numpy(image)
        return image

class Crop(object):
    def __init__(self,dims=[128,128,128],offset=[0,0,0],rand_offset=None):
        self.dims=dims
        self.offset = offset
        self.rand_offset = rand_offset
        
    def __call__(self, image):    
        dims_org = image.shape[:3]
        center = np.array([d/2 for d in dims_org]) # center coordinates
        if not (self.rand_offset is None or self.rand_offset==0):
            center += np.random.randint(-self.rand_offset,self.rand_offset,len(center))
        corner = center - np.array([of/2 for of in self.dims])
        corner -= np.array(self.offset)
        corner[corner<1]=0
        c=[int(c) for c in corner]
        image = image[c[0]:c[0]+self.dims[0],
                     c[1]:c[1]+self.dims[1],
                     c[2]:c[2]+self.dims[2]]
        return image

class ReduceSlices(object):
    def __init__(self,x_factor=2,y_factor=2):
        self.x_factor = x_factor
        self.y_factor = y_factor
        
    def __call__(self,image):
        return image[::self.x_factor,::self.y_factor]

class PerImageNormalization(object):
    """ Transforms all pixel values to to have total mean=0 and std=1"""
    def __call__(self, image):
        image -= image.mean()
        image /= image.std()
        return image

class Window(object):
    """ Cap image to be between [low,high]"""
    def __init__(self, low,high):
        self.low = low
        self.high = high
        
    def __call__(self, image):
        image[image<self.low] = self.low
        image[image>self.high] = self.high
        return image

class ComposeMRI(object):
    """ Composes transforms together """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t(input)   
        return input