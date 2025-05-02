#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform loading utility

@author: GM
@edited: CD
"""
from .transforms import (
    LoadNifti, ReturnImageData, ToTensor, Crop, 
    ReduceSlices, PerImageNormalization, Window, ComposeMRI
)

def load_transforms(args, random_chance=0):
    """Load image transforms for transfer learning
    
    Args:
        args: Dictionary containing parameters including:
            - img_dim: Target image dimensions [h,w,d]
        random_chance: Whether to use random augmentations (0 for inference)
    """
    # During inference, we don't want random offsets
    s = 1 if random_chance > 0 else 0
    
    transforms = ComposeMRI([
        LoadNifti(),
        ReturnImageData(),
        Crop(dims=args['img_dim'], offset=[0,0,0], rand_offset=5*s),
        ReduceSlices(2, 2),
        PerImageNormalization(),
        Window(-3, 3),
        ToTensor(),
    ])

    return transforms