# -*- coding: utf-8 -*-
"""
@author: Guilherme

Módulo gerador de imagens comprimidas com fator aleatório utilizando
transformada discreta dos cossenos.

Baseado em
    
HOSSAIN et al. Distortion Robust Image Classification using Deep
Convolutional Neural Network with Discrete Cosine Transform. 2018.

"""
import torch
import numpy as np
from skimage.color import rgb2ycbcr, ycbcr2rgb
from scipy.fftpack import dct, idct

class dct_transform(object):
    """Discrete Cossine Transform.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
    
        def dct2(img):
            return dct( dct( img, axis=0, norm='ortho' ), axis=1, norm='ortho' )
    
        def idct2(img):
            return idct( idct( img, axis=0, norm='ortho'), axis=1 , norm='ortho')

        img_ycbcr = rgb2ycbcr(img)      # convert to ycbcr color space
        img_dct = np.zeros(img_ycbcr.shape) # create dct vector
        
        for c in range(img_ycbcr.shape[-1]):
            img_dct[:,:,c] = dct2(img_ycbcr[:,:,c])   # Dct coefficients
            t = np.random.randint(0, 51)              # Random threshold [0,50]
            img_dct[abs(img_dct) < t] = 0             # 
            img_ycbcr[:,:,c] = idct2(img_dct[:,:,c])  # Inverse dct
        
        img = torch.tensor(ycbcr2rgb(img_ycbcr)) # convert to rgb
        
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'

def dct_t(images, threshold=None):
    
    def dct2(img):
        return dct( dct( img, axis=0, norm='ortho' ), axis=1, norm='ortho' )

    def idct2(img):
        return idct( idct( img, axis=0, norm='ortho'), axis=1 , norm='ortho')

    for i, (img_rgb) in enumerate(images):

        img_ycbcr = rgb2ycbcr(img_rgb)      # convert to ycbcr color space
        img_dct = np.zeros(img_ycbcr.shape) # create dct vector
        
        for c in range(img_ycbcr.shape[-1]):
            img_dct[:,:,c] = dct2(img_ycbcr[:,:,c])   # Dct coefficients
            if threshold:
                t = threshold
            else:
                t = np.random.randint(0, 51)              # Random threshold [0,50]
            img_dct[abs(img_dct) < t] = 0             # 
            img_ycbcr[:,:,c] = idct2(img_dct[:,:,c])  # Inverse dct
        
        images[i] = torch.tensor(ycbcr2rgb(img_ycbcr)) # convert to rgb
    
    return images