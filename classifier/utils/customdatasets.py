# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:27:40 2019

@author: Guilherme
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset

class CoffeeLeavesDataset(Dataset):
    """Coffee Leaves Dataset."""
        
    def __init__(self, csv_file, images_dir, dataset, fold=1, select_dataset=0, transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            images_dir (string): Directory with all the images.
            dataset (string) : Select the desired dataset - 'train', 'val' or 'test'
            transforms : Image transformations
            fold (int{1,5}) : The data is changed based on the selected fold
        """        
        self.fold = fold
        self.data = self.split_dataset(pd.read_csv(csv_file), dataset)
        self.images_dir = images_dir
        self.select_dataset = select_dataset
        self.transformations = transforms
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        
        # Get image name from the pandas df
        img_name = os.path.join(self.images_dir,
                                str(self.data.iloc[idx, 0]) + '.jpg')
        # Open image
        image = Image.open(img_name)        
        
        # Apply transformations
        if self.transformations:
            image = self.transformations(image)

         # Get label of the image
        label_dis = self.data.iloc[idx, 1]
        label_dis = torch.tensor(label_dis, dtype=torch.long)
    
        label_sev = self.data.iloc[idx, -1]
        label_sev =  torch.tensor(label_sev, dtype=torch.long)
        
        # Multitask
        if self.select_dataset == 0:
            return (image, label_dis, label_sev)
        # Disease
        elif self.select_dataset == 1: 
            return (image, label_dis)
        # Severity
        else: 
            return (image, label_sev)
        
    def split_dataset(self, csv, dataset):        
        seed = 150
        np.random.seed(seed)
        
        dataset_size = len(csv)
        partition_size = int(dataset_size / 5)
        indices = list(range(dataset_size))
        
        # Split: 70% for training, 15% for validation and 15% for test
        p1 = int(np.ceil( 0.7 * dataset_size ))
        p2 = int(np.ceil( 0.85 * dataset_size ))
        
        # Shuffle indices
        np.random.shuffle(indices)

        # Change folds order        
        aux = (self.fold-1) * partition_size
        indices = indices[aux:] + indices[:aux]
        
        # Get training and validation indices
        train_indices, val_indices, test_indices = indices[:p1], indices[p1:p2], indices[p2:]
        
        # Sort indices
        train_indices.sort()
        val_indices.sort()
        test_indices.sort()
        
        if dataset == 'train':
            return csv.iloc[train_indices]
        elif dataset == 'val':
            return csv.iloc[val_indices]
        else:
            return csv.iloc[test_indices]

        
