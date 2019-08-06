import os
import torch
import numpy as np
import imageio as m
import cv2
import pandas as pd
from torch.utils import data
import json


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class SegmentationLoader(data.Dataset):

    mean_rgb = {
        "standard": [0.0, 0.0, 0.0],
    }

    def __init__(
        self,
        root='dataset',
        images_folder='images',
        annotations_folder='annotations',
        split="train",
        is_transform=True,
        img_size=(512, 256),
        augmentations=None,
        img_norm=True,
        test_mode=False,
        version="standard"
    ):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}

        self.images_base = os.path.join(self.root, images_folder, self.split)
        self.annotations_base = os.path.join(self.root, annotations_folder, self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".jpg")
        
        # Reading dataset info
        self.data = pd.read_csv(os.path.join(self.root, 'dataset.csv'))
        with open(os.path.join(self.root, 'annotations-info.txt')) as json_file:
            annotations_info = json.load(json_file)
        
        # colors
        self.colors = annotations_info['colors']
        self.label_colours = dict(zip(range(3), self.colors))
        
        # class names
        self.class_names = annotations_info['class_names']
        self.n_classes = len(self.class_names)

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            os.path.basename(img_path)[:-4] + "_mask.png",
        )

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        
        idx = int(os.path.basename(img_path)[:-4]) - 1
        cls = torch.tensor([1., 1., 0])
        cls[2] = torch.tensor(self.data.iloc[idx, -1]) > 0
        
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, cls

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes
        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.
        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.colors):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

            
# Leave code for debugging purposes
# import ptsemseg.augmentations as aug
if __name__ == '__main__':
    
    from augmentations import Compose, RandomHorizontallyFlip, RandomVerticallyFlip, RandomRotate, Scale
    from augmentations import AdjustContrast, AdjustBrightness, AdjustSaturation
    import matplotlib.pyplot as plt

    bs = 4
    augmentations = Compose([Scale(512),
                             RandomRotate(10),
                             RandomHorizontallyFlip(0.5),
                             RandomVerticallyFlip(0.5),
                             AdjustContrast(0.25),
                             AdjustBrightness(0.25),
                             AdjustSaturation(0.25)])
            
    dst = SegmentationLoader(root='../dataset/', is_transform=True, augmentations=augmentations)
    trainloader = data.DataLoader(dst, batch_size=bs)
    
    for i, data_samples in enumerate(trainloader):
        imgs, labels, cls = data_samples
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0,2,3,1])
        f, axarr = plt.subplots(bs, 2)
    
        for j in range(bs):
            print(imgs[j].shape)
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
            
        plt.show()