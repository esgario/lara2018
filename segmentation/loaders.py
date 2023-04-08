import torch
import utils.augmentations as aug
from utils.customdatasets import SegmentationLoader


def data_loader(split="train", batch_size=4):
    # Augmentations
    if split == "train":
        augs = aug.Compose(
            [
                aug.RandomRotate(10),
                aug.RandomHorizontallyFlip(0.5),
                aug.RandomVerticallyFlip(0.5),
                aug.AdjustContrast(0.25),
                aug.AdjustBrightness(0.25),
                aug.AdjustSaturation(0.25),
            ]
        )
        shuffle = True
    else:
        augs = None
        shuffle = False

    dataset = SegmentationLoader(root="dataset/", augmentations=augs, split=split)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    class_weight = torch.tensor([1.0, 1.0, 2.0])
    if torch.cuda.is_available():
        class_weight = class_weight.cuda()

    if split != "test":
        return loader, class_weight, len(dataset)
    else:
        return loader, dataset
