import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler

from utils.enums import Tasks
from utils.customdatasets import CoffeeLeavesDataset


def sampler(dataset, opt):
    # Multiclass umbalanced dataset
    if opt.dataset == "leaf" and opt.model_task == Tasks.MULTITASK:
        balance_factor = 20

        data = np.array(dataset.data)
        dis = data[:, 1]
        sev = data[:, -1]

        total = len(dis)
        samplesWeight = np.zeros(total)

        for d in range(5):
            for s in range(5):
                targets_sum = sum([a and b for a, b in zip(dis == d, sev == s)])

                idx = np.where([a and b for a, b in zip(dis == d, sev == s)])
                samplesWeight[idx] = 1 / ((targets_sum + balance_factor) / total)

    elif opt.dataset == "leaf":
        data = np.array(dataset.data)

        if opt.model_task == Tasks.BIOTIC_STRESS:
            labels = data[:, 1]
        else:
            labels = data[:, -1]

        total = len(labels)
        samplesWeight = np.zeros(total)

        for i in range(5):
            targets_sum = sum(labels == i)
            idx = np.where(labels == i)
            samplesWeight[idx] = 1 / ((targets_sum) / total)

    # Symptom dataset
    else:
        targets = np.array([x[1] for x in dataset.samples])
        total = len(targets)

        samplesWeight = np.zeros(total)

        for t in np.unique(targets):
            idx = np.where(targets == t)[0]

            samplesWeight[idx] = 1 / (len(idx) / total)

    samplesWeight = samplesWeight / sum(samplesWeight)
    samplesWeight = torch.from_numpy(samplesWeight).double()

    return WeightedRandomSampler(samplesWeight, len(samplesWeight))


def data_loader(opt):
    # Transforms
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomApply([transforms.RandomRotation(10)], 0.25),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Dataset: Leaf
    if opt.dataset == "leaf":
        train_dataset = CoffeeLeavesDataset(
            csv_file=opt.csv_file,
            images_dir=opt.images_dir,
            dataset="train",
            fold=opt.fold,
            model_task=opt.model_task,
            transforms=train_transforms,
        )

        val_dataset = CoffeeLeavesDataset(
            csv_file=opt.csv_file,
            images_dir=opt.images_dir,
            dataset="val",
            fold=opt.fold,
            model_task=opt.model_task,
            transforms=val_transforms,
        )

        test_dataset = CoffeeLeavesDataset(
            csv_file=opt.csv_file,
            images_dir=opt.images_dir,
            dataset="test",
            fold=opt.fold,
            model_task=opt.model_task,
            transforms=val_transforms,
        )

    # Dataset: Symptom
    else:
        train_dataset = torchvision.datasets.ImageFolder(
            root=opt.images_dir + "/train/", transform=train_transforms
        )

        val_dataset = torchvision.datasets.ImageFolder(
            root=opt.images_dir + "/val/", transform=val_transforms
        )

        test_dataset = torchvision.datasets.ImageFolder(
            root=opt.images_dir + "/test/", transform=val_transforms
        )

    # Loader
    if opt.balanced_dataset:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=opt.batch_size, sampler=sampler(train_dataset, opt)
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=opt.batch_size, shuffle=True
        )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=opt.batch_size, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=opt.batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader
