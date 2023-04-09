import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler

from utils.enums import Tasks
from utils.customdatasets import CoffeeLeavesDataset


def _coffeeleaves_sampler(dataset, model_task):
    """Creates a custom sampler for the Coffee Leaves dataset."""
    if model_task == Tasks.MULTITASK:
        balance_factor = 20

        data = np.array(dataset.data)
        dis = data[:, 1]
        sev = data[:, -1]

        total = len(dis)
        samples_weight = np.zeros(total)

        for d in range(5):
            for s in range(5):
                targets_sum = sum([a and b for a, b in zip(dis == d, sev == s)])

                idx = np.where([a and b for a, b in zip(dis == d, sev == s)])
                samples_weight[idx] = 1 / ((targets_sum + balance_factor) / total)

    else:
        data = np.array(dataset.data)

        if model_task == Tasks.BIOTIC_STRESS:
            labels = data[:, 1]
        else:
            labels = data[:, -1]

        total = len(labels)
        samples_weight = np.zeros(total)

        for i in range(5):
            targets_sum = sum(labels == i)
            idx = np.where(labels == i)
            samples_weight[idx] = 1 / ((targets_sum) / total)

    samples_weight = samples_weight / sum(samples_weight)
    samples_weight = torch.from_numpy(samples_weight).double()

    return WeightedRandomSampler(samples_weight, len(samples_weight))


def _sampler(dataset):
    """Creates a sampler for the dataset."""
    targets = np.array([x[1] for x in dataset.samples])
    total = len(targets)

    samples_weight = np.zeros(total)

    for t in np.unique(targets):
        idx = np.where(targets == t)[0]

        samples_weight[idx] = 1 / (len(idx) / total)

    samples_weight = samples_weight / sum(samples_weight)
    samples_weight = torch.from_numpy(samples_weight).double()

    return WeightedRandomSampler(samples_weight, len(samples_weight))


def _get_transforms():
    """Transforms."""
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

    return train_transforms, val_transforms


def _build_loaders(
    train_dataset, val_dataset, test_dataset, batch_size, balanced_dataset, sampler
):
    """Build the loaders."""
    if balanced_dataset:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, sampler=sampler
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


def coffeeleaves_loader(images_dir, batch_size, balanced_dataset, csv_file, fold, model_task):
    """Build the dataloaders."""
    train_transforms, val_transforms = _get_transforms()

    train_dataset = CoffeeLeavesDataset(
        csv_file=csv_file,
        images_dir=images_dir,
        dataset="train",
        fold=fold,
        model_task=model_task,
        transforms=train_transforms,
    )

    val_dataset = CoffeeLeavesDataset(
        csv_file=csv_file,
        images_dir=images_dir,
        dataset="val",
        fold=fold,
        model_task=model_task,
        transforms=val_transforms,
    )

    test_dataset = CoffeeLeavesDataset(
        csv_file=csv_file,
        images_dir=images_dir,
        dataset="test",
        fold=fold,
        model_task=model_task,
        transforms=val_transforms,
    )

    return _build_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        balanced_dataset,
        _coffeeleaves_sampler(train_dataset, model_task),
    )


def images_loader(images_dir, batch_size, balanced_dataset):
    """Build the dataloaders."""
    train_transforms, val_transforms = _get_transforms()

    train_dataset = torchvision.datasets.ImageFolder(
        root=images_dir + "/train/", transform=train_transforms
    )

    val_dataset = torchvision.datasets.ImageFolder(
        root=images_dir + "/val/", transform=val_transforms
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root=images_dir + "/test/", transform=val_transforms
    )

    return _build_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        balanced_dataset,
        _sampler(train_dataset),
    )
