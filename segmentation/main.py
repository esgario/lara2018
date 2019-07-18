import os

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import click
import numpy as np

from pspnet import PSPNet

import torchvision.transforms as transforms
from utils.customdatasets import CoffeeSegmentationLoader
import utils.augmentations as aug


def data_loader(split='train', batch_size=4):
    # Augmentations
    if split == 'train':
        augs = aug.Compose([aug.RandomRotate(10), aug.RandomHorizontallyFlip(0.5)])
        shuffle = True
    else:
        augs = None
        shuffle = False

    dataset = CoffeeSegmentationLoader(root='dataset/', augmentations=augs, split=split, img_size=(224, 224))
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    
    class_weight = torch.tensor([1., 1., 1.])
    if torch.cuda.is_available():
        class_weight = class_weight.cuda()

    return loader, class_weight, len(dataset)

## Declare models
models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

# Building network
def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    
    # Instantiate network model
    net = models[backend]()
    
    # Parallel training
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    
    # Load a pretrained network    
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
    
    # Sending model to GPU
    if torch.cuda.is_available():
        net = net.cuda()
        
    return net, epoch

# Arguments
@click.command()
@click.option('--models-path', type=str, default='net_weights', help='Path for storing model snapshots')
@click.option('--backend', type=str, default='resnet50', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--batch-size', type=int, default=4)
@click.option('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
@click.option('--epochs', type=int, default=20, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start-lr', type=float, default=0.001)
@click.option('--milestones', type=str, default='10,20,30', help='Milestones for LR decreasing')
def train(models_path, backend, snapshot, batch_size, alpha, epochs, start_lr, milestones, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    model, starting_epoch = build_network(snapshot, backend)
    models_path = os.path.abspath(os.path.expanduser(models_path))
    os.makedirs(models_path, exist_ok=True)
    
    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''
    train_loader, class_weights, n_images = data_loader('train', batch_size)
    
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])
    
    for epoch in range(starting_epoch, starting_epoch + epochs):
        seg_criterion = nn.NLLLoss(weight=class_weights)
        cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        epoch_losses = []
        train_iterator = tqdm(train_loader, total=n_images // batch_size + 1)
        model.train()
        
        for x, y, y_cls in train_iterator:
            optimizer.zero_grad()
            x, y, y_cls = x.cuda(), y.cuda(), y_cls.cuda()
            out, out_cls = model(x)
            seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
            loss = seg_loss + alpha * cls_loss
            epoch_losses.append(loss.data.cpu())
            #status = '[{0}] loss = {1:0.5f} avg = {2:0.5f}, LR = {5:0.7f}'.format(
            #    epoch + 1, loss.data.cpu(), np.mean(epoch_losses), scheduler.get_lr())
            status = '[%i] loss = %.5f avg = %.5f, LR = %.7f' % (epoch + 1, loss.data.cpu(), np.mean(epoch_losses), scheduler.get_lr()[0])
            train_iterator.set_description(status)
            loss.backward()
            optimizer.step()
            
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", str(epoch + 1)])))
        train_loss = np.mean(epoch_losses)

        
if __name__ == '__main__':
    train()
