import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

import utils.augmentations as aug
from utils.customdatasets import SegmentationLoader
from utils.metric import scores
from net_models import PSPNet, UNetWithResnet50Encoder

import pickle
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## Declare models
models = {
    'pspsqueezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'pspdensenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'pspresnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'pspresnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'pspresnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'pspresnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'pspresnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152'),
    'unetresnet50': lambda: UNetWithResnet50Encoder(n_classes=3)
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

def adjust_learning_rate(optimizer, epoch, opt):
    if opt.optimizer == 'sgd':
        lr_values = [ 0.01, 0.005, 0.001, 0.0005, 0.0001 ]
        step = round(opt.epochs / 5)

        idx = min(math.floor(epoch/step), len(lr_values))
        learning_rate = lr_values[idx]

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    return optimizer

def data_loader(split='train', batch_size=4):
    # Augmentations
    if split == 'train':
        augs = aug.Compose([aug.RandomRotate(10),
                            aug.RandomHorizontallyFlip(0.5),
                            aug.RandomVerticallyFlip(0.5),
                            aug.AdjustContrast(0.25),
                            aug.AdjustBrightness(0.25),
                            aug.AdjustSaturation(0.25)])
        shuffle = True
    else:
        augs = None
        shuffle = False

    dataset = SegmentationLoader(root='dataset/', augmentations=augs, split=split)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    
    class_weight = torch.tensor([1., 1., 2.])
    if torch.cuda.is_available():
        class_weight = class_weight.cuda()

    if split != 'test':
        return loader, class_weight, len(dataset)
    else:
        return loader, dataset

def eval_metric(label_trues, label_preds, n_class):
    label_preds = torch.max(label_preds, 1)[1]
    
    if len(label_trues.shape) > 2:
        acc, miou = [], []
        for i in range(label_trues.shape[0]):
            score = scores(label_trues[i].cpu(), label_preds[i].cpu(), n_class)[0]
            miou.append(score['mean iou'])
            acc.append(score['overall acc'])
        return np.mean(miou), np.mean(acc) 
    else:
        score = scores(label_trues.cpu(), label_preds.cpu(), n_class)[0]
        return score['mean iou'], score['overall acc']
    
def scatterPlot(x_true, x_pred, output_name, figsize=(6,4.5), marker_color='g', curve_color='k'):
    reg = LinearRegression()
    reg.fit(x_true[:,None], x_pred[:,None])
    x_true_line = np.linspace(x_true.min(), x_true.max())
    x_pred_line = reg.predict(x_true_line[:,None])
    r2 = reg.score(x_true[:,None], x_pred[:,None])
    
    fig = plt.figure(figsize=figsize)
    #ax = fig.gca()
    
    title = 'R²=:%.4f' % r2 
    
    plt.plot(x_true_line, x_pred_line, curve_color, alpha=0.7)
    plt.plot(x_true, x_pred, 'o%s' % marker_color, markersize=6, alpha=0.7)
    plt.grid()
    plt.xlabel('True severity')
    plt.ylabel('Predicted severity')
    plt.xlim(-0.01, 0.23)
    plt.ylim(-0.01, 0.24)
    plt.title(title)
    plt.show()
    fig.savefig('results/' + output_name + '.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    
# -------------------------------------------------------------------------------------- #
    
class SemanticSegmentation:
    def __init__(self, parser):
        self.opt = parser.parse_args()
        
    def train(self, train_loader, n_images, batch_size, epoch, model, seg_criterion, cls_criterion, optimizer, data_augmentation='mixup'):
        
        model.train()
        train_metrics = { 'loss': [], 'miou': [], 'acc': [] }
        
        train_iterator = tqdm(train_loader, total=n_images // batch_size + 1)
        for x, y, y_cls in train_iterator:
            
            # Loading images on gpu
            if torch.cuda.is_available():
                x, y, y_cls = x.cuda(), y.cuda(), y_cls.cuda()
                
            if data_augmentation == 'mixup':
                x, y_a, y_b, y_cls_a, y_cls_b, lam = aug.mixup_data(x, y, y_cls)
            
            # Pass images through the network
            out, out_cls = model(x)
            
            # Compute error
            if data_augmentation == 'mixup':
                seg_loss = aug.mixup_criterion(seg_criterion, out, y_a, y_b, lam)
                cls_loss = aug.mixup_criterion(cls_criterion, out_cls, y_cls_a, y_cls_b, lam)
            else:            
                seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
            
            loss = seg_loss + cls_loss
            
            # Clear gradients parameters
            model.zero_grad()            
            
            # Getting gradients
            loss.backward()
            
            # Updating parameters
            optimizer.step()
            
            # Metrics
            train_metrics['loss'].append(loss.data.cpu())
            
            if data_augmentation == 'mixup':
                metrics_a, metrics_b = eval_metric(y_a, out, 3), eval_metric(y_b, out, 3) 
                train_metrics['miou'].append(lam * metrics_a[0] + (1 - lam) * metrics_b[0])
                train_metrics['acc'].append(lam * metrics_a[1] + (1 - lam) * metrics_b[1])
            else:    
                metrics = eval_metric(y, out, 3)
                train_metrics['miou'].append(metrics[0])
                train_metrics['acc'].append(metrics[1])
            
            status = '[%i] loss = %.4f avg = %.4f, miou = %.4f, acc = %.4f' % (epoch + 1,
                                                                              loss.data.cpu(),
                                                                              np.mean(train_metrics['loss']),
                                                                              np.mean(train_metrics['miou']),
                                                                              np.mean(train_metrics['acc']))
            train_iterator.set_description(status)
        
        train_metrics['loss'] = np.mean(train_metrics['loss'])
        train_metrics['miou'] = np.mean(train_metrics['miou'])
        train_metrics['acc'] = np.mean(train_metrics['acc'])
        return train_metrics
    
    def validation(self, val_loader, n_images, batch_size, epoch, model, seg_criterion, cls_criterion):
        # tell to pytorch that we are evaluating the model
        model.eval()
    
        val_metrics = { 'loss': [], 'miou': [], 'acc': [] }
        
        val_iterator = tqdm(val_loader, total=n_images // batch_size + 1)            
        
        
        for x, y, y_cls in val_iterator:
            with torch.no_grad():
                # Loading images on gpu
                if torch.cuda.is_available():
	                x, y, y_cls = x.cuda(), y.cuda(), y_cls.cuda()

	            # pass images through the network
                out, out_cls = model(x)
                
                # Computer error
                seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
                loss = seg_loss + cls_loss

	            # Compute metrics
	            ## Loss
                # Metrics
                val_metrics['loss'].append(loss.data.cpu())
                metrics = eval_metric(y, out, 3)
                val_metrics['miou'].append(metrics[0])
                val_metrics['acc'].append(metrics[1])

                status = '[%i] loss = %.4f avg = %.4f, miou = %.4f, acc = %.4f' % (epoch + 1,
                                                                              loss.data.cpu(),
                                                                              np.mean(val_metrics['loss']),
                                                                              np.mean(val_metrics['miou']),
                                                                              np.mean(val_metrics['acc']))
                val_iterator.set_description(status)
        
        val_metrics['loss'] = np.mean(val_metrics['loss'])
        val_metrics['miou'] = np.mean(val_metrics['miou'])
        val_metrics['acc'] = np.mean(val_metrics['acc'])
        return val_metrics
      
    
    def run_training(self):
        # Dataset
        train_loader, class_weights, n_images_train = data_loader('train', self.opt.batch_size)
        val_loader, _, n_images_val = data_loader('val', self.opt.batch_size)
        
        # Model
        model, starting_epoch = build_network(self.opt.snapshot, self.opt.extractor)
        os.makedirs(os.path.abspath('net_weights'), exist_ok=True)

        # Criterion
        seg_criterion = nn.NLLLoss(weight=class_weights)
        cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)

        # Optimizer
        if self.opt.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=self.opt.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=self.opt.weight_decay)

        record = {}
        record['extractor'] = self.opt.extractor
        record['batch_size'] = self.opt.batch_size
        record['weight_decay'] = self.opt.weight_decay
        record['optimizer'] = self.opt.optimizer
        record['epochs'] = self.opt.epochs
        record['train_loss'] = []
        record['val_loss'] = []
        record['train_miou'] = []
        record['val_miou'] = []
        record['train_acc'] = []
        record['val_acc'] = []
        
        os.makedirs(os.path.abspath('log'), exist_ok=True)

        best_loss = 1000.0
        
        for epoch in range(starting_epoch, starting_epoch + self.opt.epochs):
                
            # Training
            train_metrics = self.train(
                                    train_loader,
                                    n_images_train,
                                    self.opt.batch_size,
                                    epoch,
                                    model,
                                    seg_criterion,
                                    cls_criterion,
                                    optimizer,
                                    self.opt.data_augmentation
                                    )
            
            # Validation
            val_metrics = self.validation(
                                    val_loader,
                                    n_images_val,
                                    self.opt.batch_size,
                                    epoch,
                                    model,
                                    seg_criterion,
                                    cls_criterion
                                    )

            # Adjust learning rate
            optimizer = adjust_learning_rate(optimizer, epoch, self.opt)

            # Recording metrics
            record['train_loss'].append(train_metrics['loss'])
            record['train_miou'].append(train_metrics['miou'])
            record['train_acc'].append(train_metrics['acc'])

            record['val_loss'].append(val_metrics['loss'])
            record['val_miou'].append(val_metrics['miou'])
            record['val_acc'].append(val_metrics['acc'])

            # Record best model
            curr_loss = val_metrics['loss']
            if (curr_loss < best_loss):
                best_loss = curr_loss

                # Saving model
                torch.save(model, 'net_weights/' + self.opt.filename + '.pth')
                
                print('model saved')

            # Saving log
            fp = open('log/'+ self.opt.filename +'.pkl', 'wb')
            pickle.dump(record, fp)
            fp.close()

        # Plot
        # static_graph(np.array(record['train_dis_acc'])/100, np.array(record['val_dis_acc'])/100)

    def run_test(self):
        os.makedirs(os.path.abspath('results'), exist_ok=True)
        
        # Dataset
        test_loader, test_dataset = data_loader('test', self.opt.batch_size)

        # Loading model
        model = torch.load('net_weights/' + self.opt.filename + '.pth')
        model.cuda()
        
        # tell to pytorch that we are evaluating the model
        model.eval()
        
        test_metrics = { 'miou': [], 'acc': [] }
        severity = { 'true': np.array([]), 'pred': np.array([]) }

        with torch.no_grad():
            for imgs, labels, cls in test_loader:
                # Loading images on gpu
                if torch.cuda.is_available():
                    imgs, labels, cls = imgs.cuda(), labels.cuda(), cls.cuda()
        
                # pass images through the network
                out, out_cls = model(imgs)
                labels_pred = torch.max(out, 1)[1]
    
                # Compute metrics
                metrics = eval_metric(labels, out, 3)
                test_metrics['miou'].append(metrics[0])
                test_metrics['acc'].append(metrics[1])
                
                # Plot
                imgs = imgs.cpu().numpy()[:, ::-1, :, :]
                imgs = np.transpose(imgs, [0,2,3,1])
                
                f, axarr = plt.subplots(len(imgs), 3, figsize=(16, 11.5))
                
                for j in range(len(imgs)):
                    # Original image
                    axarr[j][0].imshow(imgs[j])
                    # True labels
                    axarr[j][1].imshow(test_dataset.decode_segmap(labels.cpu().numpy()[j]))
                    # Predicted labels
                    axarr[j][2].imshow(test_dataset.decode_segmap(labels_pred.cpu().numpy()[j]))
                
                    # Compute severity
                    aux = labels.cpu().numpy()[j]
                    severity['true'] = np.append(severity['true'], (aux==2).sum() / ((aux==1).sum() + (aux==2).sum()))
                    
                    aux = labels_pred.cpu().numpy()[j]
                    severity['pred'] = np.append(severity['pred'], (aux==2).sum() / ((aux==1).sum() + (aux==2).sum()))
                
                plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                plt.subplots_adjust(wspace=0.05, hspace=0.05)
                plt.show()

        miou = np.mean(test_metrics['miou'])
        acc = np.mean(test_metrics['acc'])

        f = open('results/' + self.opt.filename + '.csv', 'w')
        f.write('miou,acc\n%.2f,%.2f\n' % (miou*100, acc*100))
        f.close()
        
        print('miou,acc\n%.2f,%.2f\n' % (miou*100, acc*100))

        scatterPlot(severity['true'], severity['pred'], self.opt.filename)

    def get_n_params(self):
        model = torch.load('net_weights/' + self.opt.filename + '.pth')
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
