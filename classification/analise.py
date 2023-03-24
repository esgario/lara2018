import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot(file, t, plot_param, legend):
    
    # Standard
    with open(file, 'rb') as f:
        data = pickle.load(f)
        
    plt.plot(np.arange(1, len(data[t]) + 1), data[t], plot_param, linewidth=1.2, label=legend)
    plt.legend(loc='lower right')

# ================================================

# Create plot
plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(6, 4))
#ax.axis((-1,101,82,99))
ax.grid(linestyle='--')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss(%)')

# ================================================
metric = 'val_acc'
path = 'log/symptom/'
#plot(path + 'alexnet_sgd_32_standard.pkl', metric, '-r', 'AlexNet')
#plot(path + 'googlenet_sgd_32_standard.pkl', metric, '-g', 'GoogLeNet')
#plot(path + 'vgg16_sgd_32_standard.pkl', metric, '-b', 'VGG16')
#plot(path + 'resnet50_sgd_32_standard.pkl', metric, '-k', 'ResNet50')

#plot(path + 'resnet50_sgd_32_mixup.pkl', 'train_loss', '-k', 'Train')

plot(path + 'resnet50_sgd_32_standard.pkl', 'train_acc', '-b', 'Train')
plot(path + 'resnet50_sgd_32_standard.pkl', 'val_acc', '-r', 'Val')
plot(path + 'resnet50_sgd_32_mixup.pkl', 'train_acc', '-g', 'Train')
plot(path + 'resnet50_sgd_32_mixup.pkl', 'val_acc', '-k', 'Val')

#plot(path + 'resnet50_sgd_32_standard.pkl', 'train_loss', '-b', 'Std')
#plot(path + 'resnet50_sgd_32_mixup.pkl', 'train_loss', '-r', 'Mixup')

#plt.show()


