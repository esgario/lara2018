import sys
import warnings
import argparse
from segmentation import SemanticSegmentation

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Training settings
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--snapshot', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--extractor', type=str, default='pspresnet50')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_augmentation', type=str, default='std')
    # Filename
    parser.add_argument('--filename', type=str, default='default')
    # Train and Validation -> True, Test -> False
    parser.add_argument('--train', action='store_true')
    
    options = parser.parse_args()
    
    # -------------------------------------------------------- #
    Seg = SemanticSegmentation(parser)
    
    if options.train:
        Seg.run_training()
    else:
        Seg.run_test()
