import sys
import warnings
import argparse
from classifiers import OneTaskClf, MultiTaskClf

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Training settings
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--data_augmentation', type=str, default='mixup')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--balanced_dataset', type=bool, default=False)
    # Dataset
    parser.add_argument('--csv_file', type=str, default='dataset/dataset.csv')
    parser.add_argument('--fold', type=int, default=1)
    # Output filename
    parser.add_argument('--output_filename', type=str, default='resnet50_sgd_32_mixup')
    # Train and Validation -> True, Test -> False
    parser.add_argument("--train", type=bool, default=False)
    
    # Select Classifier
    # Leaf dataset
    #   0 - multitask
    #   1 - biotic stress
    #   2 - severity
    # Symptom dataset
    #   3 - biotic stress
    parser.add_argument("--select_clf", type=int, default=3)
    
    options = parser.parse_args()
    
    # Leaf
    if options.select_clf < 3:
        # Dataset
        parser.add_argument('--images_dir', type=str, default='dataset/leaf')
        Clf = MultiTaskClf(parser) if options.select_clf == 0 else OneTaskClf(parser)
        
    # Symptom
    else:
        # Dataset
        parser.add_argument('--images_dir', type=str, default='dataset/symptom')
        Clf = OneTaskClf(parser)
    
    if options.train:
        Clf.run_training()
    else:
        if options.select_clf == 0:
            y_true_dis, y_pred_dis, y_true_sev, y_pred_sev = Clf.run_test()    
            
        else:
            y_true, y_pred = Clf.run_test()
