#!/bin/bash
run(){
    python main.py --train $1 --select_clf $2 --data_augmentation $3 --model $4 --balanced_dataset $5 --output_filename $6
}

# Example
run true 0 standard resnet50 false example_experiment


