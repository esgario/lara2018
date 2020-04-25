# Classifier

## Running

```
# Training and Validation
python main.py --train --batch_size 24 --optimizer sgd --data_augmentation standard --model resnet50 --filename example

# Test
python main.py --batch_size 24 --model resnet50 --filename example
```

Obs: Other argument options can be consulted in main.py.