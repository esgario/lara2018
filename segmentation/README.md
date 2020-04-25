# Semantic segmentation

## Running

```
# Training
python main.py --train --batch_size 4 --optimizer sgd --data_augmentation standard --extractor pspresnet50 --filename example

# Validation
python main.py --batch_size 4 --extractor pspresnet50 --filename example
```

Obs: Other argument options can be consulted in main.py.