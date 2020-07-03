# Semantic segmentation

## Running

```
# Training and Validation
python main.py --train --batch_size 4 --optimizer sgd --data_augmentation standard --extractor pspresnet50 --filename example

# Testing
python main.py --batch_size 4 --extractor pspresnet50 --filename example
```

Obs: Other argument options can be consulted in main.py. Also, it is important to note that this repository does not contain the weights of a model. So, first you must train a model until it is saved, the model name is set with the --filename flag.