# Classifier

## Running

```
# Training and Validation
python main.py --train --batch_size 24 --optimizer sgd --data_augmentation standard --model resnet50 --filename example

# Testing
python main.py --test --batch_size 24 --model resnet50 --filename example
```

obs: Other argument options can be consulted in main.py. Also, it is important to note that this repository does not contain the weights of a model. So, first you must train a model until it is saved, the model name is set with the --filename flag.