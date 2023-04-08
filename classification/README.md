# Classification

## Running

```
# Training and Validation
python main.py --train --batch_size 24 --optimizer sgd --data_augmentation standard --model resnet50 --experiment_name example

# Testing
python main.py --test --batch_size 24 --model resnet50 --experiment_name example
```

| Argument           | Type       | Description       | Options       | Default     | 
| ------------------ | ---------- | ----------------- | ------------- | ----------- |
|--train             | bool       | Run in training mode. | | False |
|--test              | bool       | Run in test mode. | | False |
|--dataset           | str        | Select the dataset to use. | leaf and symptom	| leaf |
|--model_task        | int        | Select the model task according to the dataset. Leaf dataset: (0) biotic stress only, (1) severity only, (2) multitask. Symptom dataset: (0) biotic stress only. | 0, 1 or 2 | 2 |
|--results_path      | str        | Path to the results folder. | | results |
|--experiment_name   | str        | Name of the experiment. | | experiment |
|--optimizer         | str        | Select the desired optimization technique. | sgd or adam | sgd |
|--batch_size        | int        | Set images batch size. | | 24 |
|--weight_decay      | float      | Set L2 parameter norm penalty. | | 5e-4 |
|--data_augmentation | str        | Select the data augmentation technique. | standard, mixup or bc+ | standard |
|--model             | str        | Select CNN architecture. | resnet34, resnet50, resnet101, alexnet, googlenet, vgg16 or mobilenet_v2 | resnet50 |
|--epochs            | int        | Set the number of epochs. | | 80 |
|--pretrained        | bool       | Defines whether or not to use a pre-trained model. | | True |

obs: Other argument options can be consulted in main.py. Also, it is important to note that this repository does not contain the weights of a model. So, first you must train a model until it is saved, the model name is set with the --filename flag.