import sys
import warnings
import argparse
from functools import partial
from classifiers import SingleTaskClassifier, MultiTaskClassifier
from loaders import coffeeleaves_loader, images_loader
from results import save_results
from utils.enums import Tasks

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--optimizer",
        type=str,
        help="Select the desired optimization technique [sgd/adam].",
        default="sgd",
    )
    parser.add_argument("--batch_size", type=int, help="Set images batch size", default=24)
    parser.add_argument(
        "--weight_decay", type=float, help="Set L2 parameter norm penalty", default=5e-4
    )
    parser.add_argument(
        "--data_augmentation",
        type=str,
        help="Select the data augmentation technique [standard/mixup/bc+]",
        default="standard",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Select CNN architecture [resnet34/resnet50/resnet101/alexnet/googlenet/vgg16/mobilenet_v2]",
        default="resnet50",
    )
    parser.add_argument("--epochs", type=int, help="Set the number of epochs.", default=80)
    parser.add_argument(
        "--pretrained",
        type=bool,
        help="Defines whether or not to use a pre-trained model.",
        default=True,
    )
    # this is experimental, I do not recommend using it.
    parser.add_argument("--balanced_dataset", type=bool, default=False)
    parser.add_argument(
        "--csv_file",
        type=str,
        help="Path of the dataset csv file.",
        default="dataset/dataset.csv",
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="The data is changed based on the selected fold [1-5].",
        default=1,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment.",
        default="experiment",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Path to the results folder.",
        default="results",
    )
    parser.add_argument("--train", help="Run in training mode.", action="store_true")
    parser.add_argument("--test", help="Run in test mode.", action="store_true")
    parser.add_argument(
        "--dataset",
        help="Select the dataset to use. Options: leaf, symptom",
        type=str,
        default="leaf",
    )
    parser.add_argument(
        "--model_task",
        help=(
            "Select the model task according to the dataset. "
            "Leaf dataset: (0) biotic stress only, (1) severity only, (2) multitask. "
            "Symptom dataset: (0) biotic stress only."
        ),
        type=int,
        default=2,
    )

    # Parse the arguments
    options = parser.parse_args()

    if options.dataset == "leaf":
        options.model_task = Tasks(options.model_task)
    else:
        options.model_task = Tasks.BIOTIC_STRESS

    # Validate the arguments
    assert (
        options.train or options.test
    ), "You must specify wheter you want to train or test a model."

    assert options.dataset in [
        "leaf",
        "symptom",
    ], "You must specify a valid dataset."

    # Initialize the classifier
    if options.model_task == Tasks.MULTITASK:
        Clf = MultiTaskClassifier(
            images_dir=f"dataset/{options.dataset}",
            csv_file=options.csv_file,
            fold=options.fold,
            num_classes=(5, 5),
            model_task=options.model_task,
            balanced_dataset=options.balanced_dataset,
            batch_size=options.batch_size,
            epochs=options.epochs,
            model=options.model,
            pretrained=options.pretrained,
            optimizer=options.optimizer,
            weight_decay=options.weight_decay,
            data_augmentation=options.data_augmentation,
            results_path=options.results_path,
            experiment_name=options.experiment_name,
        )
    else:
        Clf = SingleTaskClassifier(
            images_dir=f"dataset/{options.dataset}",
            num_classes=5,
            balanced_dataset=options.balanced_dataset,
            batch_size=options.batch_size,
            epochs=options.epochs,
            model=options.model,
            pretrained=options.pretrained,
            optimizer=options.optimizer,
            weight_decay=options.weight_decay,
            data_augmentation=options.data_augmentation,
            results_path=options.results_path,
            experiment_name=options.experiment_name,
        )

    # Initialize the data loader
    if options.dataset == "leaf":
        loader = partial(
            coffeeleaves_loader,
            csv_file=options.csv_file,
            fold=options.fold,
            model_task=options.model_task,
        )
    else:
        loader = images_loader

    # Run the classifier
    if options.train:
        Clf.run_training(loader)

    elif options.test:
        out = Clf.run_test(loader)
        save_results(out, options.model_task, options.results_path, options.experiment_name)

    else:
        raise ValueError("You must specify wheter you want to train or test a model.")
