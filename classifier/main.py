import sys
import warnings
import argparse
from classifiers import OneTaskClf, MultiTaskClf

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
    parser.add_argument(
        "--batch_size", type=int, help="Set images batch size", default=24
    )
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
    parser.add_argument(
        "--epochs", type=int, help="Set the number of epochs.", default=80
    )
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
        "--filename",
        type=str,
        help="Network weights output file name.",
        default="default",
    )
    parser.add_argument("--train", help="Run in training mode.", action="store_true")
    parser.add_argument("--test", help="Run in test mode.", action="store_true")
    parser.add_argument(
        "--select_clf",
        help="Select the experiment to run. Leaf dataset: (0) multitask, (1) biotic stress only, (2) severity only. Symptom dataset: (3) biotic stress only.",
        type=int,
        default=0,
    )

    options = parser.parse_args()

    assert (options.train or options.test), "You must specify wheter you want to train or test a model."

    # Leaf Dataset
    if options.select_clf < 3:
        parser.add_argument("--images_dir", type=str, default="dataset/leaf")
        Clf = MultiTaskClf(parser) if options.select_clf == 0 else OneTaskClf(parser)

    # Symptom Dataset
    else:
        parser.add_argument("--images_dir", type=str, default="dataset/symptom")
        Clf = OneTaskClf(parser)

    if options.train:
        Clf.run_training()
    
    if options.test:
        if options.select_clf == 0:
            y_true_dis, y_pred_dis, y_true_sev, y_pred_sev = Clf.run_test()

        else:
            y_true, y_pred = Clf.run_test()
