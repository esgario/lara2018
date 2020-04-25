import sys
import warnings
import argparse
from segmentation import SemanticSegmentation

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
        "--batch_size", type=int, help="Set images batch size", default=4
    )
    parser.add_argument(
        "--weight_decay", type=float, help="Set L2 parameter norm penalty", default=5e-4
    )
    parser.add_argument(
        "--snapshot", type=str, help="Path to pretrained weights", default=None
    )
    parser.add_argument(
        "--extractor",
        type=str,
        help="Select features extractor architecture [unetresnet50/pspresnet50]",
        default="pspresnet50",
    )
    parser.add_argument(
        "--epochs", type=int, help="Set the number of epochs.", default=80
    )
    parser.add_argument(
        "--data_augmentation",
        type=str,
        help="Select the data augmentation technique [standard/mixup]",
        default="standard",
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="Network weights output file name.",
        default="default",
    )
    parser.add_argument("--train", help="Run in training mode.", action="store_true")

    options = parser.parse_args()


    Seg = SemanticSegmentation(parser)

    if options.train:
        Seg.run_training()
    else:
        Seg.run_test()
