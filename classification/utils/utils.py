import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from bokeh.plotting import figure
from bokeh.io import show
from sklearn.metrics import confusion_matrix



def line_graph(train, val):
    x = list(range(1, len(train) + 1))

    fig = plt.figure(figsize=(4, 4))
    plt.ylim(40, 100)
    plt.plot(
        x,
        train,
        color="b",
    )
    plt.plot(x, val, color="r")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend(["Treinamento", "Teste"], loc=0)
    plt.grid()
    plt.show()
    fig.savefig("result.png", dpi=200)


def plot_confusion_matrix(
    cm,
    target_names,
    title="Confusion matrix",
    output_path="confusion_matrix.png",
    cmap=None,
    figsize=(5, 4),
):
    """
    Given a sklearn confusion matrix (cm), make a nice plot.

    Args
        cm: Confusion matrix from sklearn.metrics.confusion_matrix
        target_names: Given classification classes such as [0, 1, 2]
            the class names, for example: ['high', 'medium', 'low']
        title: The text to display at the top of the matrix.
        output_path: The output path to save the figure.
        cmap: The gradient of the values displayed from matplotlib.pyplot.cm
            see http://matplotlib.org/examples/color/colormaps_reference.html
            plt.get_cmap('jet') or plt.cm.Blues.
        figsize: default = (5,4) the size of the figure.

    Returns
        None
    """
    cm_orig = cm.copy()
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    #    plt.figure(figsize=figsize)
    fig = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5  # if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            "{:d}\n{:.3f}".format(cm_orig[i, j], cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    accuracy = np.trace(cm_orig) / float(np.sum(cm_orig))
    balanced_accuracy = (cm * np.identity(len(cm))).sum() / len(cm)
    misclass = 1 - accuracy

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(
        "Predicted label\nacc={:0.4f}; error={:0.4f}; bac={:0.4f}".format(
            accuracy, misclass, balanced_accuracy
        )
    )

    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def write_results(y_true, y_pred, cm_target_names, results_path, task_name, experiment_name):
    """Write experiment results."""
    # calculate metrics
    acc = accuracy_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred, average="macro")
    re = recall_score(y_true, y_pred, average="macro")
    fs = f1_score(y_true, y_pred, average="macro")

    results_folder = os.path.join(results_path, experiment_name)

    print(f"Saving results to {results_folder}")

    file_path = os.path.join(results_folder, "metrics.csv")

    # check if file exists
    if not os.path.isfile(file_path):
        # create file and write header
        with open(file_path, "w", encoding="UTF-8") as fp:
            fp.write("task,acc,prec,rec,fs\n")

    with open(file_path, "a", encoding="UTF-8") as fp:
        fp.write("%s,%.2f,%.2f,%.2f,%.2f\n" % (task_name, acc * 100, pr * 100, re * 100, fs * 100))

    # Confusion matrix
    cmatrix = confusion_matrix(y_true, y_pred, labels=list(range(0, 5)))

    plot_confusion_matrix(
        cm=cmatrix,
        target_names=cm_target_names,
        title=" ",
        output_path=results_folder + f"/confusion_matrix_{task_name}.png",
    )


def create_results_folder(results_path, experiment_name):
    """
    Creates a folder to store the results of the experiment.

    Args:
        results_path (str): Path to the results folder.
        experiment_name (str): Name of the experiment.

    Returns:
        str: Path to the results folder.
    """
    results_folder = os.path.join(results_path, experiment_name)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    return results_folder


def get_file_path(opt, filename):
    return os.path.join(opt.results_path, opt.experiment_name, filename)
