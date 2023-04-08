import os


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
