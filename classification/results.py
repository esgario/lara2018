import os
from utils.enums import Tasks
from utils.utils import write_results


def _save(y_true, y_pred, model_task, results_folder):
    if model_task == Tasks.SEVERITY:
        labels = ["Healthy", "Very low", "Low", "High", "Very high"]
        task_name = "severity"
    else:
        labels = ["Healhty", "Leaf miner", "Rust", "Phoma", "Cercospora"]
        task_name = "biotic_stress"

    write_results(
        y_true=y_true,
        y_pred=y_pred,
        cm_target_names=labels,
        task_name=task_name,
        results_folder=results_folder,
    )


def save_results(test_results, model_task, results_path, experiment_name):
    """Save experiment results."""
    results_folder = os.path.join(results_path, experiment_name)

    if model_task == Tasks.MULTITASK:
        y_true_dis, y_pred_dis, y_true_sev, y_pred_sev = test_results
        _save(y_true_dis, y_pred_dis, Tasks.BIOTIC_STRESS, results_folder)
        _save(y_true_sev, y_pred_sev, Tasks.SEVERITY, results_folder)

    else:
        y_true, y_pred = test_results
        _save(y_true, y_pred, model_task, results_folder)
