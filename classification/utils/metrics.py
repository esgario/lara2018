from sklearn.metrics import accuracy_score, f1_score


def accuracy_mixup(y_pred, y_true, y2_true, lam=None):
    return lam * accuracy_score(y_true, y_pred) + (1 - lam) * accuracy_score(y2_true, y_pred)


def f1_mixup(y_pred, y_true, y2_true, lam=None):
    return lam * f1_score(y_true, y_pred, average="macro") + (1 - lam) * f1_score(
        y2_true, y_pred, average="macro"
    )
