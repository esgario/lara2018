import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def scatter_plot(x_true, x_pred, output_path, figsize=(6, 4.5), marker_color="g", curve_color="k"):
    reg = LinearRegression()
    reg.fit(x_true[:, None], x_pred[:, None])
    x_true_line = np.linspace(x_true.min(), x_true.max())
    x_pred_line = reg.predict(x_true_line[:, None])
    r2 = reg.score(x_true[:, None], x_pred[:, None])

    fig = plt.figure(figsize=figsize)
    # ax = fig.gca()

    title = "R²=:%.4f" % r2

    plt.plot(x_true_line, x_pred_line, curve_color, alpha=0.7)
    plt.plot(x_true, x_pred, "o%s" % marker_color, markersize=6, alpha=0.7)
    plt.grid()
    plt.xlabel("True severity")
    plt.ylabel("Predicted severity")
    plt.xlim(-0.01, 0.23)
    plt.ylim(-0.01, 0.24)
    plt.title(title)
    plt.show()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
