import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

metrics = [
    "Accuracy",
    "F1",
    "Precision",
    "Recall",
    "Balanced",
]

metrics_ = ["Accuracy", "F1", "Precision", "Recall", "BAC"]
bar_width = 0.25

matplotlib.rcParams["figure.dpi"] = 600
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False


def experiment_comparison_plot():
    df1 = pd.read_csv("./csv/yolo_8.csv")
    df2 = pd.read_csv("./csv/yolo_8_detect.csv")
    df3 = pd.read_csv("./csv/yolo_9.csv")

    results = np.zeros((3, 5))
    for i, metric in enumerate(metrics):
        results[0][i] = df1[metric].mean()
        results[1][i] = df2[metric].mean()
        results[2][i] = df3[metric].mean()

    plt.figure(figsize=(10, 10))
    plt.bar(
        np.arange(0.8, 5.8, 1),
        results[0],
        0.2,
        label="Yolo 8 Cls",
    )
    plt.bar(np.arange(1, 6), results[1], 0.2, label="Yolo 8")
    plt.bar(
        np.arange(1.2, 6.2, 1),
        results[2],
        0.2,
        label="Yolo 9",
    )
    plt.xlabel("Metrics")
    plt.ylabel("Value")
    plt.xticks(ticks=np.arange(1, 6), labels=metrics_)  # Set X-axis ticks
    plt.legend()
    plt.ylim(0.9)
    plt.grid(axis="y")
    plt.savefig("./results/comparison_metrics.png", bbox_inches="tight")
    plt.clf()

    plt.figure(figsize=(10, 10))
    plt.bar(
        np.arange(0.8, 10.8, 1),
        df1["Accuracy"],
        0.2,
        label="Yolo 8 Cls",
    )
    plt.bar(np.arange(1, 11, 1), df2["Accuracy"], 0.2, label="Yolo 8")
    plt.bar(
        np.arange(1.2, 11.2, 1),
        df3["Accuracy"],
        0.2,
        label="Yolo 9",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.xticks(ticks=np.arange(1, 11), labels=np.arange(1, 11))  # Set X-axis ticks
    plt.legend()
    plt.grid(axis="y")
    plt.ylim(0.9)
    plt.savefig("./results/comparison_accuracy.png", bbox_inches="tight")
    plt.clf()


def ttest():
    res = [
        pd.read_csv("./csv/yolo_8.csv"),
        pd.read_csv("./csv/yolo_8_detect.csv"),
        pd.read_csv("./csv/yolo_9.csv"),
    ]

    t_stat = np.empty(shape=(3, 3))
    p_val = np.empty(shape=(3, 3))
    results = np.empty(shape=(3, 3), dtype="bool")
    alpha = np.empty(shape=(3, 3), dtype="bool")
    cross = np.empty(shape=(3, 3), dtype="bool")

    for i in range(0, 3):
        for j in range(0, 3):
            t, p = stats.ttest_rel(res[i]["Accuracy"], res[j]["Accuracy"])
            t_stat[i, j] = t
            p_val[i, j] = p
            results[i, j] = t is not np.nan and t > 0
            alpha[i, j] = t is not np.nan and p < 0.05
            cross[i, j] = alpha[i, j] and results[i, j]
    print(cross)


if __name__ == "__main__":
    experiment_comparison_plot()
    ttest()
