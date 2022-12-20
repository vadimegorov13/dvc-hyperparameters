import pandas as pd
import matplotlib.pyplot as plt
from dvc.api import params_show
from pathlib import Path
import numpy as np
import json

# Extract the parameters
params = params_show()["train"]


def plot_metric(metric_df: pd.DataFrame, metric_name: str, plot_path: str):
    """
    A function to plot both training and validation metrics.
    """
    # Plot metric_name and val_metric_name
    fig, ax = plt.subplots()

    epochs = np.arange(1, len(metric_df) + 1)

    ax.plot(epochs, metric_df[metric_name], "b", label=f"Training {metric_name}")
    ax.plot(
        epochs, metric_df["val_" + metric_name], "bo", label=f"Validation {metric_name}"
    )

    plt.xlabel("Epoch")
    plt.title(f"Training and validation {metric_name}")
    plt.legend()
    plt.savefig(plot_path)


if __name__ == "__main__":
    # Create the path for plots
    Path("plots").mkdir(exist_ok=True)

    # Read the metrics and plot
    metrics = pd.read_csv("metrics/metrics.csv")
    metric_names = ["accuracy", "loss", "precision", "recall"]

    for current_metric in metric_names:
        plot_metric(metrics, current_metric, f"plots/{current_metric}.png")

    #########################################################################################
    # Usually, the below step would involve reporting the metric on a third, final test set #
    #########################################################################################

    # Save the best metrics as json
    sorted_metrics = metrics.sort_values("val_accuracy", ascending=False).reset_index(
        drop=True
    )
    metrics_dict = {
        "val_" + metric: sorted_metrics["val_" + metric][0] for metric in metric_names
    }

    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics_dict, f)
