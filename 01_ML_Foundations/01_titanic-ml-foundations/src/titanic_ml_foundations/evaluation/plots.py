import matplotlib.pyplot as plt
from pathlib import Path


def save_confusion_matrix(cm, path: Path) -> None:
    """Saves a confusion matrix plot to the specified path.

    Args:
        cm (array-like): Confusion matrix to plot.
        path (Path): Path to save the plot.
    """
    path.parent.mkdir(parents=True, exist_ok=True)  # ensure directory exists

    fig, ax = plt.subplots(figsize=(8, 8))  # create a figure and axis
    im = ax.imshow(cm, cmap=plt.cm.Blues)  # display the confusion matrix
    ax.set_title("Confusion Matrix", pad=10)
    fig.colorbar(im, ax=ax)  # add color bar
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Died", "Survived"])  # replace 0,1 with labels
    ax.set_yticklabels(["Died", "Survived"])  # replace 0,1 with labels
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Loop over data dimensions and create text annotations.
    tags = [["TN", "FP"], ["FN", "TP"]]
    for i in range(cm.shape[0]):  # row
        for j in range(cm.shape[1]):  # column
            ax.text(
                j,  # column index
                i,  # row index
                f"{tags[i][j]}\n{cm[i, j]}",  # row, column
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.tight_layout()  # adjust layout
    fig.savefig(path, bbox_inches="tight", dpi=150)  # save the figure
    plt.close(fig)  # close
 
