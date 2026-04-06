import matplotlib.pyplot as plt
import seaborn as sns

from house_prices_ml_foundations.config.config import RANDOM_STATE


def plot_residuals_hist(error_df, out_path, run_id):
    plt.figure(figsize=(8, 6))
    sns.histplot(error_df["residual"], bins=30, kde=True)
    plt.title("Distribution of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid()
    file_path = out_path / f"residuals_hist_{run_id}.png"
    plt.savefig(file_path)
    print(f"Residuals histogram saved to: {file_path}")
    plt.close()
    return file_path


def plot_ytrue_vs_ypred(error_df, out_path, run_id, sample_n=500, random_state=RANDOM_STATE):
    sample_n = min(sample_n, len(error_df))
    sample_df = error_df.sample(n=sample_n, random_state=random_state)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=sample_df["y_true"], y=sample_df["y_pred"])
    plt.plot(
        [sample_df["y_true"].min(), sample_df["y_true"].max()],
        [sample_df["y_true"].min(), sample_df["y_true"].max()],
        "r--",
    )
    plt.title("True vs Predicted Values")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.grid()
    file_path = out_path / f"ytrue_vs_ypred_{run_id}.png"
    plt.savefig(file_path)
    print(f"True vs Predicted plot saved to: {file_path}")
    plt.close()
    return file_path


def plot_abs_error_vs_ytrue(error_df, out_path, run_id, sample_n=500, random_state=RANDOM_STATE):
    sample_n = min(sample_n, len(error_df))
    sample_df = error_df.sample(n=sample_n, random_state=random_state)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=sample_df["y_true"], y=sample_df["abs_error"])
    plt.title("Absolute Error vs True Values")
    plt.xlabel("True Values")
    plt.ylabel("Absolute Error")
    plt.grid()
    file_path = out_path / f"abs_error_vs_ytrue_{run_id}.png"
    plt.savefig(file_path)
    print(f"Absolute Error vs True Values plot saved to: {file_path}")
    plt.close()
    return file_path


def plot_correlation_heatmap(df, out_path, run_id, title="Correlation Heatmap "):
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        raise ValueError("No numeric columns found in the DataFrame for correlation heatmap.")

    corr = numeric_df.corr()
    # if row_id drop it from the correlation matrix
    if "row_id" in corr.columns:
        corr = corr.drop("row_id", axis=0).drop("row_id", axis=1)  # both axis
    title = title.replace(" ", "_").lower()
    sns.heatmap(data=corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    file_path = out_path / f"{title}_{run_id}.png"
    plt.savefig(file_path)
    print(f"{title} saved to: {file_path}")
    plt.close()
    return file_path
