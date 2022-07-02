from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    plot_confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_dataset(X: pd.DataFrame, y: pd.DataFrame, test_size: int) -> list:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    root_dir = Path().cwd()
    data_dir = root_dir / "data"
    metrics_dir = root_dir / "metrics"

    clean_df = pd.read_csv(data_dir / "data_clean_cml.csv")

    standard_scaler = StandardScaler()

    X = clean_df.iloc[:, 2:]
    y = clean_df["diagnosis"]

    X_train, X_test, y_train, y_test = split_dataset(X, y, 0.2)

    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)

    logreg_clf = LogisticRegression()

    logreg_clf.fit(X_train, y_train)

    y_pred = logreg_clf.predict(X_test)

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    clf_report = classification_report(y_true=y_test, y_pred=y_pred)

    # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)

    plot_confusion_matrix(
        logreg_clf, X_test, y_test, normalize="true", cmap=plt.cm.Blues
    )

    if not metrics_dir.exists():
        metrics_dir.mkdir()

    with open(metrics_dir / "metrics.txt", "w") as metrics_file:
        metrics_file.write(
            f"Accuracy: {accuracy} \n\nClassification Report: \n{clf_report} \n"
        )

    plt.savefig(metrics_dir / "metrics_plot.png")
