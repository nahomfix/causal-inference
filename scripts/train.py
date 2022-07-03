from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    plot_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from logger import Logger

# enable autologging
# mlflow.sklearn.autolog()

selected_features = [
    "area_se",
    "area_worst",
    "concavity_mean",
    "radius_se",
    "symmetry_worst",
    "texture_worst",
]


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

    logger = Logger("train").get_app_logger()

    logger.info("Training model...")

    standard_scaler = StandardScaler()

    X = clean_df.iloc[:, 2:]
    y = clean_df["diagnosis"]
    X_selected = clean_df.loc[:, selected_features]
    y_selected = y

    X_train, X_test, y_train, y_test = split_dataset(X, y, 0.2)
    (
        X_train_selected,
        X_test_selected,
        y_train_selected,
        y_test_selected,
    ) = split_dataset(X_selected, y_selected, 0.2)

    X_train = standard_scaler.fit_transform(X_train)
    X_test = standard_scaler.transform(X_test)

    X_train_selected = standard_scaler.fit_transform(X_train_selected)
    X_test_selected = standard_scaler.transform(X_test_selected)

    logreg_clf = LogisticRegression()

    with mlflow.start_run():
        logreg_clf.fit(X_train, y_train)

        # metrics = mlflow.sklearn.eval_and_log_metrics(
        #     logreg_clf, X_test, y_test, prefix="validation_"
        # )

        logger.info("Trained model successfully!")

        logger.info(
            "Model saved in run %s" % mlflow.active_run().info.run_uuid
        )

        y_pred = logreg_clf.predict(X_test)

        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
        precision = precision_score(
            y_true=y_test, y_pred=y_pred, average="weighted"
        )
        recall = recall_score(y_true=y_test, y_pred=y_pred, average="weighted")
        f1 = f1_score(y_true=y_test, y_pred=y_pred, average="weighted")

        clf_report = classification_report(y_true=y_test, y_pred=y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # mlflow.sklearn.log_model(logreg_clf, "logistic-model")

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

        # train model with selected features

        logreg_clf.fit(X_train_selected, y_train_selected)

        # metrics = mlflow.sklearn.eval_and_log_metrics(
        #     logreg_clf, X_test, y_test, prefix="validation_"
        # )

        logger.info("Trained model successfully!")

        logger.info(
            "Model saved in run %s" % mlflow.active_run().info.run_uuid
        )

        y_pred_selected = logreg_clf.predict(X_test_selected)

        accuracy_selected = accuracy_score(
            y_true=y_test_selected, y_pred=y_pred_selected
        )
        precision_selected = precision_score(
            y_true=y_test_selected, y_pred=y_pred_selected, average="weighted"
        )
        recall_selected = recall_score(
            y_true=y_test_selected, y_pred=y_pred_selected, average="weighted"
        )
        f1_selected = f1_score(
            y_true=y_test_selected, y_pred=y_pred_selected, average="weighted"
        )

        clf_report_selected = classification_report(
            y_true=y_test_selected, y_pred=y_pred_selected
        )

        mlflow.log_metric("accuracy", accuracy_selected)
        mlflow.log_metric("precision", precision_selected)
        mlflow.log_metric("recall", recall_selected)
        mlflow.log_metric("f1", f1_selected)

        # mlflow.sklearn.log_model(logreg_clf, "logistic-model")

        plot_confusion_matrix(
            logreg_clf,
            X_test_selected,
            y_test_selected,
            normalize="true",
            cmap=plt.cm.Blues,
        )

        with open(metrics_dir / "metrics.txt", "a") as metrics_file:
            metrics_file.write(
                f"Accuracy: {accuracy_selected} \n\nClassification Report: \n{clf_report_selected} \n"
            )

        plt.savefig(metrics_dir / "metrics_plot_selected.png")
