""" Compare multiple classifiers on Wine Quality, logging each as an MLFlow run. """


from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



EXPERIMENT = "wine-quality"
RANDOM_STATE = 43

# Each entry becomes its own MLflow run so we can compare them in the UI.
MODELS = {
    "random_forests": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators":[100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "max_features": ["sqrt", "log2"],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"],
            "class_weight": [None, "balanced"],
            "max_samples": [None, 0.5, 0.75, 1.0],
            "max_leaf_nodes": [None, 10, 20, 30],
            "min_impurity_decrease": [0.0, 0.1, 0.2],
            "min_impurity_split": [None, 0.1, 0.2],
            "random_state": [RANDOM_STATE],
        },
    },
    "gradient_boosting": {
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 4, 5],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0],
            "max_features": ["sqrt", "log2"],
            "loss": ["deviance", "exponential"],
            "random_state": [RANDOM_STATE],
        },
    },
    "logistic_regression": {
        "estimator": LogisticRegression(random_state=RANDOM_STATE),
        "params": {
            "C": [0.1, 1.0, 10.0],
            "max_iter": 2000,
            "solver": ["lbfgs", "liblinear", "sag", "saga"],
            "penalty": ["l1", "l2", "elasticnet"],
            "class_weight": [None, "balanced"],
            "random_state": [RANDOM_STATE],
        },
    },
    # "support_vector_machines": {
    #     "estimator": SVC(random_state=RANDOM_STATE),
    #     "params": {
    #         "C": [0.1, 1.0, 10.0],
    #         "kernel": ["linear", "rbf", "poly"],
    #         "gamma": ["scale", "auto"],
    #         "degree": [2, 3, 4],
    #         "coef0": [0.0, 0.1, 0.2],
    #         "shrinking": [True, False],
    #         "probability": [True, False],
    #         "random_state": [RANDOM_STATE],
    #     },
    # },
}

def load_data():
    wine = fetch_openml(name="wine-quality", version=1, as_frame=True)
    X, y = wine.data, wine.target.astype(int)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    
def save_confusion_matrix(y_true, y_pred, out_dir: Path, name: str) -> Path:
    """Render a confusion matrix PNG and return its path."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix for {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    path = out_dir / f"{name}_confusion_matrix.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path

def save_classification_report(y_true, y_pred, out_dir: Path, name: str) -> Path:
    """Render a classification report and return its path."""
    report = classification_report(y_true, y_pred, zero_division=0)
    path = out_dir / f"{name}_classification_report.txt"
    with open(path, "w") as f:
        f.write_text(report)
    return path


def train_and_evaluate_model(model_name: str, spec, X_train, X_test, y_train, y_test):
    """Set the estimator's params, then wrap in a scaling pipeline"""
    estimator = spec["estimator"].set_params(**spec["params"])
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_params(spec["params"])
        
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        
        # Log artifacts: put files in a temp dir, then log teh whole directory.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_confusion_matrix(y_test, preds, tmp_path, name=model_name)
            save_classification_report(y_test, preds, tmp_path, name=model_name)
            mlflow.log_artifacts(tmp, artifact_path="evaluation")
            
        mlflow.sklearn.log_model(pipe, name=model_name, imput_example=X_train.head(2))
        print(f"{model_name:>20s} acc={acc:.3f} macro-f1={f1:.3f}")


def main():
    X_train, X_test, y_train, y_test = load_data()
    mlflow.set_experiment(EXPERIMENT)
    for name, spec in MODELS.items():
        train_and_evaluate_model(name, spec, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()