""" Train a classifier on the Wine Quality data set and log to MLFlow.

The Wine Quality data set from OpenML includes:
    1600 samples
    11 real-valued physicochemical features
    Imbalanced multi-class target (quality scores 3-8), so accuracy alone is missleading
"""


# # macOS SSL workaround
# fetch_openml downloads from api.openml.org over HTTPS. On macOS, python.org Python
# installs often ship without trusted CA certificates, causing:
#   URLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
# One-time fix on Mac: 
#   /Applications/Python\ 3.14/Install\ Certificates.command
# We keep this block so the script works out of the box for others who skip that step.
import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXPERIMENT = "wine-quality"
RANDOM_STATE = 43

# Hyperparams - adjust these between runs to see mlflow comparisons
params = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 2,
}

def load_data():
    # version=1 is the red-wine variabt on OpenML
    wine = fetch_openml(name="wine-quality-red", version=1, as_frame=True)
    X, y = wine.data, wine.target.astype(int)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

def main():
    X_train, X_test, y_train, y_test = load_data()
    
    pipe = Pipeline([
      ("scaler", StandardScaler()),
      ("clf", RandomForestClassifier(random_state=RANDOM_STATE, **params)),
    ])
    
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("model", "RandomForestClassifier")
        
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        
        mlflow.sklearn.log_model(pipe, name="model", input_example=X_train.head(2))
        
        print(f"accuracy: {acc:.3f} | macro-f1: {f1:.3f}")
        print(classification_report(y_test, preds, zero_division=0))

if __name__ == "__main__":
    main()