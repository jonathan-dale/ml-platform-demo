from json import load
from pyexpat import model
import mlflow
import mlflow.sklearn

from numpy.random import random_sample
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from sklearn.pipeline import _name_estimators

from train import y_pred


# config
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("sklearn-iris-demo")

RANDOM_STATE=32
EXPERIMENT="iris_pred"

# Hyperparams - adjust these between runs to see mlflow comparisons
params = {
    'n_estimators': 200,
    'max_depth': 3,
    'random_state': 32
}

def load_data():
    # the iris data built in with the 
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target.astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

def main():    
    X_train, X_test, y_train, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators = params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )
    
    mlflow.set_experiment(EXPERIMENT)
    
    with mlflow.start_run(run_name="random-forest-iris") as run:
        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("dataset", "sklearn.dataset.load_iris")
        
        # Train
        model.fit(X_train, y_train)
        
        # Perdict
        y_perdict = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1_score)
        
        # Infer input/output shcema
        signature = infer_signature(X_train, y_pred)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(5),
            registered_model_name="iris-random-forest",
        )
        
        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()