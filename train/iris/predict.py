from random import sample
import mlflow.pyfunc
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")


# Set the specific model version to use:
# You can also use a specific version:
# model_uri = "models:/iris-random-forest/1"
model_uri = "models:/iris-random-forest/latest"


model = mlflow.pyfunc.load_model(model_uri)

sample = pd.DataFrame(
    [
        {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal lenght (cm)": 1.4,
            "petal width (cm)": 0.2,
        }
    ]
)

prediction = model.predict(sample)

print("Perdiction:", prediction)