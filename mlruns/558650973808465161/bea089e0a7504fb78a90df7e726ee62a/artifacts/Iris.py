import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

n_estimators = 100
max_depth = 5

mlflow.set_experiment("Iris")

# Start MLflow run
with mlflow.start_run() as run:
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log metrics and parameters
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")

    # Log model with signature and input example
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=X_train[:5],
        signature=signature,
    )

    # Log script if running as a file
    try:
        mlflow.log_artifact(__file__)
    except NameError:
        pass

    print(f"Model accuracy: {accuracy:.4f}")
    print(signature)
