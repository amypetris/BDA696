import sys

import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def main():

    # Load Data
    iris_data = pd.read_csv("iris.data")

    # Rename columns
    iris_data = iris_data.rename(
        columns={
            "5.1": "sepal_length",
            "3.5": "sepal_width",
            "1.4": "petal_length",
            "0.2": "petal_width",
        }
    )

    # Summary statistics
    summary = pd.DataFrame.describe(iris_data)
    print(summary)

    # Plot and display figures
    fig_hist = px.histogram(iris_data, x="sepal_length", color="Iris-setosa")
    fig_hist.show()

    fig_scatter = px.scatter(
        iris_data, x="sepal_length", y="sepal_width", color="Iris-setosa"
    )
    fig_scatter.show()

    fig_violin = px.violin(iris_data, y="sepal_length", color="Iris-setosa")
    fig_violin.show()

    fig_scat_3d = px.scatter_3d(
        iris_data,
        x="sepal_length",
        y="sepal_width",
        z="petal_length",
        color="Iris-setosa",
    )
    fig_scat_3d.show()

    fig_box = px.box(iris_data, y="petal_width", color="Iris-setosa")
    fig_box.show()

    # Analyze and Build Models (Random Forest, Decision Tree, and KNeighbor)
    X_orig = iris_data[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ].values
    y = iris_data["Iris-setosa"].values

    scaler = StandardScaler()
    scaler.fit(X_orig)
    X = scaler.transform(X_orig)

    random_forest = RandomForestClassifier(random_state=1234)
    random_forest.fit(X, y)

    prediction = random_forest.predict(X)
    probability = random_forest.predict_proba(X)

    print("Model Predictions")
    print(f"Classes: {random_forest.classes_}")
    print(f"Probability Random Forest: {probability}")
    print(f"Predictions Random Forest: {prediction}")

    pipeline_RF = Pipeline(
        [
            ("Scaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )

    pipeline_RF.fit(X_orig, y)

    probability = pipeline_RF.predict_proba(X_orig)
    prediction = pipeline_RF.predict(X_orig)
    print(f"Probability Random Forest: {probability}")
    print(f"Predictions Random Forest: {prediction}")

    pipeline_DT = Pipeline(
        [
            ("Scaler", StandardScaler()),
            ("DecisionTree", DecisionTreeClassifier(random_state=1234)),
        ]
    )

    pipeline_DT.fit(X_orig, y)

    probability = pipeline_DT.predict_proba(X_orig)
    prediction = pipeline_DT.predict(X_orig)
    print(f"Probability Decision Tree: {probability}")
    print(f"Predictions Decision Tree: {prediction}")

    pipeline_KN = Pipeline(
        [
            ("Scaler", StandardScaler()),
            ("KNeighbor", KNeighborsClassifier()),
        ]
    )

    pipeline_KN.fit(X_orig, y)

    probability = pipeline_KN.predict_proba(X_orig)
    prediction = pipeline_KN.predict(X_orig)
    print(f"Probability KNeighbor: {probability}")
    print(f"Predictions KNeighbor: {prediction}")


if __name__ == "__main__":
    sys.exit(main())
