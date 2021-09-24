import sys

import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_data(data_file):
    # Load Data
    iris_data = pd.read_csv("iris.data", header=None)
    # name columns
    iris_data.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]
    return iris_data


def summary(df):
    return df.describe()


def plot(df):
    fig_hist = px.histogram(
        df, x="sepal_length", color="class", opacity=0.75, barmode="overlay"
    )
    fig_hist.show()

    fig_scatter = px.scatter(df, x="sepal_length", y="sepal_width", color="class")
    fig_scatter.show()

    fig_violin = px.violin(df, y="sepal_length", color="class")
    fig_violin.show()

    fig_scat_3d = px.scatter_3d(
        df,
        x="sepal_length",
        y="sepal_width",
        z="petal_length",
        color="class",
    )
    fig_scat_3d.show()

    fig_box = px.box(df, y="petal_width", color="class")
    fig_box.show()
    return 0


def build_and_analyze_models(X_orig, y):
    scaler = StandardScaler()
    scaler.fit(X_orig)
    X = scaler.transform(X_orig)

    # Random Forest - No Pipeline
    random_forest = RandomForestClassifier(random_state=1234)
    random_forest.fit(X, y)

    prediction_rf = random_forest.predict(X)
    probability_rf = random_forest.predict_proba(X)

    print("Model Predictions")
    print(f"Classes: {random_forest.classes_}")
    print(f"Probability Random Forest: {probability_rf}")
    print(f"Predictions Random Forest: {prediction_rf}")
    accuracy_rf = accuracy_score(y, prediction_rf)
    print(f"Random Forest Accuracy(No Train/Test Split): {accuracy_rf}")

    # Decision Tree with Pipeline
    pipeline_DT = Pipeline(
        [
            ("Scaler", StandardScaler()),
            ("DecisionTree", DecisionTreeClassifier(random_state=1234)),
        ]
    )

    pipeline_DT.fit(X_orig, y)

    prediction_dt = pipeline_DT.predict(X_orig)
    # print(f"Probability Decision Tree: {probability}")
    # print(f"Predictions Decision Tree: {prediction}")
    accuracy_dt = accuracy_score(y, prediction_dt)
    print(f"Decision Tree Accuracy(No Train/Test Split): {accuracy_dt}")

    # KNeighbor with Pipeline
    pipeline_KN = Pipeline(
        [
            ("Scaler", StandardScaler()),
            ("KNeighbor", KNeighborsClassifier()),
        ]
    )

    pipeline_KN.fit(X_orig, y)

    prediction_kn = pipeline_KN.predict(X_orig)
    # print(f"Probability KNeighbor: {probability}")
    # print(f"Predictions KNeighbor: {prediction}")
    accuracy_kn = accuracy_score(y, prediction_kn)
    print(f"Decision Tree Accuracy(No Train/Test Split): {accuracy_kn}")

    return 0


def main():

    # Load Data
    iris_data = load_data("iris.data")

    # Summary statistics
    data_summary = summary(iris_data)
    print(data_summary)

    # Plot and display figures
    plot(iris_data)

    # Analyze and Build Models (Random Forest, Decision Tree, and KNeighbor)
    X_orig = iris_data[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ].values
    y = iris_data["class"].values

    build_and_analyze_models(X_orig, y)


if __name__ == "__main__":
    sys.exit(main())
