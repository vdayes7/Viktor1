#! -*- coding: utf-8 -*-
"""
BDA 696 Homework Assignment 1 - Due Sept 17
Author Vince Dayes
Analysis of the classic Fisher's Iris Dataset
    - Get some simple summary statistics
    - Try 5 different plots of the data
    - Analyze and build models
        - Use the StandardScaler transformer
        - Fit the data against Random Forest classifier and Decision Tree classifier
        - Wrap the steps into a pipeline
Created Sept 12, 2021
"""

import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def main():

    data = datasets.load_iris()  # Load Fisher's Iris Dataset
    df = pd.DataFrame(data.data)  # Load data into Pandas Dataframe

    num = df.to_numpy()  # Get summary statistics using numpy

    print(
        " Statistics in order for Sepal Length(SL), Sepal Width(SW), Petal Length(PL) , and Petal Width(PW)"
    )
    print("               SL  SW  PL  PW ")
    print("min    =    ", np.min(num, axis=0))
    print("1st Quartile", np.quantile(num, 0.25, axis=0))
    av = np.around(
        np.mean(num, axis=0), decimals=1
    )  # Round to one decimal point to match other stats
    print("mean   =    ", av)
    print("3rd Quartile", np.quantile(num, 0.75, axis=0))
    print("max    =    ", np.max(num, axis=0))

    ply = px.data.iris()  # Do plots to visualize data (using plotly)

    fig = px.scatter(ply, x="petal_length", y="petal_width", color="species")
    fig.show()
    fig = px.violin(ply, x="sepal_width")  # Violin plot
    fig.show()
    fig = px.box(ply, y="petal_length")
    fig.show()
    fig = px.histogram(ply, x="sepal_length", nbins=60)
    fig.show()
    fig = px.density_contour(ply, x="petal_width", y="petal_length")
    fig.show()

    X = data.data  # Analyze and build models using scikit-learn
    y = data.target
    pipe = Pipeline([("scalar", StandardScaler()), ("rfc", RandomForestClassifier())])
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    print("Random Forest Accuracy = ", metrics.accuracy_score(y, y_pred))

    pipe = Pipeline([("scalar", StandardScaler()), ("rfc", DecisionTreeClassifier())])
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    print("Decision Tree Accuracy = ", metrics.accuracy_score(y, y_pred))


if __name__ == "__main__":
    sys.exit(main())
