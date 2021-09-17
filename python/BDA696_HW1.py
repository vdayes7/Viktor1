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

import numpy
import pandas as pandas
import plotly.express as plotly
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# Get statistics using numpy
def print_statistics(num):
    print(
        " Statistics in order for Sepal Length(SL), Sepal Width(SW), Petal Length(PL) , and Petal Width(PW)"
    )
    print("               SL  SW  PL  PW ")
    print("min    =    ", numpy.min(num, axis=0))
    print("1st Quartile", numpy.quantile(num, 0.25, axis=0))
    av = numpy.around(
        numpy.mean(num, axis=0), decimals=1
    )  # Round to one decimal point to match other stats
    print("mean   =    ", av)
    print("3rd Quartile", numpy.quantile(num, 0.75, axis=0))
    print("max    =    ", numpy.max(num, axis=0))


# Iris plots using plotly
def do_plots(ply):
    fig = plotly.scatter(ply, x="petal_length", y="petal_width", color="species")
    fig.show()
    fig = plotly.violin(ply, x="sepal_width")  # Violin plot
    fig.show()
    fig = plotly.box(ply, y="petal_length")
    fig.show()
    fig = plotly.histogram(ply, x="sepal_length", nbins=60)
    fig.show()
    fig = plotly.density_contour(ply, x="petal_width", y="petal_length")
    fig.show()


# Classify data using Random Forest but first standardizing data with StandardScaler
# process is wrapped in pipe using Pipeline to created and test model
def random_forest(data):
    X = data.data
    y = data.target
    pipe = Pipeline([("scalar", StandardScaler()), ("rfc", RandomForestClassifier())])
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    print("Random Forest Accuracy = ", metrics.accuracy_score(y, y_pred))


# Use Decision Tree Classifier after first standardizing data with StandardScaler
# Wrap process in Pipeline, the create and test model
def decision_tree(data):
    X = data.data
    y = data.target
    pipe = Pipeline([("scalar", StandardScaler()), ("rfc", DecisionTreeClassifier())])
    pipe.fit(X, y)  # build model
    print(
        "Decision Tree Accuracy = ", metrics.accuracy_score(y, pipe.predict(X))
    )  # Run model, get accuracy


def main():

    data = datasets.load_iris()  # Load Fisher's Iris Dataset
    df = pandas.DataFrame(data.data)  # Load data into Pandas Dataframe
    print_statistics(df.to_numpy())
    do_plots(plotly.data.iris())
    random_forest(data)
    decision_tree(data)


if __name__ == "__main__":
    sys.exit(main())
