# Script to train machine learning model.
# Add the necessary imports for the starter code.
import pandas as pd
import logging
import json
import numpy as np
from joblib import dump, load
import os

from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics

# Add code to load in the data.
def data_download(path):
    data = pd.read_csv(f"{path}/data/census.csv")
    return data

# Optional enhancement, use K-fold cross validation instead of a train-test split.
def data_split(data, cat_features, path):
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # save OneHotEncoder and LabelBinarizer under 'model' folder
    dump(encoder, f"{path}/model/encoder.joblib")
    dump(lb, f"{path}/model/lb.joblib")

    return X_train, y_train, X_test, y_test

# Train and save a model.
def train_save_model(X_train, y_train, path):
    model = train_model(X_train, y_train)

    # create a model file and save it under 'model' folder
    dump(model, f"{path}/model/RandomForest.joblib")
    
# Check metrics of the model
def check_metrics(X_test, y_test, path):
    model = load(f"{path}/model/RandomForest.joblib")
    precision, recall, fbeta = compute_model_metrics(y_test, model.predict(X_test))
    print(f"precision score: {precision:.2f}, recall score: {recall:.2f}, fbeta score: {fbeta:.2f}")

def check_sliced_data_metrics(data, cat_features, path):
    train, test = train_test_split(data, test_size=0.20)

    model = load(f"{path}/model/RandomForest.joblib")
    encoder = load(f"{path}/model/encoder.joblib")
    lb = load(f"{path}/model/lb.joblib")

    with open(f'{path}/model/slice_output.txt', 'w') as file:
        for feature in cat_features:
            for distinct_val in test[feature].unique():
                sliced_df = test[test[feature] == distinct_val]

                sliced_X_test, sliced_y_test, encoder, lb = process_data(
                    sliced_df, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

                precision, recall, fbeta = compute_model_metrics(sliced_y_test, model.predict(sliced_X_test))
                metrics = f"[feature:{feature}][value:{distinct_val}]| precision: {precision:.2f}, recall: {recall:.2f}, fbeta: {fbeta:.2f}"
                file.write(metrics + '\n')
    


