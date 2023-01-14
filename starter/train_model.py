# Script to train machine learning model.
# Add the necessary imports for the starter code.
import pandas_profiling
import pandas as pd
import logging
import json
import numpy as np
import pickle
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

from ml.data import process_data
from ml.model import train_model, compute_model_metrics

# Add code to load in the data.

data = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)

# create a model file and save it under 'model' folder
dir_path = r'../model/'
# with open(f"{dir_path}/RandomForest.pkl", 'wb') as file:
#     pickle.dump(model, file)

# with open(f"{dir_path}/encoder.joblib", 'wb') as encoder_file:
#     pickle.dump(encoder, encoder_file)

# with open(f"{dir_path}/lb.joblib", 'wb') as lb_file:
#     pickle.dump(encoder, lb_file)

dump(model, f"{dir_path}/RandomForest.joblib")
dump(encoder, f"{dir_path}/encoder.joblib")
dump(lb, f"{dir_path}/lb.joblib")

precision, recall, fbeta = compute_model_metrics(y_test, model.predict(X_test))
print(f"precision score: {precision}, recall score: {recall}, fbeta score: {fbeta}")