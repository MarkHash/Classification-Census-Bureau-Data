from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from starter.ml.data import process_data

import pickle
import json
import pandas as pd
import numpy as np
from joblib import load
import os


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference_preds(X, cat_features):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    model_file = os.path.join(os.getcwd(), 'model/RandomForest.joblib')
    encoder_file = os.path.join(os.getcwd(), 'model/encoder.joblib')
    lb_file = os.path.join(os.getcwd(), 'model/lb.joblib')

    model = load(model_file)
    encoder = load(encoder_file)
    lb = load(lb_file)

    X, _, _, _ = process_data(
        X, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )
    
    pred = model.predict(X)
    prediction = lb.inverse_transform(pred)[0]
    return prediction
