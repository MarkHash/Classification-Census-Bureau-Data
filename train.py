import os
import yaml
import logging
from starter.train_model import data_download, data_split, train_save_model, check_metrics

def go():
    logging.basicConfig(level=logging.INFO)
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        cat_features = cfg['data']['cat_features']

    root_path = os.getcwd()

    # donwload step
    logging.info("Downloading CSV data")
    data = data_download(root_path)

    # data splitting step
    logging.info("Splitting and processing data")
    X_train, y_train, X_test, y_test = data_split(data, cat_features, root_path)

    # training step
    logging.info("Training the model")
    train_save_model(X_train, y_train, root_path)

    # obtaining metric step
    logging.info("Checking metrics of the model")
    check_metrics(X_test, y_test, root_path)

if __name__ == "__main__":
    go()