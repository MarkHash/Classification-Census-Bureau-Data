# Model Card
Mark Hashimoto created the model. It is randam forest classifier using the default hyperparameters in scikit-learn 1.1.3

## Model Details

## Intended Use
This model should be used to predict the salary based off a handful of attribues. The users are researchers.

## Training Data

The data was obtained from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income).
The original dataset has 32561 rows, and a 70-30 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features.
The dataset was reasonably cleaned using the following conditions:

- Age is between 17 and 99
- Hours of work is greater than 0

## Metrics
The model was evaluated the following metrics with scores.

- precision score: 0.7259148618371919
- recall score: 0.6094043887147336
- fbeta score: 0.6625766871165644

## Ethical Considerations
The data was acquired in 1994, which could be old for practical use.

## Caveats and Recommendations
The data is biased on gender, which means that the dataset is imbalanced.
