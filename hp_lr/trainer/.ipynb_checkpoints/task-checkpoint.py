from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from google.cloud import bigquery#
from google.cloud import storage
from joblib import dump

import os
import pandas as pd

#from xgboost import XGBClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import argparse
import hypertune
from sklearn.model_selection import train_test_split as tts



STAGE_DATA_BUCKET = 'chicago_taxi_stage'
TRAIN_DATA_PATH = 'chicago_taxi_train.csv'

cols = ['trip_month', 'trip_day', 'trip_day_of_week',
       'trip_hour', 'trip_seconds', 'trip_miles', 'euclidean', 'target',
       'payment_type_Credit_Card', 'payment_type_Dispute', 'payment_type_Mobile',
       'payment_type_No_Charge', 'payment_type_Prcard', 'payment_type_Unknown']

def get_args():
    '''Parses args. Must include all hyperparameters you want to tune.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--penalty',
      required=True,
      type=str,
      help='Penalty')
    parser.add_argument(
      '--C',
      required=True,
      type=float,
      help='Inverse of regularization')
    parser.add_argument(
      '--solver',
      required=True,
      type=str,
      help='Solver')
    args = parser.parse_args()
    return args




def create_dataset():
    
    bqclient = bigquery.Client()
    storage_client = storage.Client()

    gcsclient = storage.Client() # tal vez vaya stage_data_bucket
    bucket = gcsclient.get_bucket(STAGE_DATA_BUCKET)
    blob = bucket.blob(TRAIN_DATA_PATH)
    blob.download_to_filename(TRAIN_DATA_PATH)

    data = pd.read_csv(TRAIN_DATA_PATH, usecols=cols)
    # data = data.sample(frac = 0.2, random_state = 42)
    train_data, validation_data = tts(data, test_size=0.3)
    return train_data, validation_data

def split_data_and_labels(data):
    y = data.pop('target')
    return data, y

    

def create_model(penalty, C, solver):
    model = LogisticRegression(
        penalty = penalty,
        C = C,
        solver = solver
    )
    
    return model

def main():
    args = get_args()
    
    train_data, validation_data = create_dataset()
    x_train, y_train = split_data_and_labels(train_data)
    x_test, y_test = split_data_and_labels(validation_data)
    
    model = create_model(args.penalty, args.C, args.solver)
    model = model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    f1_value = f1_score(y_test, y_pred)
    
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='f1_score',
        metric_value=f1_value
    )


if __name__ == "__main__":
    main()
