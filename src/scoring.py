from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging
import sys
from utils import compute_model_metrics,inference

logging.basicConfig(stream=sys.stdout, level=logging.INFO)




#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join('datasets',config['output_folder_path']) 
test_data_path   = os.path.join('datasets',config['test_data_path']) 
model_path       = os.path.join('model',config['output_model_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    """
    Measure the perfomance of trained model using testdata.csv
    """
    logging.info("Loading testdata.csv")
    data_test = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))

    logging.info("Loading trained model")
    model =  pickle.load(open(os.path.join(model_path,'trainedmodel.pkl'),'rb'))
    
    logging.info("Preparing test data")
    y_test = data_test['exited']
    X_test = data_test.drop(['corporation','exited'], axis=1)

    logging.info("Predicting test data")
    y_pred = model.predict(X_test)
    
    precision, recall, f1 = compute_model_metrics(y_test, y_pred)
    logging.info("Testing  metrics: precision={:1.4f} recall={:1.4f} f1={:1.4f}".format(precision, recall, f1))

    logging.info("Saving scores to text file")
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as file:
        file.write(f"f1 score  = {f1}")
        file.write("\n")
        file.write(f"precision = {precision}")
        file.write("\n")
        file.write(f"recall    = {recall}")
        file.write("\n")

    return f1,precision, recall

if __name__ == '__main__':
    logging.info("Scoring model")
    score_model()
