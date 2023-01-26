from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import sys
import logging
import numpy as np
from utils import compute_model_metrics,inference
#######logging
# 
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join('datasets',config['output_folder_path']) 
model_path       = os.path.join('model',config['output_model_path']) 


#################Function for training the model
def train_model():
    
    #use this logistic regression for training
    model = LogisticRegression(multi_class='multinomial', solver='newton-cg',random_state=0)
    
    #fit the logistic regression to your data
    
    logging.info("Load finaldata.csv")
    final_data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    #write the trained model to your workspace in a file called trainedmodel.pkl
    data_y = final_data['exited']
    
    data_x = final_data.drop(['corporation','exited'], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.25, random_state=20)  

    logging.info("Training model")
    model.fit(X_train, y_train)
    
    logging.info("Analyzing performance")
    y_train_prediction = inference(model,X_train)
    y_val_prediction  = inference(model,X_val)
    
    metrics = compute_model_metrics(y_train_prediction, y_train)
    logging.info("Training metrics: precision={:1.4f} recall={:1.4f} f1={:1.4f}".format(metrics[0],metrics[1],metrics[2]))
    metrics = compute_model_metrics(y_val_prediction, y_val)
    logging.info("Validation  metrics: precision={:1.4f} recall={:1.4f} f1={:1.4f}".format(metrics[0],metrics[1],metrics[2]))


    logging.info("Saving trained model")
    pickle.dump(model,open(os.path.join(model_path,'trainedmodel.pkl'),'wb'))




if __name__ == '__main__':
    logging.info("Training logistic regression")
    train_model()

