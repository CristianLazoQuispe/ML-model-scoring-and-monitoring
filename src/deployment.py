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
import shutil

logging.basicConfig(stream=sys.stdout, level=logging.INFO)



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join('datasets',config['output_folder_path']) 
model_path       = os.path.join('model',config['output_model_path']) 
prod_deployment_path = os.path.join('model',config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    """
    Copy the latest model pickle file, the latestscore.txt value,
    and the ingestfiles.txt file into the deployment directory
    """
    logging.info("Deploying trained model to production")
    logging.info("Copying trainedmodel.pkl")
    shutil.copy(
        os.path.join(dataset_csv_path,'ingestedfiles.txt'),
        prod_deployment_path)
    logging.info("Copying ingestfiles.txt")
    shutil.copy(
        os.path.join(model_path,'trainedmodel.pkl'),
        prod_deployment_path)
    logging.info("Copying latestscore.txt")
    shutil.copy(
        os.path.join(model_path,'latestscore.txt'),
        prod_deployment_path)


if __name__ == '__main__':
    logging.info("Model Deployment")
    store_model_into_pickle()
