import os
import sys
import logging
import requests
import json
from app import app

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path      = os.path.join("datasets",config['output_folder_path']) 
test_data_path        = os.path.join('datasets',config['test_data_path']) 
prod_deployment_path  = os.path.join('model',config['prod_deployment_path']) 
output_model_path     = os.path.join('model',config['output_model_path']) 





def test_prediction():
    #Call each API endpoint and store the responses
    logging.info("Calling /prediction")
    response1 = app.test_client().post('/prediction',json={'filepath': os.path.join(test_data_path, 'testdata.csv')})
    assert response1.status_code == 200

def test_scoring():
    logging.info("Calling /scoring")
    response2 = app.test_client().get('/scoring')#put an API call here
    assert response2.status_code == 200

def test_summarystats():
    logging.info("Calling /summarystats")
    response3 = app.test_client().get('/summarystats')#put an API call here
    assert response3.status_code == 200

def test_diagnostics():
    
    logging.info("Calling /diagnostics")
    response4 = app.test_client().get('/diagnostics')#put an API call here
    assert response4.status_code == 200
