import os
import sys
import logging
import requests
import json
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path      = os.path.join("datasets",config['output_folder_path']) 
test_data_path        = os.path.join('datasets',config['test_data_path']) 
prod_deployment_path  = os.path.join('model',config['prod_deployment_path']) 
output_model_path     = os.path.join('model',config['output_model_path']) 



#Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:8000"



def test_prediction():
    #Call each API endpoint and store the responses
    logging.info("Calling /prediction")
    response1 = requests.post(f'{URL}/prediction',json={'filepath': os.path.join(test_data_path, 'testdata.csv')})#put an API call here
    assert response1.status_code == 200

def test_scoring():
    logging.info("Calling /scoring")
    response2 = requests.get(f'{URL}/scoring')#put an API call here
    assert response2.status_code == 200

def test_summarystats():
    logging.info("Calling /summarystats")
    response3 = requests.get(f'{URL}/summarystats')#put an API call here
    assert response3.status_code == 200

def test_diagnostics():
    
    logging.info("Calling /diagnostics")
    response4 = requests.get(f'{URL}/diagnostics')#put an API call here
    assert response4.status_code == 200
