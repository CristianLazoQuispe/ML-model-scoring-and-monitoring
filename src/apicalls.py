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



#Call each API endpoint and store the responses
logging.info("Calling /prediction")
response1 = requests.post(f'{URL}/prediction',json={'filepath': os.path.join(test_data_path, 'testdata.csv')})#put an API call here
response1 = response1.text
logging.info("Calling /scoring")
response2 = requests.get(f'{URL}/scoring')#put an API call here
response2 = response2.text
logging.info("Calling /summarystats")
response3 = requests.get(f'{URL}/summarystats')#put an API call here
response3 = response3.text
logging.info("Calling /diagnostics")
response4 = requests.get(f'{URL}/diagnostics')#put an API call here
response4 = response4.text

#combine all API responses
#write the responses to your workspace
logging.info("Generating report text file")
with open(os.path.join(output_model_path, 'apireturns.txt'), 'w') as file:
    file.write('Endpoint prediction\n')
    file.write(response1)
    file.write('\nEndpoint scoring\n')
    file.write(response2)
    file.write('\nEndpoint summarystats\n')
    file.write(response3)
    file.write('\nEndpoint diagnostics\n')
    file.write(response4)



