import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion


import logging
import sys
import os
import json
from sklearn.metrics import f1_score
import re
import pandas as pd
import subprocess

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path      = os.path.join("datasets",config['output_folder_path']) 
test_data_path        = os.path.join('datasets',config['test_data_path']) 
input_data_path       = os.path.join('datasets',config['input_folder_path']) 

prod_deployment_path  = os.path.join('model',config['prod_deployment_path']) 
output_model_path     = os.path.join('model',config['output_model_path']) 

def main():
    ##################Check and read new data
    #first, read ingestedfiles.txt
    # Check and read new data
    logging.info("Checking for new data")

    # First, read ingestedfiles.txt
    with open(os.path.join(prod_deployment_path, "ingestedfiles.txt")) as file:
        ingested_files = {line.strip('\n').split('/')[-1] for line in file.readlines()[1:]}


    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_files = set(os.listdir(input_data_path))
    print("source_files   :",source_files)

    print("ingested_files :",ingested_files)

    # Deciding whether to proceed, part 1
    # If you found new data, you should proceed. otherwise, do end the process
    # here
    if len(source_files.difference(ingested_files)) == 0:
        logging.info("No new data found")
        return None

    # Ingesting new data
    logging.info("Ingesting new data")
    ingestion.merge_multiple_dataframe()

    # Checking for model drift
    logging.info("Checking for model drift")

    # Check whether the score from the deployed model is different from the
    # score from the model that uses the newest ingested data
    logging.info(os.path.join(prod_deployment_path, "latestscore.txt"))
    with open(os.path.join(prod_deployment_path, "latestscore.txt")) as file:        
        deployed_score = re.findall(r'f1 score  = \d*\.?\d+',file.readlines()[0])
        deployed_score = deployed_score[0].split("= ")[1]
        deployed_score = float(deployed_score)
        

    data_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    y_df = data_df.pop('exited')
    X_df = data_df.drop(['corporation'], axis=1)

    y_pred = diagnostics.model_predictions(X_df)
    new_score = f1_score(y_df.values, y_pred)

    # Deciding whether to proceed, part 2
    logging.info(f"Deployed score = {deployed_score}")
    logging.info(f"New score = {new_score}")

    # If you found model drift, you should proceed. otherwise, do end the
    # process here
    if(new_score >= deployed_score):
        logging.info("No model drift occurred")
        return None

    # Re-training
    logging.info("Re-training model")
    training.train_model()
    logging.info("Re-scoring model")
    scoring.score_model()

    # Re-deployment
    # If you found evidence for model drift, re-run the deployment.py script
    logging.info("Re-deploying model")
    deployment.store_model_into_pickle()

    # Diagnostics and reporting
    logging.info("Running diagnostics and reporting")

    # Run diagnostic
    diagnostics.dataframe_summary()
    diagnostics.missing_data()
    diagnostics.execution_time()
    diagnostics.outdated_packages_list()
    reporting.score_model()

    output = subprocess.run(['python', 'src/apicalls.py','-v'],
                            capture_output=True).stdout
    print(output.decode())
    output = subprocess.run(['pytest', 'src/test_api.py','-v'],
                            capture_output=True).stdout
    print(output.decode())
if __name__ == '__main__':
    main()

