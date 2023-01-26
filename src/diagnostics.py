
import pandas as pd
import numpy as np
import timeit
import os
import json
import logging
import sys
import pickle
from datetime import datetime
import subprocess

from outdated import check_outdated

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join("datasets",config['output_folder_path']) 
test_data_path   = os.path.join("datasets",config['test_data_path']) 
prod_deployment_path = os.path.join('model',config['prod_deployment_path']) 

 
##################Function to get model predictions
def model_predictions(data:pd.DataFrame) -> pd.DataFrame:
    """
    Predictions using data and trainedmodel

    Args:
        data (pandas.DataFrame): Dataframe with features

    Returns:
        prediction: Model predictions
    """
    #read the deployed model and a test dataset, calculate predictions

    logging.info("Loading deployed model")
    model = pickle.load(open(os.path.join(prod_deployment_path,'trainedmodel.pkl'),'rb'))

    logging.info("Running predictions on data")
    prediction = model.predict(data)
    return prediction

##################Function to get summary statistics
def dataframe_summary():
    """
    Calculate statisctis from finaldata

    Returns:
        list[dict]: Columns name, mean, median and std
    """
    #calculate summary statistics here
    #return value should be a list containing all summary statistics
    
    logging.info("Load finaldata.csv")
    data_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    data_df = data_df.drop(['exited'], axis=1)
    data_df_num = data_df.select_dtypes('number')

    logging.info("Compute mean, median, and std")
    logging.info("Saving summary_statistics")
    logging.info("Saving in 'summary_statistics.txt")
    statistics_dict = {}

    with open(os.path.join(dataset_csv_path, 'summary_statistics.txt'), "w") as file:
        file.write(f"summary_statistics date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write("--------------------------------------------------\n")
        for col in data_df_num.columns:
            std    = data_df_num[col].std()
            median = data_df_num[col].median()
            mean   = data_df_num[col].mean()
            statistics_dict[col] = {'mean': mean, 'median': median, 'std': std}

            file.write("mean={:5.4f} median={:5.4f} std={:5.4f}".format(mean,median,std))
            file.write("\n")
    return statistics_dict
##################Function to analyze missing data
def missing_data():
    """
    Calculate missing values

    Returns:
        None
    """
    #calculate summary statistics here
    #return value should be a list containing all summary statistics
    
    logging.info("Load finaldata.csv")
    data_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))

    logging.info("Count missing values")
    logging.info("Saving missing_values")
    logging.info("Saving in 'missing_values.txt")
    missing_dict = {}
    with open(os.path.join(dataset_csv_path, 'missing_values.txt'), "w") as file:
        file.write(f"summary_statistics date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write("--------------------------------------------------\n")
        for col in data_df.columns:
            total = len(data_df[col].index)
            nans = data_df[col].isna().sum()
            percentage = nans/total
            missing_dict[col] = {'total': total, 'nans': nans, 'percentage': percentage}

            file.write("{:20s} n_total={:5.4f} n_NA={:5.4f} n_percentage={:5.4f}".format(col,total,nans,percentage))
            file.write("\n")
    return missing_dict

##################Function to get timings

def timing_ingestion()->float:
    """
    Timing ingestion process
    Returns:
        float: running time
    """
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'ingestion.py'], capture_output=True)
    timing = timeit.default_timer() - starttime
    return timing

def timing_training()->float:
    """
    Timing traning process
    Returns:
        float: running time
    """
    starttime = timeit.default_timer()
    _ = subprocess.run(['python', 'traning.py'], capture_output=True)
    timing = timeit.default_timer() - starttime
    return timing

def execution_time()->list:
    """
    Calculate time of each process in the ML pipeline
    """
    #calculate timing of training.py and ingestion.py
    #return a list of 2 timing values in seconds
    ingestion_time = timing_ingestion()
    logging.info("Time  ingestion process : {:5.4f}s".format(ingestion_time))
    training_time = timing_training()
    logging.info("Time  training  process : {:5.4f}s".format(training_time))
    
    return [ingestion_time,training_time]
    
##################Function to check dependencies
def outdated_packages_list():
    """
    Analyze package versions
    """
    #get a list of 
    logging.info("Analyzing dependencies")
    logging.info("Saving in 'dependencies_analysis.txt")
    dependencies_dict = {}
    with open(os.path.join(dataset_csv_path, 'dependencies_analysis.txt'), "w") as file:
        file.write(f"dependencies_dict date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write("--------------------------------------------------\n")
        with open('requirements.txt') as f:
            lines = f.readlines()

            for line in lines:
                package_name = line.split("==")[0].strip()
                version      = line.split("==")[1].strip()
                is_outdated, latest_version = check_outdated(package_name,version)
                latest_version = latest_version.strip()
                latest_version = latest_version.rstrip()
                dependencies_dict[package_name] = {'version': version, 'latest_version': latest_version,'is_outdated':is_outdated}
                file.write("package_name={:20s} version={:10s} latest_version={:20s} is_outdated={:20s}".format(package_name,version,latest_version,str(is_outdated)))
                file.write("\n")
    return dependencies_dict

if __name__ == '__main__':
    
    logging.info("Load finaldata.csv")
    data_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    data_df = data_df.drop(['corporation','exited'], axis=1)

    predictions = model_predictions(data_df)
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()





    
