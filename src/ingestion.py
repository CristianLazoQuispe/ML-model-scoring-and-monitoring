"""
Author: Cristian Lazo Quispe
Date: 27th January, 2023
Ingestion data
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging
import sys

########### logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = os.path.join('datasets',config['input_folder_path'])
output_folder_path = os.path.join('datasets',config['output_folder_path'])



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    """
    Data ingestion functions and cleaning
    """
    df = pd.DataFrame()
    file_names = []

    logging.info(f"Reading files from {input_folder_path}")
    for file in os.listdir(input_folder_path):
        file_path = os.path.join(input_folder_path, file)
        if not '.csv' in file_path:
            continue
        df_tmp = pd.read_csv(file_path)

        file = os.path.join(*file_path.split(os.path.sep)[-3:])
        file_names.append(file)

        df = df.append(df_tmp, ignore_index=True)

    logging.info("Filter duplicates in data")
    df = df.drop_duplicates().reset_index(drop=1)

    logging.info("Saving metadata ingested")
    logging.info(os.path.join(output_folder_path, 'ingestedfiles.txt'))
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), "w") as file:
        file.write(
            f"Ingestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        file.write("\n".join(file_names))

    logging.info("Saving data ingested")
    logging.info(os.path.join(output_folder_path, 'finaldata.csv'))
    df.to_csv(os.path.join(output_folder_path, 'finaldata.csv'), index=False)



if __name__ == '__main__':
    merge_multiple_dataframe()
