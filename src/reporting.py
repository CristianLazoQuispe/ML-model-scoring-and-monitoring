import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import logging
import sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path      = os.path.join("datasets",config['output_folder_path']) 
test_data_path        = os.path.join('datasets',config['test_data_path']) 
prod_deployment_path  = os.path.join('model',config['prod_deployment_path']) 
output_model_path     = os.path.join('model',config['output_model_path']) 



##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    
    logging.info("Load finaldata.csv")
    data_df = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    data_y = data_df['exited']
    data_X = data_df.drop(['corporation','exited'], axis=1)

    logging.info("Loading deployed model")
    model = pickle.load(open(os.path.join(prod_deployment_path,'trainedmodel.pkl'),'rb'))

    logging.info("Running predictions on data")
    prediction = model.predict(data_X)

    logging.info("Plot confusion matrix")
    logging.info("Saving in confusion_matrix.png")
    cm= confusion_matrix(data_y, prediction)
    #Normalized confusion matrix
    #Divide each row element by the sum of the entire row
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #Both matrices side by side to see the difference

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
    sns.heatmap(cm, annot=True, ax = ax1, cmap='Blues')
    ax1.set_title('Unnormalized confusion matrix')
    ax1.set_xlabel('PREDICTED VALUES')
    ax1.set_ylabel('ACTUAL VALUES')

    sns.heatmap(cm_normalized, annot=True, ax = ax2, cmap='Blues')
    ax2.set_title('Normalized confusion matrix')
    ax2.set_xlabel('PREDICTED VALUES')
    ax2.set_ylabel('ACTUAL VALUES')
    plt.show()

    plt.savefig(os.path.join(output_model_path,"confusion_matrix.png"))





if __name__ == '__main__':
    score_model()
