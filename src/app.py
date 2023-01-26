from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics
import json
import os
import subprocess
import re

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)





######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path      = os.path.join("datasets",config['output_folder_path']) 
test_data_path        = os.path.join('datasets',config['test_data_path']) 
prod_deployment_path  = os.path.join('model',config['prod_deployment_path']) 
output_model_path     = os.path.join('model',config['output_model_path']) 

prediction_model = None


@app.route('/')
def index():
    return "Welcome to my final project!"

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    #add return value for prediction outputs
    filepath = request.get_json()['filepath']

    df = pd.read_csv(filepath)
    df = df.drop(['corporation', 'exited'], axis=1)

    preds = diagnostics.model_predictions(df)
    return jsonify(preds.tolist()) 



#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scor():        
    #check the score of the deployed model
    #add return value (a single F1 score number)
    output = subprocess.run(['python', 'src/scoring.py'],
                            capture_output=True).stdout
    print("output:",output)
    output = re.findall(r'f1=\d*\.?\d+', output.decode())[0]
    return output
#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column
     #return a list of all calculated summary statistics
    return jsonify(diagnostics.dataframe_summary())

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diag():        
    #check timing and percent NA values
    #add return value for all diagnostics
    missing_data  = diagnostics.missing_data()
    execution_time     = diagnostics.execution_time()
    packages_outdated = diagnostics.outdated_packages_list()

    ret = {
        'missing_percentage': missing_data,
        'execution_time': execution_time,
        'outdated_packages': packages_outdated
    }
    json_str = json.dumps(ret, cls=NpEncoder, indent=4)


    return json_str

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
