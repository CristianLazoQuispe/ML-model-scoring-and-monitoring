# ML-model-scoring-and-monitoring


# Ingestion

    $ python src/ingestion.py 

            INFO:root:Reading files from datasets/sourcedata
            INFO:root:Filter duplicates in data
            INFO:root:Saving metadata ingested
            INFO:root:Saving data ingested

 
    $ python src/training.py 

            INFO:root:Training logistic regression
            INFO:root:Load finaldata.csv
            INFO:root:Training model
            INFO:root:Analyzing performance
            INFO:root:Training metrics: precision=0.9000 recall=0.6429 f1=0.7500
            INFO:root:Validation  metrics: precision=1.0000 recall=0.5000 f1=0.6667
            INFO:root:Saving trained model

    $ python src/scoring.py 

            INFO:root:Scoring model
            INFO:root:Loading testdata.csv
            INFO:root:Loading trained model
            INFO:root:Preparing test data
            INFO:root:Predicting test data
            INFO:root:Testing  metrics: precisio

    $ python src/deployment.py 

            INFO:root:Model Deployment
            INFO:root:Deploying trained model to production
            INFO:root:Copying trainedmodel.pkl
            INFO:root:Copying ingestfiles.txt
            INFO:root:Copying latestscore.txt

    $ python src/diagnostics.py


            INFO:root:Load finaldata.csv
            INFO:root:Loading deployed model
            INFO:root:Running predictions on data
            INFO:root:Load finaldata.csv
            INFO:root:Compute mean, median, and std
            INFO:root:Saving summary_statistics
            INFO:root:Saving in 'summary_statistics.txt
            INFO:root:Load finaldata.csv
            INFO:root:Count missing values
            INFO:root:Saving missing_values
            INFO:root:Saving in 'missing_values.txt
            INFO:root:Time  ingestion process : 0.0196s
            INFO:root:Time  training  process : 0.0196s
            INFO:root:Analyzing dependencies
            INFO:root:Saving in 'dependencies_analysis.txt


        Example of output saved in 

            dependencies_dict date: 26/01/2023 11:51:10
            --------------------------------------------------
            package_name=click                version=7.1.2      latest_version=8.1.3                is_outdated=True                
            package_name=cycler               version=0.10.0     latest_version=0.11.0               is_outdated=True                
            package_name=Flask                version=1.1.2      latest_version=2.2.2                is_outdated=True 


    $ python src/reporting.py

<img src = "model/practicemodels/confusion_matrix.png?raw=true" width = "900" height = "400" />
