import pandas as pd 
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse
import os


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/sarankoundinya2000/MLentirepipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "sarankoundinya2000"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2430ca85e11f45f08480a346ab0359e44eb7e8b0"


#Load parameters from params.yaml

params = yaml.safe_load(open('params.yaml'))['train']

def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns = ["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/sarankoundinya2000/MLentirepipeline.mlflow")

    #loading the model from the disk 
    model  = pickle.load(open(model_path,'rb'))

    predicitions = model.predict(X)
    accuracy = accuracy_score(y,predictions)

    #log metrics to MLFlow

    mlflow.log_metric('accuracy',accuracy)
    print(f"Model accuracy:{accuracy}")


if __name__=='__main__':
    evaluate(params['data'],params['model'])