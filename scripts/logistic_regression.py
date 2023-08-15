from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
import joblib
import pandas as pd
from azureml.core.run import Run
from azureml.core.dataset import Dataset
#from azureml.data.dataset_factory import TabularDatasetFactory

def logisticReg():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--penalty', default="l2", help="penalty term")
    parser.add_argument('--solver', default="lbfgs", help="optimization algorithm")

    args = parser.parse_args()

    run = Run.get_context()
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    ws = run.experiment.workspace
    
    trainig_dataset_name = "airbnb_boston_training"
    training_ds = Dataset.get_by_name(workspace=ws,name=trainig_dataset_name)
    # training_ds = run.input_datasets[training_dataset_name]
    trainig_df = training_ds.to_pandas_dataframe()

    test_dataset_name = "airbnb_boston_test"
    test_ds = Dataset.get_by_name(workspace=ws,name=test_dataset_name)
    test_df = test_ds.to_pandas_dataframe()

    y_train = trainig_df['fraud']
    x_train = trainig_df.drop(['fraud'], axis=1)
    y_test = test_df['fraud']
    x_test = test_df.drop(['fraud'], axis=1)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter,  
            penalty=args.penalty, solver=args.solver).fit(x_train,y_train)
    
    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    # y_pred = model.predict(x_test)
    # accuracy = np.average(y_pred == y_test)

    # save the model
    os.makedirs('outputs/model',exist_ok=True)
    joblib.dump(value=model, filename='outputs/model/model.pkl')

    run.complete()

if __name__ == '__main__':
    logisticReg()
