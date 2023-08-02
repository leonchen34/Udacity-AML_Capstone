from azureml.core.dataset import Dataset
import os
import shutil

def getTrainingDataset(ws):

    training_dataset_name = "airbnb_boston_training"
    try:
        train_ds = Dataset.get_by_name(workspace=ws,name=training_dataset_name)
        print("found existing traing dataset")
        return train_ds
    except:  
        datastore = ws.get_default_datastore()
        train_data_file = "train.csv"
        src_dir = "./data"
        target_path = "airbnb_boston"
    
        train_data_dir = "./tmp_dir"
        if os.path.exists(train_data_dir) == False:
            os.mkdir(train_data_dir)

        src_file_path = os.path.join(src_dir,train_data_file)
        dest = shutil.copy(src_file_path,train_data_dir)
        #print(os.listdir(train_data_dir))

        Dataset.File.upload_directory(train_data_dir,(datastore,target_path),
                              overwrite=True, show_progress=True)

        # Upload the training data as a tabular dataset for access during training on 
        # remote compute
        datastore_path = os.path.join(target_path,train_data_file)
        print("datastore train data path: ",datastore_path)
        train_ds = Dataset.Tabular.from_delimited_files(
                path=datastore.path(datastore_path)
        )
    
        #train_ds.to_pandas_dataframe().head()

        train_ds.register(workspace=ws,name=training_dataset_name)
        print("register training dataset")

        return train_ds

def getTestDataset(ws):

    test_dataset_name = "airbnb_boston_test"
    try:
        test_ds = Dataset.get_by_name(workspace=ws,name=test_dataset_name)
        print("found existing test dataset")
        return test_ds
    except:  
        datastore = ws.get_default_datastore()
        test_data_file = "test.csv"
        src_dir = "./data"
        target_path = "airbnb_boston"
    
        test_data_dir = "./tmp_dir"
        if os.path.exists(test_data_dir) == False:
            os.mkdir(test_data_dir)

        src_file_path = os.path.join(src_dir,test_data_file)
        dest = shutil.copy(src_file_path,test_data_dir)
        #print(os.listdir(test_data_dir))

        Dataset.File.upload_directory(test_data_dir,(datastore,target_path),
                              overwrite=True, show_progress=True)

        # Upload the test data as a tabular dataset
        datastore_path = os.path.join(target_path,test_data_file)
        print("datastore test data path: ",datastore_path)
        test_ds = Dataset.Tabular.from_delimited_files(
                path=datastore.path(datastore_path)
        )
    
        #test_ds.to_pandas_dataframe().head()

        test_ds.register(workspace=ws,name=test_dataset_name)

        return test_ds
