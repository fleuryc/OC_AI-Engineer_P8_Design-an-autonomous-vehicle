FROM tensorflow/tensorflow:latest-gpu

# required for mounting : https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.file_dataset.filedataset?view=azure-ml-py#azureml-data-file-dataset-filedataset-mount
RUN apt-get install -y libfuse2

# pip requirements
RUN pip install --upgrade pip requests[socks] \
    azureml-defaults azureml-mlflow mlflow numpy matplotlib pandas
