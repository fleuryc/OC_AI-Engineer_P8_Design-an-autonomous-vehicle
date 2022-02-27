# MLflow is only compatible with TensorFlow [2.0.0 , 2.7.0]
FROM tensorflow/tensorflow:2.7.0-gpu


# required for mounting : https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.file_dataset.filedataset?view=azure-ml-py#azureml-data-file-dataset-filedataset-mount
RUN apt-get update \
    && apt-get install -y libfuse2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pip requirements
RUN pip install --upgrade pip requests[socks] \
    azureml-defaults azureml-mlflow mlflow \
    numpy matplotlib pandas albumentations \
    && pip cache purge
