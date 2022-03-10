# MLflow is only compatible with TensorFlow [2.0.0 , 2.7.0]
# FROM tensorflow/tensorflow:2.7.0-gpu
FROM nvcr.io/nvidia/tensorflow:22.02-tf2-py3

RUN apt-get update \
    # required for AzureML Dataset mounting
    # see : https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.file_dataset.filedataset?view=azure-ml-py#azureml-data-file-dataset-filedataset-mount
    && apt-get install -y libfuse2 \
    # required for Image Segmentation Keras
    # see : https://github.com/divamgupta/image-segmentation-keras#prerequisites
    && apt-get install -y libsm6 libxext6 libxrender-dev \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pip requirements
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        azureml-defaults \
        azureml-mlflow \
        requests[socks] \
        mlflow \
        numpy \
        pandas \
        pyspark \
        matplotlib \
        opencv-python-headless \
        albumentations \
    && pip cache purge
