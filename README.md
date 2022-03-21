[![Python application](https://github.com/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/actions/workflows/python-app.yml/badge.svg)](https://github.com/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/actions/workflows/python-app.yml)
[![CodeQL](https://github.com/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/actions/workflows/codeql-analysis.yml)
[![Codacy Security Scan](https://github.com/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/actions/workflows/codacy-analysis.yml/badge.svg)](https://github.com/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/actions/workflows/codacy-analysis.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/79783b97e49646b69d75353faf117832)](https://www.codacy.com/gh/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/dashboard)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/79783b97e49646b69d75353faf117832)](https://www.codacy.com/gh/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/dashboard)

- [Future Vision Transport : Design an Autonomous Vehicle](#future-vision-transport--design-an-autonomous-vehicle)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Virtual environment](#virtual-environment)
    - [Dependencies](#dependencies)
    - [Environment variables](#environment-variables)
  - [Usage](#usage)
    - [Download data](#download-data)
    - [Run Notebook](#run-notebook)
    - [Quality Assurance](#quality-assurance)
  - [Troubleshooting](#troubleshooting)

---

# Future Vision Transport : Design an Autonomous Vehicle

Repository of OpenClassrooms' [AI Engineer path](https://openclassrooms.com/fr/paths/188-ingenieur-ia), project #8

Goal : use _Azure Machine Learning_ services, _Computer Vision_ techniques and _Deep Neural Network_ models, to perform urban street scene images segmentation.

You can see the results here :

- [Presentation](https://fleuryc.github.io/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/index.html "Presentation")
- [Notebook : HTML page with interactive plots](https://fleuryc.github.io/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/main.html "HTML page with interactive plots")

## Goals

- [x] Deep neural Network models training
  - Models : [DeepLab v3+](notebooks/azureml/models/deeplab_v3plus.py "DeepLab v3+"), [U-Net with Xception backbone](notebooks/azureml/models/unet_xception.py "U-Net with Xception backbone"), [FCN-8](notebooks/azureml/models/keras_segmentation/fcn.py "FCN-8")
  - Loss function : [Jaccard index](notebooks/azureml/cityscapes.py:201 "Jaccard index")
- Strategy : [Early Stopping](notebooks/azureml/train.py:237 "EarlyStopping") and [Reduce Learning Rate On Plateau](notebooks/azureml/train.py:230 "ReduceLROnPlateau")
- [x] Models evaluation
  - Metric : [Mean Intersection over Union (IoU)](notebooks/azureml/cityscapes.py:183 "MeanIoU")
  - Training time and cost : [AzureML Experiment](https://ml.azure.com/experiments/id/a2f53c2b-086c-46d4-876f-e302c35ca761?wsid=/subscriptions/da2e4791-6dd1-422b-848a-a961cef6ab89/resourceGroups/OC_P8/providers/Microsoft.MachineLearningServices/workspaces/oc-p8-ml-workspace&tid=43204f6d-c600-4585-985a-6bafda08d2bb "AzureML Experiment")
- [x] Image augmentation techniques
  - Use [Albumentations](notebooks/azureml/train.py:81 "Albumentations")
- [x] Handle a large dataset
  - Implement a [Data Generator](notebooks/azureml/cityscapes.py:122 "Data Generator")

## Installation

### Prerequisites

- [Python 3.9](https://www.python.org/downloads/)

### Virtual environment

```bash
# python -m venv env
# > or just :
make venv
source env/bin/activate
```

### Dependencies

```bash
# pip install jupyterlab ipykernel ipywidgets widgetsnbextension graphviz python-dotenv requests mlflow azureml-core azureml-defaults azureml-sdk azureml-dataset-runtime azureml-mlflow matplotlib numpy statsmodels pandas sklearn tensorflow pyspark opencv-python-headless albumentations Pillow
# > or :
# pip install -r requirements.txt
# > or just :
make install
```

### Environment variables

- Set environment variable values in [.env](.env) file.


### Azure resources

- AzureML [Workspace](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace#-create-a-workspace "Create a workspace")
- Azure [App Service](https://docs.microsoft.com/en-us/azure/app-service/quickstart-python?tabs=flask%2Cmac-linux%2Cazure-portal%2Cterminal-bash%2Cvscode-deploy%2Cdeploy-instructions-azportal%2Cdeploy-instructions-zip-azcli "Quickstart: Deploy a Python (Django or Flask) web app to Azure App Service")

## Usage

### Download data

Download, extract and upload to Azure Cityscape zip files.

```bash
make dataset
```

### Deploy webapp

Deploy the content of [webapp](./webapp "webapp") directory to Azure App Service.

### Run Notebooks

- [Main notebook](notebooks/main.ipynb "Main notebook")
- AzureML notebooks :
  - [Train models](notebooks/train.ipynb "Train models")
  - [Deploy models](notebooks/deploy.ipynb "Deploy models")
  - [Predict segmentation](notebooks/predict.ipynb "Predict segmentation")


### Quality Assurance

```bash
# make isort
# make format
# make lint
# make bandit
# make mypy
# make test
# > or just :
make qa
```

## Troubleshooting

- Fix Plotly issues with JupyterLab

cf. [Plotly troubleshooting](https://plotly.com/python/troubleshooting/#jupyterlab-problems)

```bash
jupyter labextension install jupyterlab-plotly
```

- If using Jupyter Notebook instead of JupyterLab, uncomment the following lines in the notebook

```python
import plotly.io as pio
pio.renderers.default='notebook'
```
