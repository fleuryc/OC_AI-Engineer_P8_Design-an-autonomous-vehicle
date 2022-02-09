[![Python application](https://github.com/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/actions/workflows/python-app.yml/badge.svg)](https://github.com/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/actions/workflows/python-app.yml)
[![CodeQL](https://github.com/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/actions/workflows/codeql-analysis.yml)
[![Codacy Security Scan](https://github.com/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/actions/workflows/codacy-analysis.yml/badge.svg)](https://github.com/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/actions/workflows/codacy-analysis.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/79783b97e49646b69d75353faf117832)](https://www.codacy.com/gh/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/dashboard)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/79783b97e49646b69d75353faf117832)](https://www.codacy.com/gh/fleuryc/OC_AI-Engineer_P8_Design-an-autonomous-vehicle/dashboard)

- [Project](#project)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Virtual environment](#virtual-environment)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
    - [Run Notebook](#run-notebook)
    - [Quality Assurance](#quality-assurance)
  - [Troubleshooting](#troubleshooting)

---

# Project

- What ?
- Why ?
- How ?

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
# pip install kaggle jupyterlab ipykernel ipywidgets widgetsnbextension graphviz python-dotenv requests matplotlib seaborn plotly shap numpy statsmodels pandas sklearn nltk gensim pyLDAvis spacy transformers tensorflow
# > or :
# pip install -r requirements.txt
# > or just :
make install
```

## Usage

### Run Notebook

```bash
jupyter-lab notebooks/main.ipynb
```

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
