
[![Red Wine Quality Prediction](https://github.com/prakash90om/RedWine_Quality_Prediction/actions/workflows/python-cml.yml/badge.svg)](https://github.com/prakash90om/RedWine_Quality_Prediction/actions/workflows/python-cml.yml)

Red Wine Quality Prediction
==============================

Red Wine Quality Prediction Model. It predicts the quality of the Red Wine using [https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009] Kaggle Dataset. ML Model used for this prediction is RandomForestRegressor. Since, this is a regression task so, Metrics used are RMSE, MAE and R2 scores. The repository is configured with DVC and CML. The github actions will be carried out upon code push.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    |
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    |── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       │
       ├── data           <- Scripts to download or generate data
       │   └── make_dataset.py
       │
       ├── features       <- Scripts to turn raw data into features for modeling
       │   └── build_features.py
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions
       │   ├── predict_model.py
       │   └── train_model.py
       │
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py

Following is the file structure

## src -- Source code direcory
      data/make_dataset.py:     Preparing the Wine Quality Dataset. Get/Dowload dataset from Cloud storage (Google Drive).
      models/train_model.py:    Performing the training with RandomForest Regressor model.
      models/predict_model.py:  Performing prediction using the trained model in train stage.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
