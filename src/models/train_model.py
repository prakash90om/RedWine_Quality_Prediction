import os
import io
import pickle
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
import json
import logging
import matplotlib.pyplot as plt


# Set random seed
seed = 42

def eval_metrics(model, x, y):
    # Make predictions for the test set
    y_pred = model.predict(x)

    # Calculate accuracy score
    acc = accuracy_score(y, y_pred)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return acc, rmse 

################################
########## DATA PREP ###########
################################

# Load in the data
df = pd.read_csv("data/raw/winequality-red.csv")

# Split into train and test sections
y = df.pop("quality")
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=seed)

#################################
########## MODELLING ############
#################################

# Fit a model on the train section
regr = RandomForestRegressor(max_depth=5, random_state=seed)
regr.fit(X_train, y_train)

filename = 'models/model.h5'
pickle.dump(regr, open(filename, 'wb'))
logger = logging.getLogger(__name__)
logger.info('Model saved in the folder')

accuracy,rmse = eval_metrics(regr, X_test, y_test)
		
# Now print to file
with open("reports/metrics.json", 'w') as outfile:
        json.dump({ "RMSE": rmse, "Accuracy": accuracy}, outfile)

plt.bar(["Accuracy","RMSE"],[accuracy,rmse])
plt.title("Model Evaluation Metrics")
plt.savefig("reports/metrics.png")

logger = logging.getLogger(__name__)
logger.info('model Generated.')
logger.info(f'RMSE : {rmse} , Accuracy : {accuracy}')



