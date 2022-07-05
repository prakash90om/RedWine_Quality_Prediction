import os
import io
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Set random seed
seed = 42

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

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

predicted_qualities = regr.predict(X_test)

(rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

		
# Now print to file
with open("reports/metrics.json", 'w') as outfile:
        json.dump({ "rmse": rmse, "mae": mae, "r2":r2}, outfile)


