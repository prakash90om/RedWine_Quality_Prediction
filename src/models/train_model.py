import os
import io
import pickle
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import logging
import matplotlib.pyplot as plt


# Set random seed
seed = 42

def eval_metrics(model, x, y):
    l, acc = model.evaluate(x,y)
    return l,acc

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

loss, accuracy = eval_metrics(regr, X_test, Y_test)
		
# Now print to file
with open("reports/metrics.json", 'w') as outfile:
        json.dump({ "Loss": loss, "Accuracy": accuracy}, outfile)

plt.bar(["Accuracy","Loss"],[accuracy,loss])
plt.title("Model Evaluation Metrics")
plt.savefig("reports/metrics.png")

logger = logging.getLogger(__name__)
logger.info('model Generated.')
logger.info(f'Loss : {loss} , Accuracy : {accuracy}')



