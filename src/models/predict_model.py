import pickle
import logging
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import numpy as np



def predict_model(modelPath,data):
	""" 
	Predicts the Wine quality 
	"""

	# load the model from disk
	loaded_model = pickle.load(open(modelPath, 'rb'))

	logger = logging.getLogger(__name__)
	logger.info('model Loaded.')

	## predict data 
	pred = loaded_model.predict(data)
	logger.info(f'Predicted Value is {pred}')

	return pred


if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)
	logger = logging.getLogger(__name__)

	# Load in the data
	df = pd.read_csv("data/raw/winequality-red.csv")

	# Split into train and test sections
	y = df.pop("quality")
	X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

	predict_model("./models/model.h5",X_test[0:1]);

