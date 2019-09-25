import numpy as np 
import pandas as pd 
import daytime
import re 
import matplotlib.pyplot as plt 
import timeit
import pickle 
import scipy 
import random
from keras.models import load_model


def play(model_name, env, newdf, feature_weights):
	GAMMA = 0.9
	count=0

	#load the trained model 
	trained_model = load_model(model_name)


	#start with a random initial state
	state = random.choice(newdf.state) 
	featureExpectations = np.zeros(len(feature_weights))

	while True:
		count += 1

		# Choose an action
		action = trained_model.predict(np.asarray([np.asarray(state)]))[0]
		state, reward = env.step(state, action, feature_weights)

		if count > 100:
			featureExpectations += (GAMMA**(count-100))*state

		if count % 2000 == 0:
			print("Ending the trajectory")
			break 

	return featureExpectations