import numpy as np 
import pandas as pd 
import daytime
import re 
from scipy.stats.stats import pearsonr
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import timeit
import pickle 
import scipy 
from cvxopt import matrix, solvers 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


####################################################################################################################################
#################### WRITE A DUMMY SIMULATOR #######################################################################################
####################################################################################################################################

NUM_OF_EPOCHS = 5

def simulator(newdf):
	# This simulator basically takes input as a state s and action a and spits out the next state s' i.e the new state is 'a' is taken in state 's'. 
	print("Creating the (s, a, r, s') pairs...")
	ls = []
	l = len(newdf)
	for index, row in newdf.iterrows():
	    if index!= l-1:
	        ls.append(newdf.iloc[index+1, 0])
	    else:
	        break

	newdf.drop(newdf.tail(1).index,inplace=True)
	newdf['next_state'] = ls

	print("Printing the new dataframe with (s, a, r, s') trajectories...")
	print("----------")
	print(newdf.columns)

	# Creating the training data 
	X = []
	for index, rows in newdf.iterrows():
		r = [rows[0], [rows[1]]]
		r = sum(r, [])
		X.append(r)	
	y = newdf.iloc[:,3]

	X_stack, y_stack = np.stack(X, axis=0), np.stack(y, axis=0)

	print("Creating the model for simulator...")
	model = Sequential()
	model.add(Dense(30, input_dim=len(newdf.iloc[0,0]) + 1, activation= 'relu'))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(len(newdf.iloc[0,0])))
	model.compile(loss='mean_squared_error', optimizer= Adam(lr= 0.001))

	print("Started training...")
	model.fit(X_stack, y_stack, epochs=NUM_OF_EPOCHS, verbose=1)
	print("Simulator trained to predict s' from (s,a)...")

	return model