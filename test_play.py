import numpy as np 
import pandas as pd 
import daytime
import re 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from collections import deque
import random
from keras.models import load_model
import data_process


####################################################
########## IMPORT THE PROCESSED DATA ###############
####################################################
newdf = data_process.data_processing()

def create_data(newdf):
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
	return newdf

newdf = create_data(newdf)




##################################################################
################ CREATING THE ENVIRONMENT CLASS ##################
##################################################################
class Environment:

	def __init__(self, transition_model):
		self.transition_model = transition_model


	def step(self, state, action, wt):
		self.current_state = state
		self.current_action = action 
		self.W = wt

		if isinstance(self.current_state, list):
			self.current_state = self.current_state
		else:
			self.current_state = list(self.current_state)

		if isinstance(self.current_action, list):
			self.current_action = self.current_action
		else:
			self.current_action = list(self.current_action)

		stack_ = [self.current_state, [self.current_action]]
		stack_ = reduce(operator.concat, stack_)
		stack_ = np.asarray(stack_)
		a = [stack_]
		a = np.asarray(a)
		self.next_state = self.transition_model.predict(a)
		self.reward = np.sum(np.multiply(self.W, self.next_state.tolist()[0]), axis=0)
		return self.next_state[0], self.reward



def load_trained_weights(feature_weights):
	#load the weights for reward function
	f = open(feature_weights, 'r')
	if f.mode == 'r':
		trained_weights = f.read()

	trained_weights = trained_weights.split()
	tw = []
	print(len(trained_weights))
	for i in range(1,len(trained_weights)):
		print(i)
		if i == (len(trained_weights)-1):
			temp = trained_weights[i]
			temp = temp.replace(']', '')
			print(temp)
			tw.append(float(temp))
		else:
			tw.append(float(trained_weights[i]))

	return tw


def test_play(model_name, env, newdf, feature_weights):
	GAMMA = 0.9
	count=0

	#load the trained model 
	trained_model = load_model(model_name)

	#load trained weights
	trained_weights = load_trained_weights(feature_weights)

	#start with a random initial state
	state = random.choice(newdf.state) 
	featureExpectations = np.zeros(len(trained_weights))
	reward_list = []
	while True:
		count += 1

		# Choose an action
		action = trained_model.predict(np.asarray([np.asarray(state)]))[0]
		print("old state:", state)
		next_state, reward = env.step(state, action, trained_weights)
		print("new state", next_state)
		print("***********")
		if count > 100:
			featureExpectations += (GAMMA**(count-100))*next_state
			reward_list.append(reward)

		if count % 2000 == 0:
			print("Ending the trajectory...")
			mean_reward = np.mean(reward_list)
			SD_reward = np.std(reward_list)
			break 
		state = next_state


	return featureExpectations, mean_reward, SD_reward



if __name__ == '__main__':


	# load the transition model
	transition_model = load_model('transition_model.h5')

	# Create the environment
	env = Environment(transition_model)

	print("###########################################")
	print("### STARTING TO TEST THE TRAINED POLICY ###")
	print("###########################################")
	print(" ")
	print(" ")
	print(" *** This policy will return featureExpectations of the new trained policy along with mean and standard deviation of reward values for a trajectory of length 2000 ***")
	FE, mean_reward, SD_reward = test_play('model.h5', env, newdf, 'weights-optimal.txt')
	print(" ")
	print("The feature expectation of trained policy is: ", FE)
	print(" ")
	print("Mean reward obtained by this policy over a trajectory of 2000 steps is %s and SD is %s"%(mean_reward, SD_reward))
	print(" ")
	























