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
import tensorflow as tf
from keras import losses
from keras.optimizers import Adam
from collections import deque
import data_process
import simulator
import random



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




#########################################################################
########## LEARN THE TRANSITION MODEL ON SUB OPTIMAL DATA ###############
#########################################################################
transition_model = simulator.simulator(newdf)




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

		stack_ = [self.current_state, self.current_action]
		stack_ = sum(stack_, [])
		stack_ = np.asarray(stack_)
		a = [stack_]
		a = np.asarray(a)
		self.next_state = self.transition_model.predict(a)
		self.reward = np.sum(np.multiply(self.W.tolist()[0], self.next_state.tolist()[0]), axis=0)
		return self.next_state[0], self.reward


# Create the environment
env = Environment(transition_model)



####################################################
################ WRITE THE DQN #####################
####################################################

STATE_SPACE_DIM = len(newdf.iloc[0,0]) 
RANDOM_REWARD = newdf.reward
TRAINING_FRAMES = 10000
UPDATE_TARGET_NETWORK = 50

class DQN:
	def __init__(self):
		self.gamma = 0.9
		self.epsilon = 1.0
		self.epsilon_min = 0.0
		self.epsilon_decay = 0.80
		self.tau = 0.125
		self.learning_rate = 0.0005
		self.memory = deque(maxlen=2000)
		self.model = self.create_model()
		self.target_model = self.create_model()


	# Create the neural network model to train the Q-func 
	def create_model(self):
		model = Sequential()
		model.add(Dense(30, input_dim= STATE_SPACE_DIM, activation= 'relu'))
		model.add(Dense(24, activation='relu'))
		model.add(Dense(10, activation='relu'))
		model.add(Dense(1))
		model.compile(loss='logcosh', optimizer=Adam(lr=self.learning_rate))
		return model 


	# Action function to choose the best action given the Q-function if not exploring based on epsilon
	def choose_action(self, state):
		self.epsilon *= self.epsilon_decay
		self.epsilon = max(self.epsilon_min, self.epsilon)
		r = np.random.random()
		if r < self.epsilon:
			print(" *** Choosing a random action ***")
			act = random.choice(RANDOM_REWARD)
			return [act]
		print(" *** Predicting a new action ***")
		state = np.asarray([np.asarray(state)])
		pred = self.model.predict(state)[0]

		return pred 


	# Create replay buffer memory to sample randomly
	def remember(self, state, action, reward, next_state):
		self.memory.append([state, action, reward, next_state])



	# Build the replay buffer
	def replay(self):
		batch_size = 32
		if len(self.memory) < batch_size:
			return
		samples = random.sample(self.memory, batch_size)
		for sample in samples:
			state, action, reward, new_state = sample
			state = np.asarray([np.asarray(state)])
			target = self.target_model.predict(state)

			new_state = np.asarray([np.asarray(new_state)])
			next_pred = self.target_model.predict(new_state)[0]

			Q_future = next_pred

			target[0] = reward + Q_future * self.gamma 
			self.model.fit(state, target, epochs=1, verbose=1)


	# Update our target network 
	def train_target(self):
		weights = self.model.get_weights()
		target_weights = self.target_model.get_weights()
		for i in range(len(target_weights)):
			target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
		self.target_model.set_weights(target_weights)


	# Save the model 
	def save_model(self, fn):
		self.model.save(fn)



def IRL_helper(feature_weights):

	# Create the DQN agent
	dqn_agent = DQN()

	#start with a random initial state
	state = random.choice(newdf.state) 
	t = 0
	# Start the training loop for DQN  
	while t < TRAINING_FRAMES:
		action = dqn_agent.choose_action(state)
		next_state, reward = env.step(state, action, feature_weights)
		dqn_agent.remember(state, action, reward, next_state)
		dqn_agent.replay()
		t += 1

		if t%UPDATE_TARGET_NETWORK == 0:
			dqn_agent.train_target()

		state = next_state
	dqn_agent.save_model("model.h5")















































