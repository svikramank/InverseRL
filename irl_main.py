import numpy as np 
import pandas as pd 
import daytime
import re 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import timeit
import pickle 
import scipy 
from cvxopt import matrix, solvers 
import DQN
import play
import random


class IRL_Agent:
	
	def __init__(self, randomFE, expertFE, epsilon):
		self.randomPolicy = randomFE
		self.expertPolicy = expertFE
		self.epsilon = epsilon  #termination when t < 0.1
		self.randomT = np.linalg.norm(np.asarray(self.expertPolicy) - np.asarray(self.randomPolicy)) # Norm of diff in the expert and random 
		self.policiesFE = {self.randomT : self.randomPolicy} # storing the policies and their respective t values in a dictionary 
		print("Expert - Random at the Start (t) :: ", self.randomT)
		self.currentT = self.randomT
		self.minimumT = self.randomT



	def getRLAgentFE(self, W):			# get the feature expectations of a new policy using RL agent
		DQN.IRL_helper(W)				# train the agent and save the model in a file that we pass to play
		fe = play.play('model.h5', DQN.env, DQN.newdf, W)	# return the feature expectations by executing the learned policy
		return fe 



	def policyListUpdater(self, W): 	# add the policyFE list and differences
		tempFE = self.getRLAgentFE(W)	# get feature expectations of a new policy respective to the input weights
		hyperDistance = np.abs(np.sum(np.multiply(W, np.asarray(self.expertPolicy) - np.asarray(tempFE)), axis=0)) #hyperDistance = t
		self.policiesFE[hyperDistance] = tempFE
		return hyperDistance



	def optimization(self):
		m = len(self.expertPolicy)
		P = matrix(2.0*np.eye(m), tc ='d') #min ||w||
		q = matrix(np.zeros(m), tc='d')
		policyList = [self.expertPolicy]
		h_list = [1]
		
		for i in self.policiesFE.keys():
			policyList.append(self.policiesFE[i])
			h_list.append(1)

		policyMat = np.matrix(policyList)
		policyMat[0] = -1*policyMat[0]
		G = matrix(policyMat, tc='d')
		h = matrix(-np.asarray(h_list), tc='d')
		sol = solvers.qp(P, q, G, h)

		weights = np.squeeze(np.asarray(sol['x']))
		norm = np.linalg.norm(weights)
		weights = weights/norm

		return weights



	def optimalWeightFinder(self):
		f = open('weights-optimal.txt', 'w')

		while True:
			W = self.optimization() # Optimize to find new weights in the list of policies
			print("weights ::", W)
			f.write((str(W)))
			f.write('\n')
			print("The distances ::", self.policiesFE.keys())
			self.currentT = self.policyListUpdater(W)
			print("Current distance (t) is ::", self.currentT)
			if self.currentT <= self.epsilon:  # terminate if the point reached close enough
				break 
		f.close()
		return W




if __name__ == '__main__':

	epsilon = 0.1

	# Get the expert policy feature expectations
	df_subset = DQN.newdf.iloc[:2000, :]
	fe = np.zeros(len(df_subset.iloc[0,0]))
	for index, row in df_subset.iterrows():
		fe+= (0.9**index)*np.asarray(row.iloc[0])
	expertPolicyFE = fe

	mu, sigma = np.mean(expertPolicyFE), np.std(expertPolicyFE)	# Find the mean and std of expert policy FE

	randomPolicyFE = np.random.normal(mu, sigma, expertPolicyFE.shape[0]) # Sample the random policy FE from the distribution of expert policy to initialize 

	irl_learner = IRL_Agent(randomPolicyFE, expertPolicyFE, epsilon)

	print(irl_learner.optimalWeightFinder())
















































