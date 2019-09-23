# Apprenticeship learning using Inverse Reinforcement Learning

## Introduction 
Reinforcement learning (RL) is is the very basic and most intuitive form of trial and error learning. Often referred to as learning by exploration, that is by taking random actions initially and then slowly figuring out the actions which lead to the desired motion. In any RL setting, to make the agent perform different behaviors, it is the reward structure that one must modify/exploit. But assume we only have the knowledge of the behavior of the expert with us, then how do we estimate the reward structure given a particular behavior in the environment? This is the problem of Inverse Reinforcement Learning (IRL), where given the optimal/sub-optimal expert policy, we wish to determine the underlying reward structure.

## Setup 
In our case, we assume the actions taken by the humans are sub-optimal and we need the RL agent to take better actions. Apprenticeship Learning via IRL will try to inder the goal of a teacher. It will learn a reward function from observations, which can then be used for reinforcement learning. If it discovers that the goal is to hit a nail with a hammer, it will ignore blinks and scratches from the teacher, as they are irrelevant to the goal.

## Agent
An agent is the algorithm which will make the decisions given a set of observations. 

## Sensors
An agent will be equipped with sensing capabilities to collect raw data (409 in our case). 

## State Space
The state of the agent consists of 409 observable features.

## Rewards
The reward after every decision is calculated as a weighted linear combination of the feature values observed in that frame. Here the reward rt in the tth frame, is calculated by the dot product of the weight vector w with the vector of feature values in tth frame, that is the state vector φt. Such that rt =wT ∗φt.

## Inverse RL
- item1 
- item 2
