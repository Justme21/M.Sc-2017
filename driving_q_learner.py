#!/usr/bin/python

import math
import numpy as np
import random


#Translates the numeric index into the corresponding
# action label
index_to_direc = ["L","F","R"]


class DrivingLearner():
    def __init__(self,look_back):

        #Numeric index (0-2) of the action chosen by the learner
        self.act_index = None
        #Current state as define in getState
        self.cur_state = []
        #Store a copy of the previous state so learning can occur    
        self.prev_state = []
        #Copy of previous cur_state used to create change variables
        self.state_copy = []
        
        #Lists that contain the last "look_back" actions that occurred
        #and the last "look_back" actions that occurred before the previous iteration
        self.prev_acts = []
        self.cur_acts = []

        #e for epsilon-greedy action selection
        self.e = .05
        #gamma = look ahead parameter. Want as high as possible
        self.gam = .9999
        #alpha = learning parameter, how much the current signal affects value function
        self.alph = 1

        #Reward signal
        self.reward = 0
        #Cumulative reward achieved over an episode. Stored for plotting purposes
        self.total_reward = 0
        #List of total rewards achieved. Used to plot learning over time
        self.reward_list = []

        #Used to specify how many previous actions to look at
        self.look_back = look_back
      
        #Dictionary to store the q-values of the state-space
        self.q_values = {}

        self.random_choice = 0


    def getState(self,cur_state):
        """Given input in the form of output generated by the markov model, generates
           state classification using assignment rules"""
        state = []

        #self.state_copy is [] only at the start of a run
        #state_copy is used to hold the value of the state generated
        #in the last iteration
        if self.state_copy == []:
            self.state_copy = [0 for _ in range(3)]

        #First part of the state will be the last look_back-1 states we were in
        #state += list(self.prev_acts[1:])
        state += self.prev_acts[-1] #only include the last action performed. Less information than was used to generate the data       

        #Binary values for the entries in the state relating to acceleration  
        #Only use first 3 accelerations to reduce state space       
        for i in range(3):
            state.append(cur_state[i]>0)
        #state.append(cur_state[1]>0)

        #The values for car position and road width are all
        # in the 1.5-3 region
        for i in [6,8]:
            state.append(int(cur_state[i]))

        #Distance to car in front can be -1 which should be a separate state
        state.append(min(int(cur_state[9]),int(cur_state[9]/20))) #dist to car
        state.append(min(int(cur_state[10]),int(cur_state[10]/10))) #time to collision
        state.append(int(cur_state[11])) #number of cars on road
        state.append(int(cur_state[12]/40)) #GPS-speed

        #Relative variables
        for i in range(3):
            state.append((cur_state[i]-self.state_copy[i])>0)
        state.append((cur_state[1]-self.state_copy[1])>0)

        #Ratio of distance from left over distance from right
        #There are issues with this since the readings are approx
        #Might not be worth keeping
        z_t = cur_state[6]
        z_l = (cur_state[8]/2)+z_t-(self.car_width/2)
        z_r = (cur_state[8]/2)-z_t-(self.car_width/2)
        if z_l == 0: z_l = .001
        if z_r == 0: z_r = .001
        state.append(int((z_l/z_r)))
        
        #Return tuple since tuples can be used as keys for dictionaries
        return tuple(state)    


    def initialise(self,init_act_list):
        """Called to initialise the values for the learner at the start of each
           episode. Takes as input list of starting actions for the sequence (prior)"""

        #Store reward from previous episode and then reset all values
        self.reward_list.append(self.total_reward)
        self.reward = 0 
        self.total_reward = 0

        #Prev-acts stores the sequece of actions up to this point
        self.prev_acts = list(init_act_list)
        self.cur_acts = None #cur acts will store the acts including one chosesn in this it
        self.cur_state = None #Stores state as defined in getState
        self.prev_state = None #Stores last state for learning purposes
        self.state_copy = [] #Copy of state input used to generate relative state values

        #Hard-coded car width
        self.car_width = 1.75

        self.random_choice = 0
    
    def getMaxAction(self,cur_state,sample=True):
        """Given the current state as an input parameter returns the index thas has the
           highest q_value for the current state. If sample is true then epsilon-greedy
           sampling occurs in which case a random action will be chosen self.e of the time"""
        
        action = None
        top_q = None
        act = None

        if tuple(cur_state) in self.q_values.keys():
            act_list = self.q_values[tuple(cur_state)]
            #use np.random here so that when two actions have the same q-value the first
            # one in the list isn't always selected
            top_q = max(act_list)
            max_list = [i for i in range(len(act_list)) if act_list[i]==top_q] 
            act = np.random.choice(max_list)
        else:
            self.q_values[tuple(cur_state)] = [0,0,0]

        if act is None or (sample and random.random()<self.e):
            act = random.randint(0,2)
            self.random_choice += 1        

        return act

    #def move(self,act_seq,model):
    #    """Called during simulation. act_seq is a list of the last self.look_back actions that
    #       actually occurred. Calculates the reward that the learner's action choice should
    #       return"""
    #    prob_dict = model.getProb(''.join(self.prev_acts))
    #    if ''.join(act_seq) in prob_dict:
    #        # We want to incentivise the learner for selecting more probable actions
    #        action_prob = prob_dict[''.join(act_seq)]
    #        max_prob = max([prob_dict[x] for x in prob_dict if x != "count"])
    #        return 1-(max_prob-action_prob)
    #    else:
    #        #If the sequence does not have a probability assigned then it is not a legal
    #        # move and should be severely penalised
    #        return -1


    def act(self,model,learning=True):
        """Called during simulation. Returns the action that the learner chooses to perform"""

        #Get the index for the action
        #self.act_index = self.getMaxAction(self.prev_acts,sample=True)
        self.act_index = self.getMaxAction(self.prev_state,sample=True)

        #Assign corresponding label to index
        if self.act_index == 0:
            action = "L"
        elif self.act_index == 1:
            action = "F"
        else:
            action = "R"

        #Revise the definition of cur_acts to include the most recent action
        self.cur_acts = self.prev_acts[1:] + [action]
        #If we are not learning we don't want to calculate or store rewards
    #    if learning == True:
    #        self.reward = self.move(tuple(self.cur_acts),model) 
        return action


    def sense(self,state):
        """Takes the raw markov model output as input and from this generates the 
           state variable"""
        
        #If cur_state is None this is the first iteration, so there is no prev_state
        if self.cur_state == None:
            self.prev_state = []
        else:
            self.prev_state = tuple(self.cur_state)

        #Revise the definition of cur_state and then copy the input state
        self.cur_state = self.getState(state)
        self.state_copy = tuple(state)


    def getReward(self,action,true_action):
        reward = 0
        if action == true_action:
            reward += .5
            if true_action in ["L","R"]: reward += 8.5
        else:
            if action in ["L","R"] and true_action in ["L","R"]:
                reward -= 1.5
            else:
                reward -=.5

        return reward


    def learn(self,true_action):
        """Defines how learning is performed by the learner. In particular self.reward is 
           already initialised in move() where we reward the learner choosing the most
           probably action. Now if the action was also the correct action this is further
           rewarded. Due to the overwhelming majority "F" has, if the correctly predicted 
           action was a turn then we further reward this to incentivise selecting turns"""


        #This should be the only place in the program where reward is specified
        self.reward = self.getReward(index_to_direc[self.act_index],true_action)
        #The max possible reward that could have been attained is one that guessed the action correctly
        max_reward = self.getReward(true_action,true_action)
        #Scale the recorded reward plotted to make it more interpretible. 
        #1 is now the max that can be received in any iteration
        if max_reward != 0: 
            self.reward/=max_reward
     
        self.total_reward += self.reward
            
        
        #q_list = self.q_values[tuple(self.prev_acts)]
        q_list = self.q_values[tuple(self.prev_state)]

        #max_act = self.getMaxAction(self.cur_acts,sample=False)
        max_act = self.getMaxAction(self.cur_state,sample=False)
        
        #max_q = self.q_values[tuple(self.cur_acts)][max_act]
        max_q = self.q_values[tuple(self.cur_state)][max_act]

        q_list[self.act_index] += self.alph*(self.reward+self.gam*max_q - q_list[self.act_index])


    def callback(self,learn,episode,iteration,true_to_state):
        """Called at the end of each iteration. Resets values as required and manages 
           learning parameters."""
        #print("{}/{}: {}".format(episode,iteration+1,self.total_reward))
        self.prev_acts = list(true_to_state)
        if learn:
            self.e = 1.0/(episode)
            self.alph = .01 + 1/episode#.01 + 10.0/episode

        else:
            self.e = 0

    def getRules(self):
        """Used by the simulator to print out the rules learnt by the learner"""
        max_act = None
        rule_dict = {}
        for entry in self.q_values:
            max_act = np.argmax(self.q_values[entry])
            rule_dict[tuple(entry)] = index_to_direc[max_act]
        return rule_dict
