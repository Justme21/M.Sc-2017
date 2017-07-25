#!/usr/bin/python

import math
import numpy as np
import random


index_to_direc = ["L","F","R"]


class DrivingLearner():
    def __init__(self,look_back):
        self.act_index = None
        self.cur_state = []
        self.prev_state = []
        self.state_copy = []
        self.prev_acts = []
        self.cur_acts = []

        self.e = .05
        self.gam = .95#.9999
        self.alph = 1

        self.reward = 0
        self.total_reward = 0
        self.reward_list = []

        self.look_back = look_back
      

        self.q_values = {}


    def getState(self,cur_state):
        state = []

        if self.state_copy == []:
            self.state_copy = [0 for _ in range(6)]

        #First part of the state will be the last 4 states we were in
        state += list(self.prev_acts[1:])
       
        #Binary values for the entries in the state relating to acceleration         
        for i in range(6):
            state.append(cur_state[i]>0)

        #The values for car position and road width are all
        # in the 1.5-3 region
        for i in [6,8]:
            state.append(int(cur_state[i]/.5))

        #Distance to car in front can be -1 which should be a separate state
        state.append(min(int(cur_state[9]),int(cur_state[9]/10)))
        state.append(min(int(cur_state[10]),int(cur_state[10]/10)))
        state.append(int(cur_state[11]))
        state.append(int(cur_state[12]/10))

        for i in range(6):
            state.append(int((cur_state[i]-self.state_copy[i])>0))

        z_t = cur_state[6]
        z_l = (cur_state[8]/2)+z_t-(self.car_width/2)
        z_r = (cur_state[8]/2)-z_t-(self.car_width/2)
        state.append(int((z_l/z_r)))

        return tuple(state)    


    def initialise(self,init_act_list):
        self.reward_list.append(self.total_reward)

        self.reward = 0
        self.total_reward = 0

        self.prev_acts = list(init_act_list)
        self.cur_acts = None
        self.cur_state = None
        self.prev_state = None

        self.car_width = 1.75

        self.random_guess = 0

    def getMaxAction(self,cur_acts,sample=True):
        action = None
        top_q = None
        act = None

        if tuple(cur_acts) in self.q_values.keys():
        #if tuple(self.cur_state) in self.q_values.keys():
            act_list = self.q_values[tuple(cur_acts)]
            #act_list = self.q_values[tuple(self.cur_state)]
            top_q = max(np.random.choice(act_list,size=len(act_list),replace=False))
            act = act_list.index(top_q)
        else:
            self.q_values[tuple(cur_acts)] = [0,0,0]
            #self.q_values[tuple(self.cur_state)] = [0,0,0]

        if act is None or (sample and random.random()<self.e):
            self.random_guess +=1
            act = random.randint(0,2)
        
        return act

    def move(self,action,model):
        prob_dict = model.getProb(''.join(self.prev_acts))
        if ''.join(action) in prob_dict:
            action_prob = prob_dict[''.join(action)]
            max_prob = max([prob_dict[x] for x in prob_dict if x != "count"])
            return 1-(max_prob-action_prob)
        else:
            return -1

    def act(self,model,learning=True):
        self.act_index = self.getMaxAction(self.prev_acts,sample=True)

        if self.act_index == 0:
            action = "L"
        elif self.act_index == 1:
            action = "F"
        else:
            action = "R"

        self.cur_acts = self.prev_acts[1:] + [action]
        if learning == True:
            self.reward = self.move(tuple(self.cur_acts),model) 
            self.total_reward += self.reward
        return action

    def sense(self,state):
        #Only store the bits from the state that can't be computed
        if self.cur_state == None:
            self.prev_state = []
        else:
            self.prev_state = self.cur_state
        self.cur_state = self.getState(state)
        self.state_copy = state


    def learn(self,true_action):
        if index_to_direc[self.act_index] == true_action:
            self.reward += 1
            if true_action in ["L","R"]: self.reward += .1
        
        q_list = self.q_values[tuple(self.prev_acts)]
        #q_list = self.q_values[tuple(self.prev_state)]
        max_act = self.getMaxAction(self.cur_acts,sample=False)
        
        max_q = self.q_values[tuple(self.cur_acts)][max_act]
        #max_q = self.q_values[tuple(self.cur_state)][max_act]

        q_list[self.act_index] += self.alph*(self.reward+self.gam*max_q - q_list[self.act_index])

    def callback(self,learn,episode,iteration,true_to_state):
        #print("{}/{}: {}".format(episode,iteration+1,self.total_reward))
        self.prev_acts = list(true_to_state)
        if learn:
            self.e = .001+1.0/(episode)
            self.alph = .001 + 1.0/episode

        else:
            self.e = 0

    def getRules(self):
        max_act = None
        rule_dict = {}
        for entry in self.q_values:
            max_act = np.argmax(self.q_values[entry])
            rule_dict[''.join(entry)] = index_to_direc[max_act]
        return rule_dict
