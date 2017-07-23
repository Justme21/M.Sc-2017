#!/usr/bin/python

import math
import random

class DrivingLearner():
    def __init__(self,look_back):
        self.state = []
        self.from_state = []
        self.prev_states = []

        self.e = .05
        self.gam = .99
        self.alph = 1

        self.reward = 0
        self.total_reward = 0
        self.reward_list = []

        self.look_back = look_back

        self.q_values = {}


    def getState(self,cur_state):
        state = []

        if self.from_state == []:
            self.from_state = [0 for _ in range(6)]

        #First part of the state will be the last 4 states we were in
        state += list(self.prev_states)
       
        #Binary values for the entries in the state relating to acceleration         
        for i in range(6):
            state.append(cur_state[i]>0)

        #The values for car position and road width are all
        # in the 1.5-3 region
        for i in [6,8]:
            state.append(int(cur_state[i]/.5))

        #Distance to car in front can be -1 which should be a separate state
        state.append(min(cur_state[9],int(cur_state[9]/10)))
        state.append(min(cur_state[10],int(cur_state[10]/10)))
        state.append(int(cur_state[11]))
        state.append(int(cur_state[12]/10))

        for i in range(6):
            state.append((cur_state[i]-self.from_state[i])>0)

        z_t = state[6]
        z_l = (state[8]/2)+z_t-(self.car_width/2)
        z_r = (state[8]/2)-z_t-(self.car_width/2)
        state.append(int((z_l/z_r)/.2))

        return state    


    def initialise(self,init_state_list):
        self.reward_list.append(self.total_reward)

        self.reward = 0
        self.total_reward = 0

        self.prev_states = list(init_state_list)
        self.state = None

        self.car_width = 1.75


    def getMaxAction(self):
        action = None
        top_q = None
        act = None

        if ''.join(self.prev_states) in self.q_values.keys():
            act_list = self.q_values[''.join(self.prev_states)]
            top_q = max(act_list)
            act = act_list.index(top_q)
        else:
            self.q_values[''.join(self.prev_states)] = [0,0,0]

        if act is None or random.random()<self.e:
            act = random.randint(0,2)
        
        return act

    def move(self,action,model):
        prob_dict = model.getProb(''.join(self.prev_states))
        if action in prob_dict:
            action_prob = prob_dict[action]
            max_prob = max([prob_dict[x] for x in prob_dict if x != "count"])
            print("{}\t{}\t{}".format(max_prob,action_prob,max_prob-action_prob))
            exit(-1)
            return 1-(max_prob-action_prob)
        else:
            return -1

    def act(self,model):
        action = None
        act = self.getMaxAction()

        if act == 0:
            action = "L"
        elif action == 1:
            action = "F"
        else:
            action = "R"

        self.reward = self.move(''.join(self.prev_states[1:]+[action]),model)
        if self.reward>0: #Only make the move if it's a possible move to make
            self.prev_states = self.prev_states[1:]+[action]
        if math.fabs(self.reward)>1:
            print("THE BIG 'UN: {}".format(self.reward))
            exit(-1)
        self.total_reward += self.reward
        return ''.join(self.prev_states)


    def sense(self,state):
        #Only store the bits from the state that can't be computed
        if self.state == None:
            self.from_state = []
        else:
            self.from_state = self.state[self.look_back:13+self.look_back]
        self.state = self.getState(state)


    def learn(self,model):
        q_list = self.q_values[self.from_state]
        max_act = self.getMaxAction()
        max_q = self.q_values[self.state][max_act]

        q_list[self.act_index] += self.alph*(self.reward+self.gam*max_q - q_list[self.act_index])

    def callback(self,learn,episode,iteration):
        print("{}/{}: {}".format(episode,iteration,self.total_reward))
        if learn:
            self.e = 1.0/(1+3*episode)
            self.alph = 1.0/episode
        else:
            self.e = 0
