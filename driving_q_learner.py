#!/usr/bin/python

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

        self.look_back = 5

        self.q_values = {}


    def getState(self,cur_state)
        state = []

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
            state.append((cur_state[i]-self.prev_state[i])>0)

        z_t = state[6]
        z_l = (state[8]/2)+z_t-(self.car_width/2)
        z_r = (state[8]/2)-z_t-(self.car_width/2)
        state.append(int((z_l/z_r)/.2))

        return state    


    def initialise(self,model):
        self.reward_list.append(self.total_reward)

        self.reward = 0
        self.total_reward = 0

        self.prev_states = list(model.getInitStates())
        self.state = self.getState(model.getState())

        self.car_width = model.getCarWidth()


    def getMaxAction(self):
        action = None
        top_q = None
        act = None

        if self.state in self.q_values.keys():
            act_list = self.q_values[self.state]
            top_q = max(act_list)
            act = act_list.index(top_q)
        else:
            self.q_values[self.state] = [0,0,0]

        if act is None or random.random()<self.e:
            act = random.randint(0,2)
        
        return act

    def act(self):
        action = None
        act = self.getMaxAction()

        if act == 0:
            action = "L"
        elif action == 1:
            action = "F"
        else:
            action = "R"

        self.reward = self.move(action)
        self.prev_states = self.prev_states[1:]+[action]
        self.total_reward += self.reward


    def sense(self,model):
        #Only store the bits from the state that can't be computed
        self.from_state = self.state[5:18]
        self.state = self.getState(model.getState())


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
