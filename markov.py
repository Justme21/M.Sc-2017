#!/usr/bin/python
import math
import numpy as np

class MarkovModel():
   
    def __init__(self,state_len=10):
        self.state_dict_list = {}
        self.state_action_dict = {}
        self.state_len = state_len


    def addData(self,data_annotation):
        (data,annotation) = data_annotation
        cur_state = ['0'] #This is the trigger for sequence starts. Should be helpful for simulation
        prev_state = None
        state_dict = {}
        for row,anno in zip(data,annotation):
            cur_state += [anno]
            if len(cur_state)>self.state_len:
                prev_state = cur_state[:-1]
                prev_key = ''.join(str(x) for x in prev_state)

                cur_state = cur_state[1:]
                cur_key = ''.join(str(x) for x in cur_state)

            if prev_state is not None:
                if prev_key not in self.state_dict_list:
                    self.state_dict_list[prev_key] = {"count":0}
                    self.state_action_dict[prev_key] = {}
                state_dict = self.state_dict_list[prev_key]
                action_dict = self.state_action_dict[prev_key]
                state_dict["count"] += 1
                if cur_key not in state_dict:
                    state_dict[cur_key] = 0
                    action_dict[cur_key] = []
                state_dict[cur_key] += 1
                action_dict[cur_key].append(row)


    def changeToProb(self):
        content_dict = {}
        count = None
        for entry in self.state_dict_list:
            content_dict = self.state_dict_list[entry]
            count = content_dict["count"]
            for entry2 in content_dict:
                content_dict[entry2]/=count


    def printContents(self):
        for entry in self.state_dict_list:
            if entry != "count":
                print("{}: {}".format(entry,self.state_dict_list[entry]))


    def buildModel(self,data): 
        for entry in data:
            self.addData(entry) 


    def averageStates(self):
        average,std = None,None
        state_samples = []
        for from_state in self.state_action_dict:
            for to_state in self.state_action_dict[from_state]:
                state_samples = self.state_action_dict[from_state][to_state]
                average = []
                std = []
                for i in range(len(state_samples[0])):
                    average.append(np.mean([x[i] for x in state_samples]))
                    if len(state_samples)>1:
                        std.append(np.std([x[i] for x in state_samples]))
                    else:
                        #Default, non-zero standard deviation used to ensure operation
                        std.append(math.fabs(average[i])/10)
                self.state_action_dict[from_state][to_state] = (list(average),list(std))


    def finishAdd(self):
        self.changeToProb()
        self.averageStates() 
    

    def sampleAction(self,avg_std_list):
        (average_list,std_list) = avg_std_list
        action = []
        for avg,std in zip(average_list,std_list):
            action.append(np.random.normal(avg,std))
        return action

    def cumulProb(self,action_dict):
        prev = 0
        act_dict = dict(action_dict)
        for action in act_dict:
            if action != "count":
                act_dict[action] += prev
                prev = act_dict[action]

        return act_dict


    def getProb(self,key):
        return self.state_dict_list[key]


    def simulate(self,from_state=None):
        if from_state == None:
            action_set = sorted([x for x in self.state_dict_list.keys() if "0" in x])
            index = np.random.randint(0,len(action_set)-1)
            return action_set[index],None
        else:
            num = np.random.random()
            action_dict = self.cumulProb(self.state_dict_list[from_state])
            for to_state in [key for key in action_dict if key != "count"]:
                if num<action_dict[to_state]: 
                    params = self.state_action_dict[from_state][to_state]
                    return to_state,self.sampleAction(params)


    def generateData(self,num_iterations,rand_seed):
        np.random.seed(rand_seed)
        datalist = []
        init_state,_ = self.simulate(None)
        from_state = init_state
        for _ in range(num_iterations):
            from_state,action_data = self.simulate(from_state)
            datalist.append((from_state[-1],action_data))
        return init_state,datalist
