#!/usr/bin/python
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
        prev = None
        for entry in self.state_dict_list:
            content_dict = self.state_dict_list[entry]
            count = content_dict["count"]
            prev = None
            for entry2 in content_dict:
                if entry2 != "count":
                    content_dict[entry2]/=count
                    if prev is not None: content_dict[entry2] += prev
                    prev = content_dict[entry2]


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
                    std.append(np.std([x[i] for x in state_samples]))
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

    def simulate(self,start=None):
        if start == None:
            action_set = [x for x in self.state_dict_list.keys() if "0" in x]
            return np.random.choice(action_set),None
        else:
            num = np.random.random()
            action_dict = dict(self.state_dict_list[start])
            for state in [key for key in action_dict if key != "count"]:
                if num<action_dict[state]: 
                    params = self.state_action_dict[start][state]
                    return state, self.sampleAction(params)
