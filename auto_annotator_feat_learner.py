#!/usr/bin/python

from datastore import Datastore
import math
import matplotlib.pyplot as plt
import random

class ChangeLearner():

    def __init__(self,sample_state):
        self.prev_state = [0 for _ in range(len(sample_state))]
        self.cur_state = sample_state

        self.alpha = .01
        self.gamma = 1.0

        self.epsilon = .05
        self.num_rounds = {"F":0,"L":0,"R":0}

        #The set of actions available
        self.action_list = ["F","L","R"]

        self.mag_list = []

        #For each action we want to store the average state change induced by that action
        #And the set of weights corresponding to that action
        self.action_state = {}
        self.action_weight = {}
        for act in self.action_list: 
            self.action_state[act] = [0 for _ in range(len(self.cur_state))]
            self.action_weight[act] = [1.0/len(self.cur_state) for _ in range(len(self.cur_state))]

        #Here we are going to use our knowledge of the current structure of the state to try and influence 
        #We know right should tend to be positve prp-acceleration and left negative. By intitialising these
        #Values towards that direction we make it more likely they will be selected
        random.seed(270394)
        self.action_state["R"][3] = math.fabs(random.random())
        self.action_state["L"][3] = -math.fabs(random.random())

        #Variables for the time dilation
        #self.capacity = 50
        #self.num_reps = 500
        #self.entry_window = []


    def updateState(self,new_state):
        self.prev_state = list(self.cur_state)
        self.cur_state = new_state


    def learn(self,action,difference):
        weight_sum = 0
        self.num_rounds[action]+=1
        num_rounds = self.num_rounds[action]
        for i,entry in enumerate(difference):
            self.action_weight[action][i] += (self.alpha+(1.0/num_rounds))*entry
            weight_sum += math.fabs(self.action_weight[action][i])

        #Normalise the weights
        self.action_weight[action] = [x/weight_sum for x in self.action_weight[action]]

        #Use moving average to keep track of the average change induced by an action
        self.action_state[action]  = [((num_rounds-1.0)/num_rounds)*x + (1.0/num_rounds)*difference[i] for i,x in enumerate(self.action_state[action])]


    def getAction(self):
        approx_state = []
        total_num_rounds = sum([self.num_rounds[x] for x in self.action_list])+1
        diff_mags = [0 for _ in range(len(self.action_list))]
       
        true_change = [self.cur_state[i] - self.prev_state[i] for i in range(len(self.prev_state))]
 
        #Given this setup we are trying to determine the most likely action to have just been taken
        #state_diff = [self.cur_state[i]-self.prev_state[i] for i in range(len(self.cur_state))]
        for i,action in enumerate(self.action_list):
            approx_state = [self.prev_state[j]+self.action_state[action][j] for j in range(len(self.prev_state))]
            for j,entry in enumerate(approx_state): 
                diff_mags[i]+= math.fabs(self.action_weight[action][j]*(self.cur_state[j] - approx_state[j]))
       
        #Stored separately for plotting purposes
        self.mag_list = list(diff_mags)


        # Epsilon - greedy action sampling
        alt = random.random()
        #Annealing epsilon so that we make fewer random guesses over time
        if alt<(self.epsilon+(1.0/(total_num_rounds))):
            #Here the action is sampled at random based on 1-the proportion of the 
            # sum of the magnitudes of the difference determined by that action
            #In this the action with the lowest magnitude difference is still
            # most probable to be selected
            sum_mags = sum(diff_mags)
            for i in range(len(diff_mags)):
                diff_mags[i]/=sum_mags
                if i!=0: diff_mags[i]+= diff_mags[i-1]

            diff_mags = [1-x for x in diff_mags]            

            min_ind = len(diff_mags)-1
            choice = random.random()
            while choice<diff_mags[min_ind]:
                min_ind-=1
            min_mag = diff_mags[min_ind] 
        
        else:
            #Greedy action selection. Choose action with difference of lowest magnitude
            min_mag = min(diff_mags)
            min_ind = diff_mags.index(min_mag)
        
        min_action = self.action_list[min_ind]

        self.num_rounds[min_action] += 1

        self.learn(min_action,true_change)
        
        #self.entry_window.append((min_action,state_diff))
        #if len(self.entry_window) == self.capacity:
        #    action,diff = None, None
        #    for _ in range(self.num_reps):
        #        index_list = [random.randint(0,self.capacity-1) for _ in range(self.capacity)]
        #        for index in index_list:
        #            (action,diff) = self.entry_window[index]
        #            self.learn(action,diff)
        #    self.entry_window = [] 
        
        return min_action,min_mag


    def whatDoYouKnow(self):
        return self.action_state, self.action_weight


def advanceAll(source_list):
    index_list = []
    for source in source_list:
        source.advance()
    cur_time = max(source.time for source in source_list)
    for source in source_list:
        while source.time<cur_time and source.index!=None:
            source.advance()
    index_list.append(source.index)
    return index_list

#Locations and storage variables
folder_loc = "Datasets/UAH-DRIVESET-v1/D2/20151120133502-26km-D2-AGGRESSIVE-MOTORWAY/"
#folder_loc = "Datasets/UAH_DRIVESET-v1/D3/20151126110502-26km-D3-NORMAL-MOTORWAY"
files_of_interest = ["RAW_ACCELEROMETERS","PROC_LANE_DETECTION","PROC_VEHICLE_DETECTION"]
entries_of_interest = [[0,3,4,6,7,8,9,10],[0,1,2,3],[0,1,2]]


#Creating a list with access to all the data_folders we are interested in
datastore_dict = {}
for i, entry in enumerate(files_of_interest):
    datastore_dict[entry] = DataStore("{0}{1}.txt".format(folder_loc,entry),entries_of_interest[i])

start_time = max([datastore_dict[x].time for x in files_of_interest])
for entry in files_of_interest:
    while datastore_dict[entry].time<start_time: datastore_dict[entry].advance()
datasource_list = [datastore_dict[entry] for entry in files_of_interest]

#We want the entries in the datastore to all be in common units metres
# Accelerations are all currently in Gs
for i in range(len(datasource_list[0].file_content)):
    for index in [2,3,4,5,6,7]:
        if index in entries_of_interest[0]:
            datasource_list[0].file_content[i][entries_of_interest[0].index(index)]*= 9.81

for source in datasource_list:
    source.rescale()

road_posit = datastore_dict["PROC_LANE_DETECTION"].getAll(1)
plt.plot(road_posit)
plt.show()
exit(-1)

time = []
action = None
new_state = None

plot_list = []
conv_list = []

state_len = 0
for source in datasource_list:
    state_len += len(source.getRow()) - 1 #Don't want to include the timestamp as a state


learner = ChangeLearner([0 for _ in range(state_len)])
while None not in advanceAll(datasource_list):
    new_state = []
    for source in datasource_list:
        new_state += source.getRow()[1:] #Omitting the timestamp
    time = max([source.time for source in datasource_list])
   
    learner.updateState(new_state)
    action,mag = learner.getAction()

    #Keep track of the magnitude of the difference for each action at each timestep
    plot_list.append(list(learner.mag_list))
    print("{}:{:02d} ({}) \t {}\t{}".format(int(time/60),int(time%60),time,action,mag))

print("\n What the Learner Learnt:")
learnt_dict,weight_dict = learner.whatDoYouKnow()
learnt_list,weight_list = None,None
for direc in learnt_dict:
    learnt_list = learnt_dict[direc]
    weight_list = weight_dict[direc]
    learnt_list = [round(x,3) for x in learnt_list]
    weight_list = [round(x,3) for x in weight_list]
    print("{}: {}".format(direc,learnt_list))
    print("{}: {}".format(direc,weight_list))
    print("")

print(learner.num_rounds)

plot_1 = [entry[0] for entry in plot_list]
plot_2 = [entry[1] for entry in plot_list]
plot_3 = [entry[2] for entry in plot_list]
plt.plot(plot_1,'r-')
plt.plot(plot_2,'g-')
plt.plot(plot_3,'b-')
plt.show()
