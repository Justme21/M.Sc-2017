#!/usr/bin/python

import math

class DataStore():

    def __init__(self,file_name,content_list):
        self.name = file_name #This is the full address of datasource
        self.file_content = [] #Will hold the desired contents from the file
        op_file = open("{}".format(file_name))
        if content_list is []:
            content_list = [i for i in range(len(op_file[0].split()))]
        for entry in op_file: #Iterate through the rows in the file
            #We only include the entries corresponding to the 0-indexed columns specified in content_list
            entry = [float(x) for i,x in enumerate(entry.split()) if i in content_list]
            self.file_content.append(entry)

        self.time = self.file_content[0][0] #Initially set to first entry in the timestamp
        self.index = 0


    def getAll(self,i):
        tot_list = []
        for row in self.file_content:
            if -9 not in row: #This is a sloppy way to omit the problem entries in PROC_LANE_DETECTION
                tot_list.append(row[i])

        return tot_list

    def getRow(self):
        return self.file_content[self.index]

    def advance(self):
        self.index+=1
        if self.index==len(self.file_content): self.index = None
        else:
            self.time = self.file_content[self.index][0]


class ChangeLearner():

    def __init__(self,sample_state):
        self.prev_state = None
        self.cur_state = sample_state

        self.gradient = .01
        self.gamma = .95

        self.action_list = ["F","L","R"]
        self.action_dict = {}
        for act in self.action_list: self.action_dict[act] = [0 for _ in range(len(self.cur_state))]

        #Here we are going to use our knowledge of the current structure of the state to try and influence 


    def updateState(self,new_state):
        self.prev_state = list(self.cur_state)
        self.cur_state = new_state


    def learn(self,action,difference):
        for i,entry in enumerate(difference):
            self.action_dict[action][i] += self.gradient*(self.gamma*entry-self.action_dict[action][i])


    def getAction(self):
        diff_mags = [0 for _ in range(len(self.action_list))]
        
        state_diff = [self.cur_state[i]-self.prev_state[i] for i in range(len(self.cur_state))]
        for i,action in enumerate(self.action_list):
            for j,entry in enumerate(state_diff): 
                diff_mags[i]+= math.fabs(entry - self.action_dict[action][j])
        
        min_mag = min(diff_mags)
        min_ind = diff_mags.index(min_mag)
        min_action = self.action_list[min_ind]

        self.learn(min_action,state_diff)

        return min_action,min_mag

    def whatDoYouKnow(self):
        return self.action_dict


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

time = []
action = None
new_state = None

state_len = 0
for source in datasource_list:
    state_len += len(source.getRow())


learner = ChangeLearner([0 for _ in range(state_len)])
while None not in advanceAll(datasource_list):
    new_state = []
    for source in datasource_list:
        new_state += source.getRow()
    time = max([source.time for source in datasource_list])
   
    learner.updateState(new_state)
    action,mag = learner.getAction()

    print("{}:{:02d} ({}) \t {}\t{}".format(int(time/60),int(time%60),time,action,mag))

print("\n What the Learner Learnt:")
learnt_dict = learner.whatDoYouKnow()
learnt_list = None
for direc in learnt_dict:
    learnt_list = learnt_dict[direc]
    learnt_list = [round(x,3) for x in learnt_list]
    print("{}: {}".format(direc,learnt_list))
