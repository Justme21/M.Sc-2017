#!/usr/bin/python

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

        self.gradient = .1

        self.action_list = ["L","F","R"]
        self.action_dict = {}
        for act in self.action_list: self.action_dict[act] = [0 for _ in range(len(prev_state))]


    def updateState(self,new_state):
        self.prev_state = list(self.cur_state)
        self.cur_state = new_state


    def getAction(self):
        diff_mags = [0 for _ in range(len(self.action_list))]
        
        state_diff = [cur_state[i]-prev_state[i] for i in range(len(cur_state))]
        for i,action in enumerate(self.action_list):
            


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

time = 0
action = None

learner = ChangeLearner([source.getRow() for source in datasource_list])
while None not in advanceAll(datasource_list):
    new_state = [source.getRow() for source in datasource_list]
    time = datasource_list[0].time
   
    learner.updateState(new_state)
    action = learner.getAction()

    print("{}:{} \t {}".format(int(time/60),int(time%60),action))

