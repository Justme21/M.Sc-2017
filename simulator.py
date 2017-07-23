#!/usr/bin/python

from markov import MarkovModel
from driving_q_learner import DrivingLearner
from datastore import DataStore

import matplotlib as plt
import os
import re

def advanceAll(source_list):
    """Advances all entries in source_list at least once and then ensures
       that all of them have jumped. Returns the new indices for all the sources.
       If None is in the index list then one of the sources has reached the end of their datasets"""
    index_list = []
    # Advance all sources once
    for source in source_list:
        source.advance()

    #This is done to ensure that one data doesn't make a large timestep
    # While the others make a smaller timestep, leading to desynchronised data
    cur_time = max(source.time for source in source_list)
    for source in source_list:
        while source.time<cur_time and source.index!=None:
            source.advance()
        index_list.append(source.index)
    return index_list


def advanceIncr(source_list):
    """Only advances one source at a time. The source that is advanced is the one with the lowest
       timestamp value. This maximises the number of distinct readings that can be taken from the data"""
    index_list = []
    cur_time = None
    cur_source = None
    for source in source_list:
        if cur_source is None or cur_time>source.time:
            cur_time = source.time
            cur_source = source
    cur_source.advance()
    index_list = [source.index for source in source_list]
    return index_list

def granularise(dataset,granularity=.5):
    """Given a dataset with time as its first element binds dataset time component to 
       nearest time step and reduces dataset to only stores one entry per time step
       at granularity specified. e.g. if granularity=.5 all entries time component is
       brought to the nearest .5 second and if multiple entries have same time value
       only first entry is kept"""
    last,ind = 0,1
    #Bind time to nearest granular time step
    for i in range(len(dataset)):
        if dataset[i][0]%granularity<granularity/2:
            dataset[i][0]-=dataset[i][0]%granularity
        else:
            dataset[i][0]+= granularity - dataset[i][0]%granularity

    #Delete multiple entries occurring in the same time step
    while ind+1<len(dataset):
        while ind<len(dataset) and dataset[ind][0]-dataset[last][0]<granularity:
            del dataset[ind]

        last = ind
        ind += 1

    return dataset


def getAnnotation(location):
    anno_file = open("{}-annotation.txt".format(location),"r")
    annotation = []
    for line in anno_file:
        (time,direc) = line.split()
        annotation.append([float(time),direc])
    return annotation

def getDirectories():
    datasources = []
    root_dir = "./Datasets/UAH-DRIVESET-v1/"
    directory_tag = re.compile('D.')
    folder_tag = re.compile('.*-MOTORWAY')
    dir_list = [x for x in os.listdir(root_dir) if re.match(directory_tag,x)]
    for direc in dir_list:
        sub_dir_list = [x for x in os.listdir(root_dir+"/"+direc) if re.match(folder_tag,x)]
        for sub_dir in sub_dir_list:
            if '-annotation.txt' in os.listdir(root_dir+"/"+direc+"/"+sub_dir):
                datasources.append(root_dir+"/"+direc+"/"+sub_dir+"/")
    return datasources


def getTimeSlots(time_list):
    begin = time_list[0]
    time_slots = []
    for i,entry in enumerate(time_list):
        if i>0 and entry-time_list[i-1]>.5:
            time_slots.append((begin,time_list[i-1]))
            begin = entry
        if i==len(time_list)-1:
            time_slots.append((begin,entry))

    return time_slots


def getData(dataset,start_time,end_time):
    i = 0
    while i<len(dataset) and dataset[i][0]<start_time: i+=1
    j=i
    while j<len(dataset) and dataset[j][0]<end_time: j+=1
    if j-i<120: return None #Want all data intervals to be at least a minute long
    else:
        return [dataset[it] for it in range(i,min(j+1,len(dataset)))]


def makeDataList(location):
    data_list = []
   
    files_of_interest = ["RAW_ACCELEROMETERS","PROC_LANE_DETECTION","PROC_VEHICLE_DETECTION"]
    entries_of_interest = [[0,2,3,4,5,6,7],[0,1,2,3,4],[0,1,2,3,4]]


    datastore_list = []
    for i,entry in enumerate(files_of_interest):
        datastore_list.append(DataStore("{0}{1}.txt".format(location,entry),entries_of_interest[i]))

    dataset = []
    row_list = []

    advanceAll(datastore_list) #Bring all sources up to a common starting point

    for entry in datastore_list: row_list += entry.getRow()[1:]
    dataset.append([min(x.time for x in datastore_list)]+row_list)

    while None not in advanceIncr(datastore_list):
        row_list = []
        for entry in datastore_list: row_list+= entry.getRow()[1:]
        dataset.append([min([x.time for x in datastore_list])]+row_list)

    granularity = .5
    dataset = granularise(dataset,granularity)

    annotation = getAnnotation(location)

    time_list = [x[0] for x in annotation]
    time_slots = getTimeSlots(time_list)
   
    data = None
    for (begin,end) in time_slots:
        data = getData(dataset,begin,end)
        if data is not None:
            data_list.append([data,getData(annotation,begin,end)])

    for i,[dataset,annotation] in enumerate(data_list):
        data_list[i] = ([x[1:] for x in dataset],[y[1] for y in annotation]) #remove time from entries

    return data_list

def runSimulation(markov_model,learner,num_ep,num_it):
    annotes,_ = markov_model.simulate(None)
    learner.initialise(annotes)
    for i in range(num_it):
        _,state = markov_model.simulate(annotes)
        learner.sense(state) #First detect the new state
        annotes = learner.act(markov_model)
        learner.callback(True,num_ep,i)
    

look_back = 5
num_episodes = 2000
num_iterations = 3600 #3600 half seconds = 30 minutes

sources = getDirectories()

data_list = []
for entry in sources:
    data_list += makeDataList(entry)

markov_model = MarkovModel(look_back)
learner = DrivingLearner(look_back)

for entry in data_list:
    markov_model.addData(entry)

markov_model.finishAdd()

for i in range(1,num_episodes+1):
    runSimulation(markov_model,learner,i,num_iterations)

learner.reward_list.append(learner.total_Reward)
plt.plot(learner.reward_list)