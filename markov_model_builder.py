#!/usr/bin/python

from datastore import DataStore

import math
import markov
import numpy as np
import os
import random
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


def restructure(row,num_contributers):
    """Reorder the entries in the windowed dataset so that common entries are grouped together
       and are ordered from largest to smallest"""
    #len_single_entry should be integer valued anyway, so this won't induce any rounding
    len_single_entry = int(len(row)*1.0/num_contributers)

    posit = None
    sub_row = None
    row_redux = [] #Keep the timestamp at the start
    for i in range(len_single_entry):
        sub_row = [row[i]]
        posit = 1
        while i+posit*len_single_entry < len(row):
            sub_row.append(row[i+posit*len_single_entry])
            posit += 1
        row_redux += sorted(sub_row)

    return row_redux


def getAnnotation(location):
    anno_file = open("{}-annotation.txt".format(location),"r")
    annotation = []
    for line in anno_file:
        (time,direc) = line.split()
        annotation.append([float(time),direc])
    return annotation


def makeWindows(dataset,granularity,time_width=5,overlap=1):
    """Chunks a given dataset up into "windows" of time of length "time_width"
       "granularity" is how granular the data is, e.g. if the data is stored in .5 second incremements then granularity
       is .5.
       Overlap determines how much the windows overlap; if it's 2 then the entries that make up the second half of window j
       make up the first half of window j+1 """
    posit=0
    window_len =  math.ceil(time_width/granularity) #Assume time_width>granularity
    windowed_dataset = []
    while posit<len(dataset):
        row = []
        j = 0
        while posit+j<len(dataset) and j<window_len: #dataset[posit+j][0]-dataset[posit][0]<window_len:
            row += dataset[posit+j][1:]
            j += 1
 
        if posit+j<len(dataset) and  j==window_len: #dataset[posit+j][0]-dataset[posit][0] >= window_len:    
            row = restructure(row,window_len)
            windowed_dataset.append([dataset[posit][0]]+row) #want to include the timestep for reference
            posit+= int(window_len/overlap)#posit+=int(window_len/2) #Overlapping windows
        else:
            posit += j

    return windowed_dataset


def windowAnnotation(anno,time_list):
    ann_pos,max_count,time_pos = 0,0,0
    ann_window,ann_list = [],[]
    max_ann = ""
    time_len = time_list[1]-time_list[0]
    while time_pos<len(time_list):
        ann_window = []
        while anno[ann_pos][0]>=time_list[time_pos] and anno[ann_pos][0]<time_list[time_pos]+time_len:
            ann_window.append(anno[ann_pos][1])
            ann_pos += 1
        
        max_count = 0 
        for entry in set(ann_window):
            if ann_window.count(entry)>max_count:
                max_count = ann_window.count(entry)
                max_ann = entry
        ann_list.append([time_list[time_pos],max_ann])
        time_pos += 1

    return ann_list 


def augmentDataset(dataset):
    """Augments the given dataset to include values that either include values from previous observations
       or else provide non-linear combinations of variables."""
    extra_vals = None
    prev_accl = [0,0,0]
    prev_accl_kal = [0,0,0]
    prev_veh = [0,0]

    car_width = 1.75 #Hardcoded car width
    max_dist = max([entry[11] for entry in dataset]) #The max distance a car is ever away from the ego 
    max_time = max([entry[12] for entry in dataset]) #The max time to collision


    car_rat = None
    for i,entry in enumerate(dataset):
        extra_vals = []
        #Include the change in acceleration in 3 directions
        extra_vals.append(entry[1]-prev_accl[0])
        extra_vals.append(entry[2]-prev_accl[1])
        extra_vals.append(entry[3]-prev_accl[2])

        #Include the change in the accelerations recorded by the kalman filter
        extra_vals.append(entry[4]-prev_accl_kal[0])
        extra_vals.append(entry[5]-prev_accl_kal[1])
        extra_vals.append(entry[6]-prev_accl_kal[2])

        #Include the change in distance to the car in front and time to collision with the car in front
        if prev_veh[0] == -1 or entry[10] == -1:
            extra_vals+= [max_dist,max_time] #Here distance and speed were both -1
        else:
            extra_vals.append(entry[10]-prev_veh[0])
            extra_vals.append(entry[11]-prev_veh[1])

        #Include ration of distance from the left side of the road to that from the right side of the road
        road_width = entry[9]
        z_t = entry[7]
        z_l = (road_width/2)+z_t-(car_width/2)
        z_r = (road_width/2)-z_t-(car_width/2) 
        if z_r == 0: 
            z_r = .005
        extra_vals.append(z_l/z_r)

        #Augment values
        dataset[i]+=extra_vals

        #Reset values for next iteration
        prev_accl = [entry[1],entry[2],entry[3]]
        prev_accl_kal = [entry[4],entry[5],entry[6]]
        prev_veh = [entry[10],entry[11]]

    return dataset


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
        data_list[i] = ([x[1:] for x in augmentDataset(dataset)],[y[1] for y in annotation]) #remove time from entries

    return data_list



sources = getDirectories()

data_list = []
for entry in sources:
    data_list += makeDataList(entry)

markov_model = markov.MarkovModel(5)

for entry in data_list:
    markov_model.addData(entry)

markov_model.finishAdd()

i = 0
cur_state = None
while i<1000:
    #Cur_state is the true annotation for the action given by actions
    #So when in prev_state, and this set of actions are observed, the "best" action
    # should the one leading to cur_state
    cur_state,actions = markov_model.simulate(cur_state)
    if i==0:
        for entry in cur_state[1:]:
            print("{} : {}".format(i,entry))
            i += 1
    else:
        print("{} : {}\t{}".format(i,cur_state[-1],actions))
        i += 1
