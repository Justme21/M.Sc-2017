#!/usr/bin/python

import math
import matplotlib.pyplot as plt
import numpy as np
from pyearth import Earth
from sklearn import decomposition,manifold,mixture,preprocessing
from sklearn.cluster import KMeans #used to perform k-means
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster



class DataStore():

    def __init__(self,file_name,content_list):
        """Initialise the Datastore object. Store the address of the data as well as the 
           desired content which is read from the file at the specified address.
           Data is then cleaned to remove "bad" entries and impute values (this is only relevant for
           PROC_LANE_DETECTION file"""
        self.name = file_name #This is the full address of datasource
        self.file_content = [] #Will hold the desired contents from the file

        #Read relevant entries in from the file
        op_file = open("{}".format(file_name))
        for entry in op_file: #Iterate through the rows in the file
            #We only include the entries corresponding to the 0-indexed columns specified in content_list
            entry = [float(x) for i,x in enumerate(entry.split()) if i in content_list]

            #-9 is a default value for the detectors and, given the scale of the data
            if -9.0 not in entry:
                self.file_content.append(entry)

        #Aside from the -9's which are only at the start or end of files, PROC_LANE_DETECTION
        # is the only file that has missing values in it. These are handled in rehab
        if "PROC_LANE_DETECTION" in self.name:
            self.rehab() #Where data goes to get clean
            #The last feature in this dataset is an indicator variable indicating when data is bad
            # this feature does not relate to the driver's behaviour so is omitted
            self.file_content = [entry[:-1] for entry in self.file_content] #Omit unrelated variable
        
        self.index = 0 #Initialise the index to 0
        self.time = self.file_content[0][0] #Initialise time to be the time of the first entry


    def getAll(self,i):
        """Return all the entries of index i in self.file_content
           NOTE: i here is the index IN THE DATASTORE and not in the original file"""
        tot_list = []
        for row in self.file_content:
            if -9 not in row: #This is a sloppy way to omit the problem entries in PROC_LANE_DETECTION
                tot_list.append(row[i])

        return tot_list


    def getRow(self,index=None):
        """Return the row at a specified index in the datastore. If no index is provided then it returns
           row at the current index"""
        if index is None: index = self.index
        return self.file_content[index]


    def getIndex(self,time=None):
        """Returns the index that has timestamp closest to the specified time value (in seconds).
           If time is none then returns the entry relating to the current timestamp"""
        i = 0
        min_dist = None
        if time is None: time=self.time
        while min_dist is None or math.fabs(time-self.file_content[i][0])<=min_dist:
            min_dist = math.fabs(time-self.file_content[i][0])
            i+=1
        return i-1
 

    def advance(self):
        """Increments index and increases the timestamp to correspond. If index exceeds length of
           dataset index is set to None to indicate reaching the end of the data"""
        self.index+=1
        if self.index>=len(self.file_content): self.index = None
        else:
            self.time = self.file_content[self.index][0]


    def reverse(self):
        """Decrements the index and adjusts timestamp to correspond. If index is already 0 then it
           stays at 0"""
        self.index-=1
        if self.index<0: self.index = 0
        self.time = self.file_content[self.index][0]

    
    def rehab(self):
        """Cleans the PROC_LANE_DETECTION data and imputes missing values.
           Imputation is done by identifying, using the indicator feature, values we are confident in
           immediately before and after regions of missing data. We then assume the data increases/decreases
           linearly in the interim, computing the mean and adding this to the values to fill in the unknowns"""
        start,end,direc = -1,-1,None #direc is used for position on road since it is only discontinuous value
        i = 0
        entry, change = None,None
        err_index,break_list = [],[]     

        #-1 means calibrating. Assume this only happens at start of dataset
        while self.file_content[0][-1] == -1:
            del self.file_content[0]
        
        #A list of all the points where the detection software gives bad output
        err_index = [j for j,entry in enumerate(self.file_content) if entry[-1] in [0,-1]]
        
        #Here we define the beginning and end index of regions of unreliable data 
        while i<len(err_index):
            start = err_index[i]
            #There are singular entries where the indicator suggests the data is reliable but inspection
            # reveals this not to be the case. If there are less than 5 entries between two confirmed errors
            # then we assume that any entries between them are errors/unreliable
            while i+1<len(err_index) and err_index[i+1]-err_index[i]<5: i+=1
            end = err_index[i]
            break_list.append((start,end)) #Breaklist stores the indices for beginning and end of unreliable regions
            i+=1

        for (a,b) in break_list:
            #Start and end are the entries immediately before and after each problem region
            start = list(self.file_content[a-1])
            end = list(self.file_content[b+1])
            #For the sake of accuracy in calculating the mean road_width is, when calculating the averages
            # assumed to be the value at the end of the region. This is to prevent discontinuities
            # between imputed values and real values  
            road_width = end[3]
 
            #Only need direc for car position as is discontinuous
            direc = start[1]-self.file_content[a-2][1] #Instantaneous slope before data loss            
            if (direc<0 and start[1]<0 and end[1]>0) or (direc>0 and start[1]>0 and end[1]<0):
                end[1] += (direc/math.fabs(direc))*road_width #Add or subtract road width as necessary

            #Calculate the amount of change required per step through the problem region to get from one known value to the other
            change = [(end[x] - start[x])*1.0/((b+1)-(a-1)) for x in range(len(start))]
            for j in range(a,b+1):
                #timestamp is a confirmed value, so we don't need to change that. Remaining values are calculated by adding to the values from the previous
                # row. 3 is appended to the end as an indicator variable indicating the values have been changed.This is not used in the program and is mainly
                # to preserve row lengths.
                self.file_content[j] = [self.file_content[j][0]]+[round(self.file_content[j-1][x] + change[x],3) for x in range(1,len(change)-1)] +[3.0]
                
                #This is handling the special case of discontinuity for road position. If you are more than half the road width from the 
                # centre you are on the adjacent road. From the ego perspective your position shifts from being very far left to very far right
                # on the road (or vice versa)
                if math.fabs(self.file_content[j][1])>road_width/2:
                    self.file_content[j][1] = round(self.file_content[j][1]-(direc/math.fabs(direc))*road_width,3)            
       
     

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


def restructure(row,num_contributers):
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
        


folder_loc = "Datasets/UAH-DRIVESET-v1/D1/20151111123124-25km-D1-NORMAL-MOTORWAY/"

files_of_interest = ["RAW_ACCELEROMETERS","PROC_LANE_DETECTION","PROC_VEHICLE_DETECTION"]
entries_of_interest = [[0,2,3,4,5,6,7],[0,1,2,3,4],[0,1,2,3,4]]
#Omit entry 8 as this is Roll which is only significantly different on  outgoing and return trips

datastore_list = []
for i,entry in enumerate(files_of_interest):
    datastore_list.append(DataStore("{0}{1}.txt".format(folder_loc,entry),entries_of_interest[i]))

dataset = []
row_list = []


#375 marks the start of the turn back 
for entry in datastore_list: entry.file_content = entry.file_content[:entry.getIndex(375)]+entry.file_content[entry.getIndex(490):]


advanceAll(datastore_list) #Bring all sources up to a common starting point


for entry in datastore_list: row_list += entry.getRow()[1:]
dataset.append([min(x.time for x in datastore_list)]+row_list)

while None not in advanceIncr(datastore_list):
    row_list = []
    for entry in datastore_list: row_list+= entry.getRow()[1:]
    dataset.append([min([x.time for x in datastore_list])]+row_list)


last,ind = 0,1
granularity = .5

for i in range(len(dataset)):
    if dataset[i][0]%granularity<granularity/2:
        dataset[i][0]-=dataset[i][0]%granularity
    else:
        dataset[i][0]+= granularity - dataset[i][0]%granularity

while ind+1<len(dataset):
    while ind<len(dataset) and dataset[ind][0]-dataset[last][0]<granularity:
        del dataset[ind]

    last = ind
    ind += 1

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
    extra_vals.append(entry[1]-prev_accl[0])
    extra_vals.append(entry[2]-prev_accl[1])
    extra_vals.append(entry[3]-prev_accl[2])

    extra_vals.append(entry[4]-prev_accl_kal[0])
    extra_vals.append(entry[5]-prev_accl_kal[1])
    extra_vals.append(entry[6]-prev_accl_kal[2])

    if prev_veh[0] == -1 or entry[10] == -1:
        extra_vals+= [max_dist,max_time] #Here distance and speed were both -1
    else:
        extra_vals.append(entry[10]-prev_veh[0])
        extra_vals.append(entry[11]-prev_veh[1])

    road_width = entry[9]
    z_t = entry[7]
    z_l = (road_width/2)+z_t-(car_width/2)
    z_r = (road_width/2)-z_t-(car_width/2) 
    extra_vals.append(z_l/z_r)

    dataset[i]+=extra_vals

    prev_accl = [entry[1],entry[2],entry[3]]
    prev_accl_kal = [entry[4],entry[5],entry[6]]
    prev_veh = [entry[10],entry[11]]

posit=0
time_width = 5 #seconds
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
        windowed_dataset.append([dataset[posit+j-1][0]]+row) #want to include the timestep for reference
        posit+= int(window_len/2)#posit+=int(window_len/2) #Overlapping windows
    else:
        posit += j
       
   
#Remove time so that it doesn't get factored into analysis
dataset_np = np.array([entry[1:] for entry in windowed_dataset])


#Scales data to have 0 mean and unit variance
scaled_dataset = preprocessing.StandardScaler().fit_transform(dataset_np)



turn_times = {36:[1,0,0],56:[1,0,0],66:[0,0,1],83:[0,0,1],98:[1,0,0],104:[0,0,1]}#,164:[1,0,0],305:[0,0,1],501:[0,0,1],513:[1,0,0]}
turn_times.update({44:[0,1,0],75:[0,1,0],110:[0,1,0],115:[0,1,0],120:[0,1,0]})#,125:[0,1,0],494:[0,1,0],525:[0,1,0]})
turn_set = []
turns = sorted([x for x in turn_times])

for turn in turns:
    for entry in windowed_dataset:
        if turn>=entry[0] and turn<=entry[0]+time_width:
            turn_set.append(entry[1:])
            break

scaled_turn_set = preprocessing.StandardScaler().fit_transform(turn_set)

KPCA = decomposition.KernelPCA(kernel="sigmoid")
KPCA.fit(scaled_dataset)

Y = KPCA.transform(scaled_dataset)


gmm = mixture.BayesianGaussianMixture(n_components=3, covariance_type='full').fit(scaled_dataset)
labels_gmm = gmm.predict(scaled_dataset)


kmeans = KMeans(n_clusters=3)
kmeans.fit(scaled_dataset)
labels_kmean = kmeans.predict(scaled_dataset)


max_index = None

for i,entry in enumerate(windowed_dataset):
    print("{}:{:02d}-{}:{:02d}\t".format(int(entry[0]/60),int(entry[0]%60),int((entry[0]+time_width)/60),\
          int((entry[0]+time_width)%60)),end='')
    print(" {}\t{}".format(labels_kmean[i],labels_gmm[i]))

exit(-1)
plt.scatter(Y[:,0],Y[:,1],c=labels.astype(np.float),cmap=plt.cm.Spectral)
plt.show()

exit(-1)

posit = [i for i in range(len(Y))]

start,end = 0,0


for entry in lc:
    while entry not in range(int(windowed_dataset[start][0]),int(windowed_dataset[start][0]+time_width)): start+=1
    end = start+1
    while end<len(windowed_dataset) and entry in range(int(windowed_dataset[end][0]),int(windowed_dataset[end][0]+time_width)): end+=1

    lc_posit[entry] = []
    for i in range(start,end):
        lc_posit[entry].append(i)


for val in sorted(lc_posit):
    print(val)
    for entry in lc_posit[val]: 
        print("{}\t{}".format(windowed_dataset[entry][0],labels_revised[entry]))
        #print("{}\t{}".format(windowed_dataset[entry][0],labels[entry]))
    print("\n")

print("\n\n")
exit(-1)

for i,entry in enumerate(labels):
    if windowed_dataset[i][0] in lc: print("(*)",end='')
    print("{}:{}".format(windowed_dataset[i][0],entry))
    print("\n")
