#!/usr/bin/python

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition,manifold,preprocessing
from sklearn.cluster import KMeans #used to perform k-means
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster



class DataStore():

    def __init__(self,file_name,content_list):
        self.name = file_name #This is the full address of datasource
        self.file_content = [] #Will hold the desired contents from the file
        op_file = open("{}".format(file_name))
        for entry in op_file: #Iterate through the rows in the file
            #We only include the entries corresponding to the 0-indexed columns specified in content_list
            entry = [float(x) for i,x in enumerate(entry.split()) if i in content_list]

            if -9.0 not in entry:
                self.file_content.append(entry)

        if "PROC_LANE_DETECTION" in self.name:
            self.rehab()
            self.file_content = [entry[:-1] for entry in self.file_content] #Omit unrelated variable
        
        self.index = 0 #Initialise the index to 0
        self.time = self.file_content[0][0]

    def getAll(self,i):
        tot_list = []
        for row in self.file_content:
            if -9 not in row: #This is a sloppy way to omit the problem entries in PROC_LANE_DETECTION
                tot_list.append(row[i])

        return tot_list

    def getRow(self,index=None):
        if index is None: index = self.index
        return self.file_content[index]

    def getIndex(self,time=None):
        i = 0
        min_dist = None
        if time is None: time=self.time
        while min_dist is None or math.fabs(time-self.file_content[i][0])<min_dist:
            min_dist = math.fabs(time-self.file_content[i][0])
            i+=1
        return i-1
 
    def advance(self):
        self.index+=1
        if self.index>=len(self.file_content): self.index = None
        else:
            self.time = self.file_content[self.index][0]

    def reverse(self):
        self.index-=1
        if self.index<0: self.index = 0
        self.time = self.file_content[self.index][0]

    
    def rehab(self):
        start,end,direc = -1,-1,None
        i = 0
        entry, change = None,None
        err_index,break_list = [],[]     

        #-1 means calibrating. Assume this only happens at start of dataset
        while self.file_content[0][-1] == -1:
            del self.file_content[0]
        
        #A list of all the points where the detection software gives bad output
        err_index = [j for j,entry in enumerate(self.file_content) if entry[-1] in [0,-1]]
        
        while i<len(err_index):
            start = err_index[i]
            while i+1<len(err_index) and err_index[i+1]-err_index[i]<5: i+=1
            end = err_index[i]
            break_list.append((start,end))
            i+=1

        for (a,b) in break_list:
            start = list(self.file_content[a-1])
            end = list(self.file_content[b+1])
            road_width = end[3]
 
            #Only need direc for car position as is discontinuous
            direc = start[1]-self.file_content[a-2][1] #Instantaneous slope before data loss            
            if (direc<0 and start[1]<0 and end[1]>0) or (direc>0 and start[1]>0 and end[1]<0):
                end[1] += (direc/math.fabs(direc))*road_width #Add or subtract road width as necessary

            change = [(end[x] - start[x])*1.0/((b+1)-(a-1)) for x in range(len(start))]
            for j in range(a,b+1):
                self.file_content[j] = [round(self.file_content[j-1][x] + change[x],3) for x in range(len(change)-1)] +[3.0]
                if math.fabs(self.file_content[j][1])>road_width/2:
                    self.file_content[j][1] = round(self.file_content[j][1]-(direc/math.fabs(direc))*road_width,3)            
       
     

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


def advanceIncr(source_list):
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


folder_loc = "Datasets/UAH-DRIVESET-v1/D1/20151111123124-25km-D1-NORMAL-MOTORWAY/"

files_of_interest = ["RAW_ACCELEROMETERS","PROC_LANE_DETECTION","PROC_VEHICLE_DETECTION"]
entries_of_interest = [[0,2,3,4,5,6,7,8,9,10],[0,1,2,3,4],[0,1,2,3,4]]
#entries_of_interest = [[0,2,3,4,5,6,7,10],[0,1,2,3,4],[0,1,2,3,4]]

datastore_list = []
for i,entry in enumerate(files_of_interest):
    datastore_list.append(DataStore("{0}{1}.txt".format(folder_loc,entry),entries_of_interest[i]))

dataset = []
row_list = []

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
prev_angles = [0,0,0]
car_width = 1.75


car_rat = None

for i,entry in enumerate(dataset):
    extra_vals = []
    extra_vals.append(entry[1]-prev_accl[0])
    extra_vals.append(entry[2]-prev_accl[1])
    extra_vals.append(entry[3]-prev_accl[2])

    extra_vals.append(entry[4]-prev_accl_kal[0])
    extra_vals.append(entry[5]-prev_accl_kal[1])
    extra_vals.append(entry[6]-prev_accl_kal[2])

    extra_vals.append(entry[7]-prev_angles[0])
    extra_vals.append(entry[8]-prev_angles[1])
    extra_vals.append(entry[9]-prev_angles[2])


    road_width = entry[12]
    z_t = entry[10]
    z_l = (road_width/2)+z_t-(car_width/2)
    z_r = (road_width/2)-z_t-(car_width/2) 
    extra_vals.append(z_l/z_r)

    dataset[i]+=extra_vals

posit=0
time_width = 1 #seconds
window_len =  math.ceil(time_width/granularity) #Assume time_width>granularity
windowed_dataset = []
while posit<len(dataset):
    row = []
    j = 0
    while posit+j<len(dataset) and j<window_len: #dataset[posit+j][0]-dataset[posit][0]<window_len:
        row += dataset[posit+j][1:]
        j += 1
    
 
    if posit+j<len(dataset) and  j==window_len: #dataset[posit+j][0]-dataset[posit][0] >= window_len:    
        windowed_dataset.append([dataset[posit+j-1][0]]+row) #want to include the timestep for reference
        posit+=window_len#posit+=int(window_len/2) #Overlapping windows
    else:
        posit += j
       
   
#dataset_np = np.array(dataset[1:])
dataset_np = np.array([entry[1:] for entry in windowed_dataset])


#Scales data to have 0 mean and unit variance
#scaled_dataset = preprocessing.MaxAbsScaler().fit_transform(dataset_np)
scaled_dataset = preprocessing.StandardScaler().fit_transform(dataset_np)

n_comp = 2


isMap = manifold.Isomap(n_neighbors = 10,n_components = n_comp)
Y = isMap.fit_transform(scaled_dataset)
#Y = isMap.fit_transform(dataset_np)

t= isMap.transform(np.identity(scaled_dataset.shape[1]))


for i in range(n_comp):
    for j,entry in enumerate(t.T[i]):
        print("{}: {} \t {}".format(j,entry,math.fabs(entry/sum(math.fabs(x) for x in t.T[i]))))
    print("\n")
    


kmeans = KMeans(n_clusters=3)
kmeans.fit(Y)
labels = kmeans.predict(Y)

posit = [i for i in range(len(Y))]

lc = [35,55,65,83,98,104,139,164,177,208,257,272,305,371,376]
lc_posit = []
start,end = 0,0

for entry in lc:
    while entry> windowed_dataset[start+1][0]: start+=1
    end = start+1
    while entry>windowed_dataset[end+1][0]: end+=1

    for i in range(start-1,end+1):
        lc_posit.append(i)


#for entry in lc_posit:
#    print("{}\t{}".format(windowed_dataset[entry][0],labels[entry]))

#print("\n\n")

#for entry in labels:
#    print(entry)


#for i in range(n_comp):
#    plt.plot(posit,Y[:,i],'b-')
#    plt.plot(lc_posit,Y[lc_posit,i],'r.')
#    plt.title("This is {}".format(i))
#    plt.show()


plt.scatter(posit,Y[:,1],c=labels.astype(np.float),cmap=plt.cm.Spectral)
#plt.scatter(Y[:,0],Y[:,1],c=labels.astype(np.float),cmap=plt.cm.Spectral)
plt.show()



#printable = True
#for i in range(len(dataset)):
#    window_it = 0
#    while printable and dataset[i][0]>windowed_dataset[window_it][0]:
#        window_it+=1
#        if window_it == len(windowed_dataset):
#            printable = False
#    if printable:
#        print("{}:{:02d}\t{}".format(int(dataset[i][0]/60),int(dataset[i][0]%60),labels[window_it]))

#for i,entry in enumerate(dataset_np):
#    print("{}\t{}:{}\t{}".format(i,int(entry[0]/60),int(entry[0]%60),labels[i]))
