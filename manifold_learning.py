#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold,preprocessing
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

        self.index = 0 #Initialise the index to 0
        self.time = self.file_content[0][0]

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
        if self.index>=len(self.file_content): self.index = None
        else:
            self.time = self.file_content[self.index][0]


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

folder_loc = "Datasets/UAH-DRIVESET-v1/D2/20151120133502-26km-D2-AGGRESSIVE-MOTORWAY/"

files_of_interest = ["RAW_ACCELEROMETERS","PROC_LANE_DETECTION","PROC_VEHICLE_DETECTION"]
entries_of_interest = [[0,2,3,4,5,6,7,8,9,10],[0,1,2,3],[0,1,2,3,4]]

datastore_list = []
for i,entry in enumerate(files_of_interest):
    datastore_list.append(DataStore("{0}{1}.txt".format(folder_loc,entry),entries_of_interest[i]))

dataset = []
row_list = []
for entry in datastore_list: row_list += entry.getRow()[1:]
dataset.append([min(x.time for x in datastore_list)]+row_list)

while None not in advanceAll(datastore_list):
    row_list = []
    for entry in datastore_list: row_list+= entry.getRow()[1:]
    dataset.append([min([x.time for x in datastore_list])]+row_list)

posit=0
window_len = 30
windowed_dataset = []
while posit<len(dataset):
    row = []
    j = 0
    while posit+j<len(dataset) and j<window_len and dataset[posit+j][0]-dataset[posit][0]<window_len:
        row += dataset[posit+j][1:]
        j += 1
    
    if j == window_len:    
        windowed_dataset.append([dataset[posit+j-1][0]]+row) #want to include the timestep for reference
        posit+=int(window_len/2) #Overlapping windows
    else:
        posit += j
       
    
#dataset_np = np.array(dataset[1:])
dataset_np = np.array([entry[1:] for entry in windowed_dataset])


#Scales data to have 0 mean and unit variance
scaled_dataset = preprocessing.MaxAbsScaler().fit_transform(dataset_np)


Y = manifold.Isomap(n_neighbors = 5,n_components = 2).fit_transform(scaled_dataset)

kmeans = KMeans(n_clusters=3)
kmeans.fit(Y)
labels = kmeans.predict(Y)

plt.scatter(Y[:,0],Y[:,1],c=labels.astype(np.float))
plt.show()

#This is the linkage used to measure the distance between clusters
Z = linkage(Y, 'ward') 

#plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
Z_tree = dendrogram(
    Z,
    p=2,
    truncate_mode = 'level',
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)

plt.axhline(y=55, c='k')

plt.show()


k=3
point_clas = fcluster(Z, k, criterion='maxclust')


printable = True
for i in range(len(dataset)):
    window_it = 0
    while printable and dataset[i][0]>windowed_dataset[window_it][0]:
        window_it+=1
        if window_it == len(windowed_dataset):
            printable = False
    if printable:
        print("{}\t{}".format(i,window_it))
        print("{}:{}\t{}\t{}".format(int(dataset[i][0]/60),int(dataset[i][0]%60),\
                                 labels[window_it],point_clas[window_it]))

#for i,entry in enumerate(dataset_np):
#    print("{}\t{}:{}\t{}".format(i,int(entry[0]/60),int(entry[0]%60),labels[i]))
