#!/usr/bin/python

import math

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

            if "VEHICLE_DETECTION" in self.name and (entry[1] ==-1 and entry[2]==-1):
                entry[1] = 50
                entry[2] = 2.5

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


    def rescale(self):
        """Rescales the data in the datastore for use in the feature learner"""
        for i,entry in enumerate(self.file_content):
            self.file_content[i] = [entry[0]]+[2.0/(1+math.exp(-.5*x)) - 1 for x in entry[1:]]
    
    def rehab(self):
        """Cleans the PROC_LANE_DETECTION data and imputes missing values.
           Imputation is done by identifying, using the indicator feature, values we are confident in
           immediately before and after regions of missing data. We then assume the data increases/decreases
           linearly in the interim, computing the mean and adding this to the values to fill in the unknowns"""
        start,end,direc = -1,-1,None #direc is used for position on road since it is only discontinuous value
        scale = None
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
            try:
                end = list(self.file_content[b+1])
            except:
                end = list(self.file_content[b])
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
                    if direc<0: scale = -1
                    else: scale = 1
                    self.file_content[j][1] = round(self.file_content[j][1]-scale*road_width,3)            
