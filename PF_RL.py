#!/usr/bin/python

import math
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import decomposition,preprocessing

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
       


class Feature_Learner():
    def __init__(self):
        self.total_reward = 0

        self.weights = None
        self.factors = []
        self.reward = 0
        self.q = 0

        self.alpha = .00001
        self.gamma = .99

        self.action_list = ["L","F","R"]
        self.reward_list = []
        self.register = [0,0,0]
        self.act_dist = [0,0,0]


    def setFactors(self,factors):
        self.factors = factors+[1]
        if self.weights is None:
            self.setWeights()


    def setWeights(self):
        self.weights = {}
        for act in self.action_list:
            self.weights[act] = [.1 for _ in self.factors]


    def getQ(self,act):
        q_sum = 0
        for (weight,factor) in zip(self.weights[act],self.factors):
            q_sum += weight*factor

        return q_sum        

    
    def getMaxQ(self):
        max_q,temp_q,max_act = None,0,0
        for i in np.random.choice(["L","F","R"],size=3,replace=False):
            temp_q = self.getQ(i)
            if max_q == None or temp_q>max_q:
                max_q = temp_q
                max_act = i

        return max_q,max_act
        

    def move(self,act,true_act):
        if true_act is None:
            return 0
        elif act == true_act:
            return 5
        elif act+true_act in ["RL","LR"]:
            return 2
        else:
            return -20


    def act(self,true_act):        
        self.q,max_act = self.getMaxQ()
        
        self.reward = self.move(max_act,true_act)
        self.total_reward += self.reward
        self.reward_list.append(self.reward)

        if self.reward >0: self.register[0] += 1
        elif self.reward == 0: self.register[1] += 1
        else: self.register[2] += 1

        if max_act == "L": self.act_dist[0] += 1
        elif max_act == "F": self.act_dist[1] += 1
        else: self.act_dist[2] += 1

        return max_act        


    def learn(self,act):
        err_mag = self.alpha*(self.reward+self.gamma*(self.getMaxQ()[0]) - self.q)
        for i in range(len(self.weights[act])):
            self.weights[act][i] += err_mag*self.factors[i]


    def runThrough(self,dataset,annotation):
        act = "F" 
        for entry,answer in zip(dataset,[None]+annotation):
            if answer is not None:
                act = self.act(answer)
            self.setFactors(entry)
            self.learn(act) 


    def explosionCheck(self):
        for act in self.weights:
            for entry in self.weights[act]:
                if np.isnan(entry) or entry>100000: return True
        return False


    def run(self,episodes,dataset,annotation):
        for i in range(episodes):
            if i%100 == 0 :
                for j in range(len(self.register)): self.register[j]/=100 
                for j in range(len(self.act_dist)): self.act_dist[j]/=100
                print("Episode #{}\t{}\t{}\t{}".format(i,self.total_reward,self.register,self.act_dist))
                self.register = [0,0,0]
            self.total_reward = 0
            self.runThrough(dataset,annotation)
            self.reward_list += [self.total_reward]
            if self.explosionCheck():
                print("Weight explosion after {} episodes".format(i))
                break
        return self.reward_list


class ParticleFilter():
    def __init__(self):
        self.dataset = None

        #self.car_width = 1.6
        self.num_particles = 100

        self.k_prob = None
        self.k_samp = None
        self.f_k_prob = None
        self.f_k_samp = None
       
        self.weights = []
        self.particle_dict = {}
        self.temp_weights = []
        self.temp_partlce_dict = {}

        self.accl = []
        self.accl_change = []
        self.lane_posit = []
        self.road_width = []

        self.register = [0,0,0]


    def extractValues(self):
        accl = []
 
        for entry in self.dataset:
            accl.append(entry[1])
            self.lane_posit.append(entry[6])
            self.road_width.append(entry[8])

        self.accl += [9.81 * x for x in accl]
        self.accl_change += [0] + [9.81*(accl[i] - accl[i-1]) for i in range(1,len(accl))]


    def initialise(self,dataset):
        self.dataset = dataset 
        self.weights = [(i+1)*(1.0/self.num_particles) for i in range(self.num_particles)]
        for i in range(self.num_particles): self.particle_dict[i] = "F"
        self.extractValues()
        
        z_rat_list = []
        z_l,z_r = 0,0
        for road_width,z_t in zip(self.road_width,self.lane_posit):
            z_l = (road_width/2)+z_t#-(self.car_width/2)
            z_r = (road_width/2)-z_t#-(self.car_width/2)
            if z_l<0: z_l = 0
            if z_r<0: z_r = 0
           
            z_rat_list.append(max(z_l/max(.001,z_r),z_r/max(.001,z_l)))

        mean_chng = sum([math.fabs(x) for x in self.accl_change])/len(self.accl_change)
        mean_accl = sum(self.accl)/len(self.accl)
        mean_lane_posit = sum(self.lane_posit)/len(self.lane_posit)
        mean_rat = sum(z_rat_list)/len(z_rat_list)

        #print("CHNG: {}".format(mean_chng))
        #print("ACCL: {}".format(mean_accl))
        #print("LNE_POS: {}".format(mean_lane_posit))
        #print("RAT: {}".format(mean_rat))
        #exit(-1)

        #k_samp = math.log(3)/(.165*.15) #.165 is approximately the observed magnitude averages for acceleration to both the left and right. REVISE THIS 
        self.k_samp = math.log(3)/mean_chng
        #k = math.log(3) #This is based on the assumption that when z_l/z_r = 1 the probability of being in the state should be 1
        self.k_prob = math.log(3)/mean_rat

        self.f_k_samp = math.log(3)/mean_accl
        self.f_k_prob = math.log(3)/mean_lane_posit


    def sampleFromDist(self,x_t_1,accel,chng):
        rnd_val = random.random()
        #k = math.log(3)/(.165*.15) #.165 is approximately the observed magnitude averages for acceleration to both the left and right. REVISE THIS 
        if x_t_1 == "L":
            # chng negative => e^-chng has large value for change of small magnitude
            # and small value for large magnitude. Thus for small changes its is unlikely we
            # are still going left, and for large changes it is very likely
            if rnd_val<1-2/(math.e**(self.k_samp*(-chng))+1): return "L"
            else: return "F"
        elif x_t_1 == "R":
            if rnd_val<1-2/(math.e**(self.k_samp*chng)+1): return "R"
            else: return "F"
        else:
            #t_val = (1/math.sqrt(2*math.pi*(std_dev**2)))*math.e**(-((accel-mean)**2)/(2*(std_dev**2)))
            #We omit the normalisation so that the distribution we are sampling from has the same range as the random number
            t_val = 2/(math.e**(self.f_k_samp*math.fabs(accel))+1)
            if rnd_val>t_val: return "F"
            else:
                if accel<0: return "L"
                else: return "R"


    def probFromDist(self,z_t,x_t,road_width):
        z_l = (road_width/2)+z_t#-(self.car_width/2)
        z_r = (road_width/2)-z_t#-(self.car_width/2)
        #k = math.log(3) #This is based on the assumption that when z_l/z_r = 1 the probability of being in the state should be 1
        if z_l < 0: z_l = 0
        if z_r < 0: z_r = 0

        if x_t not in {"R","L"}:
            return 2/(math.e**(self.f_k_prob*math.fabs(z_t)))
        else:
            if x_t == "R":
                z_rat = z_l/max(.001,z_r)
            else:
                z_rat = z_r/max(.001,z_l)

            if (math.fabs(z_r)<.001 and x_t=="R") or (math.fabs(z_l)<.001 and x_t=="L"):
                return 1.0
            
            return 1-2/(math.e**(self.k_prob*z_rat)+1) #Functional form of hyperbolic tan (tanh) modified to include k as the slope parameter


    def dfK(self,k,x):
        return math.fabs(2*x*(math.e**(k*x))/((1+math.e**(k*x))**2))

    def train(self,data_index,prev_assign,pf_assign,annote):
        err_mag = None
        turns = ["R","L"]
        if pf_assign == annote:
            err_mag = -5
            self.register[0] += 1
        elif pf_assign+annote in ["LR","RL"]:
            err_mag = 5
            self.register[1] += 1
        else:
            err_mag = 20
            self.register[2] += 1

        #Rule 3: Decrease k_samp
        if prev_assign in turns and prev_assign == pf_assign and annote == "F":
            self.k_samp -= err_mag*self.dfK(self.k_samp,math.fabs(self.accl_change[data_index]))
        #Rule 4: Decrease k_prob <- We guess L/R but true is R/L
        elif pf_assign in turns and annote in turns and pf_assign != annote:
            self.k_prob -= err_mag*self.dfK(self.k_prob,self.lane_posit[data_index])
        #Rule 5: Decrease f_k_samp
        elif prev_assign == "F" and pf_assign == "F" and annote in turns:
            self.f_k_samp -= err_mag*(self.dfK(self.f_k_samp,math.fabs(self.accl[data_index])))
        #Rule 6: Increase f_k_samp
        elif prev_assign == "F" and pf_assign in turns and annote == "F":
            self.f_k_samp += err_mag*(self.dfK(self.f_k_samp,math.fabs(self.accl[data_index])))
        #Rule 7: Increase k_samp
        elif prev_assign in turns and annote == prev_assign and pf_assign == "F":
            self.k_samp += err_mag*self.dfK(self.k_samp,math.fabs(self.accl_change[data_index]))
        #Rule 8: Increase f_k_prob
        elif prev_assign in turns and pf_assign == "F" and annote!= prev_assign:
            self.f_k_prob += err_mag*(self.dfK(self.f_k_prob,math.fabs(self.lane_posit[data_index])))
        
        #Not real rules.... may or may not be worth including
        #elif pf_assign in turns and pf_assign == annote:
        #    self.k_prob += self.alpha*.1*self.dfK(self.k_prob,self.lane_posit[data_index])
        #elif pf_assign == "F" and pf_assign == annote:
        #    self.f_k_prob += self.alpha*.1*(-self.dfK(self.f_k_prob,math.fabs(self.lane_posit[data_index])))
 
        #print("F_K_SAMP: {}".format(self.f_k_samp))

    def singleIt(self,data_index):
        w_norm,x_t,x_t_1 = 0,0,0
        rand_val = None
        self.temp_particle_dict = {}
        self.temp_weights = [0 for _ in range(self.num_particles)]

        for i in range(self.num_particles):
            rand_val = random.random()
            j = 0
            while j<len(self.weights) and self.weights[j]<rand_val: j += 1
            x_t_1 = self.particle_dict[j]
            x_t = self.sampleFromDist(x_t_1,self.accl[data_index],self.accl_change[data_index])
            self.temp_weights[i] = self.probFromDist(self.lane_posit[data_index],x_t,self.road_width[data_index])
            
            if self.temp_weights[i] == 0:
                print("{}\t{}\t{}\t{}".format(self.lane_posit[data_index],x_t,self.prob_params,self.road_width[data_index]))
                exit(-1) 
            w_norm += self.temp_weights[i]
            self.temp_particle_dict[i] = x_t

            
        for i in range(len(self.temp_weights)):
            if i== 0:
                self.temp_weights[i]/= w_norm
            else:
                self.temp_weights[i] = self.temp_weights[i-1] + self.temp_weights[i]/w_norm


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
    ahd_veh = [0,0]

    #car_width = 1.6 #Hardcoded car width
    max_dist = max([entry[10] for entry in dataset]) #The max distance a car is ever away from the ego 
    max_time = max([entry[11] for entry in dataset]) #The max time to collision


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
        if ahd_veh[0] == -1 or entry[10] == -1:
            extra_vals+= [max_dist,max_time] #Here distance and speed were both -1
        else:
            extra_vals.append(entry[10]-ahd_veh[0])
            extra_vals.append(entry[11]-ahd_veh[1])

        #Include ration of distance from the left side of the road to that from the right side of the road
        road_width = entry[9]
        z_t = entry[7]
        z_l = (road_width/2)+z_t#-(car_width/2)
        z_r = (road_width/2)-z_t#-(car_width/2) 

        #This isn't ideal but the inaccuracy arising from the readings means
        # No car width satisfies every value
        if z_l < 0 : z_l = 0
        if z_r <0 : z_r = .001
        extra_vals.append(z_l/z_r)

        #Augment values
        dataset[i]+=extra_vals

        #Reset values for next iteration
        prev_accl = [entry[1],entry[2],entry[3]]
        prev_accl_kal = [entry[4],entry[5],entry[6]]
        ahd_veh = [entry[10],entry[11]]

    return dataset


def scaleAndChangeBase(dataset,n_comp = None):
    dataset = preprocessing.StandardScaler().fit_transform(dataset)
    KPCA = decomposition.KernelPCA(n_components = n_comp,kernel="sigmoid")
    KPCA.fit(dataset)
    
    #lmd = KPCA.lambdas_
    #sum_lmd = sum(lmd)
    #for i,entry in enumerate(lmd):
    #    if i>0: lmd[i]+=lmd[i-1]
    #for i,entry in enumerate(lmd): print("{}:{}".format(i,entry/sum_lmd))
    #exit(-1)

    Y = KPCA.transform(dataset)
    return Y


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

granularity = .5
dataset = granularise(dataset,granularity)

dataset = augmentDataset(dataset)
annotation = getAnnotation(folder_loc)


while dataset[0][0]<annotation[0][0]: del dataset[0]

#pseudo_set = scaleAndChangeBase([x[1:] for x in dataset],n_comp = 25)

#dataset = [[dta[0]]+list(psudo) for dta,psudo in zip(dataset,pseudo_set)]

windowed_dataset = makeWindows(dataset,granularity,overlap=1)
w_annotation = windowAnnotation(annotation,[x[0] for x in windowed_dataset])


dataset = [x[1:] for x in dataset]
w_dataset = [x[1:] for x in windowed_dataset]

state_list = None
state_ind,state = None,None
win_count = 0
state_sum = 0
rnd = 0

trans_dict = {"L":0,"F":1,"R":2}

pf = ParticleFilter()
pf.initialise(dataset)
for _ in range(1):
    win_count = 0
    win_list = []
    for i in range(len(dataset)):
        pf.singleIt(i)

        state_list = [0,0,0]
        for entry in pf.temp_particle_dict:
            state_list[trans_dict[pf.temp_particle_dict[entry]]] += 1
        state_sum = sum(state_list)
        state_list = [x/state_sum for x in state_list]
 
        for entry in pf.particle_dict:
            pf.train(i,pf.particle_dict[entry],pf.temp_particle_dict[entry],annotation[i][1])
            rnd = random.random()
            if rnd<(1-state_list[trans_dict[pf.temp_particle_dict[entry]]]): pf.temp_particle_dict[entry] = annotation[i][1]

        pf.particle_dict = dict(pf.temp_particle_dict)
        pf.weights = list(pf.temp_weights)


        state_list = [0,0,0]
        for entry in pf.temp_particle_dict:
            if pf.temp_particle_dict[entry] == "L": state_list[0] += 1
            elif pf.temp_particle_dict[entry] == "F": state_list[1] += 1
            else: state_list[2] += 1

        print("{}\t{}\t{}".format(i,state_list,annotation[i][1]),end='')
        state_ind = np.argmax(state_list)
        if state_ind == 0: state = "L"
        elif state_ind == 1: state = "F"
        else: state = "R"


        if state == annotation[i][1]: 
            win_count+=1
            win_list.append(i)
            print("*")
        else: print("")

    print("")
    print("K_PROB: {}".format(pf.k_prob))
    print("K_SAMP: {}".format(pf.k_samp))
    print("F_K_PROB: {}".format(pf.f_k_prob))
    print("F_K_SAMP: {}".format(pf.f_k_samp))
    print("WIN: {}/{}".format(win_count,len(dataset)))
    #print("LIST: {}".format(win_list))
    print("\n\n")
     

exit(-1)
learner = Feature_Learner()
reward_list = learner.run(20000,w_dataset,[x[1] for x in w_annotation])

print("Max Reward Achieved in Windowed dataset: {}".format(max(reward_list)))

plt.plot(reward_list)
plt.title("Total Rewards for 20000 episodes on windowed dataset")
plt.show()

