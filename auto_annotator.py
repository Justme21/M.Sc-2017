#!/usr/bin/python
import math
import matplotlib.pyplot as plt
import random


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

    def advance(self):
        self.index+=1
        if self.index>=len(self.file_content): self.index = None
        else:
            self.time = self.file_content[self.index][0]



def sampleFromDist(x_t_1,accel,chng,mean,std_dev):
    rnd_val = random.random()
    k = math.log(3)/(.165*.15) #.165 is approximately the observed magnitude averages for acceleration to both the left and right. REVISE THIS 
    if x_t_1 == "L":
        # chng negative => e^-chng has large value for change of small magnitude
        # and small value for large magnitude. Thus for small changes its is unlikely we
        # are still going left, and for large changes it is very likely
    #    print("{}\t{}\t{}".format(rnd_val,chng,math.e**(lmda*chng)))
        if rnd_val<1-2/(math.e**(k*(-chng))+1): return "L"
        else: return "F"
    elif x_t_1 == "R":
        if rnd_val<1-2/(math.e**(k*chng)+1): return "R" 
        else: return "F"
    else:
        #t_val = (1/math.sqrt(2*math.pi*(std_dev**2)))*math.e**(-((accel-mean)**2)/(2*(std_dev**2)))
        #We omit the normalisation so that the distribution we are sampling from has the same range as the random number
        t_val = math.e**(-((accel-mean)**2)/(2*(std_dev**2)))
        if accel<0 and rnd_val>t_val: return "L"
        elif accel>0 and rnd_val>t_val: return "R"
        else: return "F"


def probFromDist(z_t,x_t,mean,std_dev,road_width,car_width):
    z_l = (road_width/2)+z_t-(car_width/2)
    z_r = (road_width/2)-z_t-(car_width/2)
    k = math.log(3) #This is based on the assumption that when z_l/z_r = 1 the probability of being in the state should be 1

    if x_t not in {"R","L"}:
        return (1/math.sqrt(2*math.pi*(std_dev**2)))*math.e**(-((z_t-mean)**2)/(2*(std_dev**2)))
    else:
        if x_t == "R":
            z_rat = z_l/z_r
        else:
            z_rat = z_r/z_l
        
        if (z_r <.01 and x_t=="R") or (z_l<.01 and x_t=="L"):
            return 1.0  
        return 1-2/(math.e**(k*z_rat)+1) #Functional form of hyperbolic tan (tanh) modified to include k as the slope parameter

def maxState(w,particle_dict):
    count_dict = {"L":0,"R":0,"F":0}
    cur_max = -1
    max_state = None
    prev_w = 0
    for entry in particle_dict:
        count_dict[particle_dict[entry]] += w[entry]-prev_w
        prev_w = w[entry]

    for entry in count_dict:
        if count_dict[entry]>cur_max:
            cur_max = count_dict[entry]
            max_state = entry
    
    #return count_dict
    return (max_state,cur_max)


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
entries_of_interest = [[0,1,2,3,4],[0,1,3,4],[0,1,2,3,4]]


#Creating a list with access to all the data_folders we are interested in
datastore_dict = {}
datastore_list = []
for i, entry in enumerate(files_of_interest):
    datastore_dict[entry] = DataStore("{0}{1}.txt".format(folder_loc,entry),entries_of_interest[i])
    datastore_list.append(datastore_dict[entry])

###NOTE: Here we are assuming that the values observed are simultaneous
###      i.e. we assume that the i^th entry in Z is the position observed in the same
###      timestep that acceleration a_i was observed

prp_accel = datastore_dict["RAW_ACCELEROMETERS"].getAll(3)
Z = datastore_dict["PROC_LANE_DETECTION"].getAll(1)
road_width = datastore_dict["PROC_LANE_DETECTION"].getAll(2)
data_qual = datastore_dict["PROC_LANE_DETECTION"].getAll(3)

car_width = 1.75 #given in metres

prp_accel = [9.81 * x for x in prp_accel] #Translating this to m/s for common units
accel_chng = [0]+[prp_accel[i]-prp_accel[i-1] for i in range(1,len(prp_accel))]


###NOTE: While this all works fine in theory, in practise the datasets used tend to have missing/
# incorrect values. In the dataset this is represented by a 0 in the last column of the Lane Data
# The issue is that during this time the accelerometer readings are unaffected, but the Z values 
# revert to 0, and the road_width is also incorrect. I have not yet decided how to resolve this
mean_accel = sum(prp_accel)/len(prp_accel)
std_dev_accel = math.sqrt((1.0/(len(prp_accel)-1))*sum([(x-mean_accel)**2 for x in prp_accel]))

mean_Z = sum(Z)/len(Z)
std_dev_Z = math.sqrt((1.0/(len(Z)-1))*sum([(x-mean_Z)**2 for x in Z]))


n = 50 #This is probably too many particles, but it works
w_old = [(i+1)*1.0/n for i in range(n)]
w_new = [_ for _ in range(n)]
particle_dict_old = {}
particle_dict_new = {}
rand_val = None
state = None
state_path = []

for i in range(n):
    particle_dict_old[i] = "F"

while None not in advanceAll(datastore_list):
    w_norm = 0
    t = [source.index for source in datastore_list]
    for i in range(n):
        rand_val = random.random()
        j = 0
        while w_old[j]<rand_val: j += 1
        x_t_1 = particle_dict_old[j]
        x_t = sampleFromDist(x_t_1,prp_accel[t[0]],accel_chng[t[0]],mean_accel,std_dev_accel)
        w_new[i] = probFromDist(Z[t[1]],x_t,mean_Z,std_dev_Z,road_width[t[1]],car_width)
        w_norm += w_new[i]
        particle_dict_new[i] = x_t

    for i in range(len(w_new)):
        if i== 0:
            w_new[i]/= w_norm
        else:
            w_new[i] = w_new[i-1] + w_new[i]/w_norm

    particle_dict_old = dict(particle_dict_new)
    w_old = list(w_new)
    particle_dict_new = {}
    w_new = [_ for _ in range(n)]
    time = max([source.time for source in datastore_list])
    if data_qual[t[1]] in {0,-9}: print("*"),
    print("{}:{:02d} ({}) \t {}".format(int(time/60),int(time%60),time,maxState(w_old,particle_dict_old)))

    
