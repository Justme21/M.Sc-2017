#!/usr/bin/python


from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from datastore import DataStore
from scipy.cluster.hierarchy import dendrogram,fcluster,inconsistent,linkage
from sklearn import decomposition,manifold,preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os,re


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
        data_list[i] = ([x[0:] for x in dataset],[y[1] for y in annotation]) #remove time from entries

    return data_list


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


def getMaxContributer(row):
    max_entry,max_count = None,0
    for entry in set(row):
        if row.count(entry)>=max_count:
            max_entry = entry
            max_count = row.count(entry)
    return max_entry


def makeWindows(dataset,annotation,granularity,time_width=5,overlap=1):
    """Chunks a given dataset up into "windows" of time of length "time_width"
       "granularity" is how granular the data is, e.g. if the data is stored in .5 second incremements then granularity
       is .5.
       Overlap determines how much the windows overlap; if it's 2 then the entries that make up the second half of window j
       make up the first half of window j+1 """
    posit=0
    window_len =  math.ceil(time_width/granularity) #Assume time_width>granularity
    windowed_dataset, windowed_annotation = [],[]
    while posit<len(dataset):
        row,row_anno = [],[]
        j = 0
        while posit+j<len(dataset) and j<window_len: #dataset[posit+j][0]-dataset[posit][0]<window_len:
            if dataset[posit+j]==-10:
                posit += j+1
                j = 0
                row = []
                row_anno = []
            else:
                row += dataset[posit+j][1:]
                row_anno.append(annotation[posit+j])
                j += 1

        
        if posit+j<len(dataset) and  j==window_len: #dataset[posit+j][0]-dataset[posit][0] >= window_len:    
            row = restructure(row,window_len)
            windowed_dataset.append([dataset[posit][0]]+row) #want to include the timestep for reference
            posit+= int(window_len/overlap)#posit+=int(window_len/2) #Overlapping windows
           
            windowed_annotation.append(getMaxContributer(row_anno))
        else:
            posit += j

    return windowed_dataset,windowed_annotation


def KMeansTest(cluster_dataset,cluster_anno):

    n_clusters = 2

    range_n_clusters = [2,3]

    silhouette_avg = silhouette_score(cluster_dataset, cluster_anno)
    print("Basic silhouette average on data is: {}".format(silhouette_avg))

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
       #fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(cluster_dataset) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(cluster_dataset)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(cluster_dataset, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(cluster_dataset, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(cluster_dataset[:, 0], cluster_dataset[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=200)

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()

        cross_section = [[0 for _ in range(3)] for _ in range(n_clusters)]
        for lab,true in zip(cluster_labels,cluster_anno):
            cross_section[int(lab)][index_dict[true]]+=1

        for entry in cross_section:
            print(entry)


def isoMap(dataset,n_comp):
    isMap = manifold.Isomap(n_neighbors = 10,n_components = n_comp)
    return isMap.fit_transform(dataset)


def KPCA(dataset,kernel,n_comp):
    kpca = decomposition.KernelPCA(kernel=kernel, n_components = n_comp, gamma=10)
    return kpca.fit_transform(dataset)    



def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def hierarchicalCluster(dataset,cluster_anno):
    Z = linkage(dataset, 'average')
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    depth = 350
    fancy_dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
        max_d=depth#inconsistent(Z)
    )
    plt.show()

    for i in [2,3]:
        cluster_labels = fcluster(Z, i,criterion='maxclust')
        cross_section = [[0 for _ in range(3)] for _ in range(len(set(cluster_labels)))]
        for lab,true in zip(cluster_labels,cluster_anno):
            cross_section[int(lab)-1][index_dict[true]]+=1

        for entry in cross_section:
            print(entry)
        print("")


def affinityCluster(dataset,cluster_anno):
    af = AffinityPropagation(preference=-50).fit(dataset)
    cluster_centers_indices = af.cluster_centers_indices_
    cluster_labels = af.labels_

    n_clusters = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % n_clusters)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(cluster_anno, cluster_labels))
    print("Completeness: %0.3f" % metrics.completeness_score(cluster_anno, cluster_labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(cluster_anno, cluster_labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(cluster_anno, cluster_labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(cluster_anno, cluster_labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(dataset, cluster_labels, metric='sqeuclidean'))


    cross_section = [[0 for _ in range(3)] for _ in range(n_clusters)]
    for lab,true in zip(cluster_labels,cluster_anno):
        cross_section[int(lab)][index_dict[true]]+=1

    for entry in cross_section:
        print(entry)


def GMM(dataset,cluster_anno):
    for n_clusters in [2,3]:
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(dataset)
        cluster_labels = gmm.predict(dataset)

        cross_section = [[0 for _ in range(3)] for _ in range(n_clusters)]
        for lab,true in zip(cluster_labels,cluster_anno):
            cross_section[int(lab)][index_dict[true]]+=1

        print("AIC: {}".format(gmm.aic(dataset)))
        print("BIC: {}".format(gmm.bic(dataset)))

        for entry in cross_section:
            print(entry)
        print("")


index_dict = {"L":0,"F":1,"R":2}
sources = getDirectories()
sources = [x for x in sources if "D1" in x or "D2" in x]

data_list,anno_list = [],[]
temp_list = None
for entry in sources:
    temp_list = makeDataList(entry)
    for t_list in temp_list:
        data_list += t_list[0] + [-10]
        anno_list += t_list[1] + [-10]

granularity = .5
time_width = 5
overlap = 10
window_set,window_anno = makeWindows(data_list,anno_list,granularity,time_width,overlap)

n_comp = 3

dataset_np = np.array([entry[1:] for entry in window_set if entry != -10])
anno_list_np = np.array([entry for entry in window_anno if entry != -10])
#dataset_np = np.array([entry[1:] for entry in window_set])

#dataset_np = isoMap(dataset_np,n_comp)

#dataset_np = KPCA(dataset_np,"cosine",n_comp)

#scaled_dataset = preprocessing.StandardScaler().fit_transform(dataset_np)

#KMeansTest(dataset_np,anno_list_np)
#hierarchicalCluster(dataset_np,anno_list_np)
#affinityCluster(dataset_np,anno_list_np)
GMM(dataset_np,anno_list_np)


#plt.scatter([int(x) for x in labels],[index_dict[x] for x in anno_list])
#plt.show()
