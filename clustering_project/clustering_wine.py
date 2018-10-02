import numpy as np
from sklearn import preprocessing


file = open("wine.data")
raw_data = []
for row in file:
	row = row.strip("\r\n")
	raw_data.append(row.split(','))

for item in raw_data:
	tmp = item[0]
	item.append(tmp)
	del item[0]

print raw_data

data=[]
for item in raw_data:
	tmp = []
	for value in item:
		tmp.append(float(value)) 
	data.append(tmp)
"""
scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
raw_data = scaler.fit_transform(data)
print raw_data
"""
raw_data = data




def distanceCalculator(input_set,attribute_index):
	input_num = len(input_set)
	distance_matrix = [([0.0]*input_num) for j in range(input_num)]
	for i in range(0, input_num):
		for j in range(0, input_num):
			distance = 0.0
			for index in attribute_index:
				"""
				print i
				print input_set[i][index]
				print j
				print input_set[j][index]
				"""
				diff = float(input_set[i][index]) - float(input_set[j][index])
				distance = distance + pow(diff,2)
			distance = pow(distance,0.5)
			distance_matrix[i][j] = distance
	# print distance_matrix
	return distance_matrix

# initialize clusters so that each item is a cluster
def initialClusters(num):
	res = []
	for i in range(0,num):
		res.append([i])
	return res

# calculate the minimum distance of two clusters
def minDistanceCalculator(one, two, distance_matrix):
	min_distance = distance_matrix[one[0]][two[0]]
	for i in one:
		for j in two:
			if(min_distance > distance_matrix[i][j]):
				min_distance = distance_matrix[i][j]
	return min_distance

def maxDistanceCalculator(one, two, distance_matrix):
	max_distance = distance_matrix[one[0]][two[0]]
	for i in one:
		for j in two:
			if(max_distance < distance_matrix[i][j]):
				max_distance = distance_matrix[i][j]
	return max_distance

def averageDistanceCalculator(one, two, distance_matrix):
	total_distance = 0
	for i in one:
		for j in two:
			total_distance = total_distance + distance_matrix[i][j]
	average_distance = total_distance/(len(one)*len(two))
	return average_distance





def clustering(input_clusters,distance_matrix,linkage_type):
	# print input_clusters
	combine_list = [input_clusters[0],input_clusters[1]]

	if linkage_type == "single":
		global_min_distance = minDistanceCalculator(input_clusters[0],input_clusters[1],distance_matrix)
	elif linkage_type == "complete":
		global_min_distance = maxDistanceCalculator(input_clusters[0],input_clusters[1],distance_matrix)
	elif linkage_type == "average":
		global_min_distance = averageDistanceCalculator(input_clusters[0],input_clusters[1],distance_matrix)


	for one in range(0, len(input_clusters)):
		for two in range(0,len(input_clusters)):
			if (one != two):
				if linkage_type == "single":
					local_min_distance = minDistanceCalculator(input_clusters[one],input_clusters[two],distance_matrix)
				elif linkage_type == "complete":
					local_min_distance = maxDistanceCalculator(input_clusters[one],input_clusters[two],distance_matrix)
				elif linkage_type == "average":
					local_min_distance = averageDistanceCalculator(input_clusters[one],input_clusters[two],distance_matrix)

				if (global_min_distance > local_min_distance):
					global_min_distance = local_min_distance
					combine_list = [input_clusters[one],input_clusters[two]]
	input_clusters.remove(combine_list[0])
	input_clusters.remove(combine_list[1])
	combine_list[0].extend(combine_list[1])
	input_clusters.append(combine_list[0])
	return input_clusters




def singleLinkageClustering(input_set,attribute_index,target_num):
	distance_matrix = distanceCalculator(input_set,attribute_index)
	clusters = initialClusters(len(input_set))
	while(len(clusters) > target_num):
		cluster = clustering(clusters,distance_matrix,"single")
	return cluster

def completeLinkageClustering(input_set,attribute_index,target_num):
	distance_matrix = distanceCalculator(input_set,attribute_index)
	clusters = initialClusters(len(input_set))
	while(len(clusters) > target_num):
		cluster = clustering(clusters,distance_matrix,"complete")
	return cluster

def averageLinkageClustering(input_set,attribute_index,target_num):
	distance_matrix = distanceCalculator(input_set,attribute_index)
	clusters = initialClusters(len(input_set))
	while(len(clusters) > target_num):
		cluster = clustering(clusters,distance_matrix,"average")
	return cluster




def clusterIndexCalculator(instance,seeds):
	min_distance = 0
	res = 0
	for index,value in enumerate(seeds[0]):
		diff = float(instance[index])-value
		min_distance = min_distance + pow(diff,2)
 
	for seed_index, item in enumerate(seeds):
		distance = 0
		for index, value in enumerate(item):
			diff = float(instance[index])-value
			distance = distance + pow(diff,2)
		if(distance < min_distance):
			min_distance = distance
			res = seed_index
	return res
	    
def centerCalculator(clusters,attribute_index,input_set):
	centers = []
	for item in clusters:
		# if no point is assigned to this cluster, random to another one
		if(len(item) == 0):
			candidate = random.randint(0, len(input_set))
			position = []
			for index in attribute_index:
				position.append(input_set[index])
			centers.append(position)
		else:
			position = []
			for index in attribute_index:
				tmp = 0.0
				for instance in item:
					tmp = tmp + float(input_set[instance][index])
				position.append(tmp/len(item))
			centers.append(position)
	return centers

def KMeansFunctionCalculator(cluster,centers,input_set,attribute_index):
	res = 0.0
	for index,item in enumerate(cluster):
		for instance in item:
			for attr in attribute_index:
				diff = float(input_set[instance][attr]) - centers[index][attr]
				res = res + pow(diff,2)
	return res
	



def LloydMethodClusteringOnce(input_set,attribute_index,target_num):
	pool = np.random.permutation(len(input_set)).tolist()
	seeds_index = pool[:target_num]
	seeds=[]
	for index in seeds_index:
		tmp = []
		for i in attribute_index:
			tmp.append(float(input_set[index][i]))
		seeds.append(tmp)
	# print seeds
	centers = seeds
	seeds =[]
	cluster = [[] for i in range(target_num)]
	while(seeds!=centers):
		cluster = [[] for i in range(target_num)]
		seeds = centers
		for i,instance in enumerate(input_set):
			cluster_index = clusterIndexCalculator(instance,seeds)
			cluster[cluster_index].append(i)
		centers = centerCalculator(cluster,attribute_index,input_set)
		# print centers
		# print clusters

	return (cluster,centers)


def LloydMethodClustering(input_set,attribute_index,target_num):
	(cluster,centers) = LloydMethodClusteringOnce(input_set,attribute_index,target_num)
	k_means_result = KMeansFunctionCalculator(cluster,centers,input_set,attribute_index)
	for i in range(0,100):
		(tmp_cluster, tmp_centers)=LloydMethodClusteringOnce(input_set,attribute_index,target_num)
		tmp_k_means_result = KMeansFunctionCalculator(tmp_cluster,tmp_centers,input_set,attribute_index)
		# print tmp_k_means_result
		if(tmp_k_means_result < k_means_result):
			(cluster,centers) = (tmp_cluster,tmp_centers)
			k_means_result = tmp_k_means_result

	return cluster


def hammingDistanceCalculator(cluster, raw_data,target_index):
	diff_num = 0.0
	cluster_list = [-1]*len(raw_data)

	for index,item in enumerate(cluster):
		for num in item:
			cluster_list[num] = index

	for i in range(0,len(raw_data)):
		for j in range(0,len(raw_data)):
			if i > j:
				if (cluster_list[i] == cluster_list[j] and raw_data[i][target_index] != raw_data[j][target_index]) or (cluster_list[i] != cluster_list[j] and raw_data[i][target_index] == raw_data[j][target_index]):
					diff_num = diff_num + 1
	hamming_distance = 2*diff_num/(len(raw_data)*(len(raw_data)-1))
	return hamming_distance



def SilhouetteCoefficientACalculator(cluster,input_set,attribute_index):
	distance_matrix = distanceCalculator(input_set,attribute_index)
	#print distance_matrix
	a = [0]*len(input_set)
	for item in cluster:
		for i in item:
			distance_sum = 0
			for j in item:
				distance_sum = distance_sum + distance_matrix[i][j]
			if len(item) == 1:
				a[i] = 0
			else:
				a[i] = distance_sum/(len(item)-1)
	#print a
	return a

def SilhouetteCoefficientBCalculator(cluster,input_set,attribute_index):
	distance_matrix = distanceCalculator(input_set,attribute_index)
	b = [0]*len(input_set)
	for item in cluster:
		for i in item:
			for another_item in cluster:
				if(item != another_item):
					distance_sum = 0
					for j in another_item:
						distance_sum = distance_sum + distance_matrix[i][j]
					average_distance = distance_sum/len(another_item)
					if(b[i] == 0 or b[i] > average_distance):
						b[i] = average_distance
	#print b
	return b

def SilhouetteCoefficientSCalculator(a,b):
	s = [0]*len(a)
	for i in range(0,len(s)):
		if a[i] < b[i]:
			s[i] = 1 - a[i]/b[i]
		elif a[i] > b[i]:
			s[i] = b[i]/a[i] - 1
	#print s
	return s


def SilhouetteCoefficientCalculator(cluster,raw_data,attribute_index):
	distance_matrix = distanceCalculator(raw_data,attribute_index)
	a = SilhouetteCoefficientACalculator(cluster, raw_data,attribute_index)
	b = SilhouetteCoefficientBCalculator(cluster, raw_data,attribute_index)
	s = SilhouetteCoefficientSCalculator(a,b)
	silhouette_sum = 0
	for item in s:
		silhouette_sum = silhouette_sum + item
	average_silhouette = silhouette_sum/len(s)
	return average_silhouette

def intersectionNumberCalculator(cluster_one, cluster_two):
	return len(list(set(cluster_one).intersection(set(cluster_two))))

def classificationErrorDistanceCalculator(label_clusters, clusters,raw_data):
	map_list = [[0,1,2],[0,2,1],[2,1,0],[2,0,1],[1,0,2],[1,2,0]]
	max_count = 0
	for item in map_list:
		tmp = 0
		tmp = intersectionNumberCalculator(label_clusters[item[0]],clusters[0])+intersectionNumberCalculator(label_clusters[item[1]],clusters[1])+intersectionNumberCalculator(label_clusters[item[2]],clusters[2])
		#print tmp
		if tmp > max_count:
			max_count = tmp
	#print "max"
	#print max_count
	n = len(raw_data)
	d = 1.0 - float(max_count)/n
	return d

def targetClusters(raw_data, target_index):
	cluster = [[] for i in range(3)]
	for index,instance in enumerate(raw_data):
		if instance[target_index] == 1:
			cluster[0].append(index)
		elif instance[target_index] == 2:
			cluster[1].append(index)
		else:
			cluster[2].append(index)
	return cluster





attribute_index = [0,1,2,3,4,5,6,7,8,9,10,11,12]
target_index = 13

correct = targetClusters(raw_data,target_index)
print correct

print "Result of Single Linkage:"
result = singleLinkageClustering(raw_data,attribute_index,3)
print result
print len(result[0])
print len(result[1])
print len(result[2])
hamming_distance = hammingDistanceCalculator(result,raw_data,target_index)
print "hamming distance:"
print hamming_distance
print "Silhouette Coefficient:"
s = SilhouetteCoefficientCalculator(result,raw_data,attribute_index)
print s
print "classification error:"
print classificationErrorDistanceCalculator(correct, result,raw_data)


print "Result of Complete Linkage:"
result = completeLinkageClustering(raw_data,attribute_index,3)
print result
print len(result[0])
print len(result[1])
print len(result[2])
hamming_distance = hammingDistanceCalculator(result,raw_data,target_index)
print "hamming distance:"
print hamming_distance
print "Silhouette Coefficient:"
s = SilhouetteCoefficientCalculator(result,raw_data,attribute_index)
print s
print "classification error:"
print classificationErrorDistanceCalculator(correct, result,raw_data)

print "Result of Average Linkage:"
result = averageLinkageClustering(raw_data,attribute_index,3)
print result
print len(result[0])
print len(result[1])
print len(result[2])
hamming_distance = hammingDistanceCalculator(result,raw_data,target_index)
print "hamming distance:"
print hamming_distance
print "Silhouette Coefficient:"
s = SilhouetteCoefficientCalculator(result,raw_data,attribute_index)
print s
print "classification error:"
print classificationErrorDistanceCalculator(correct, result,raw_data)

print "Result of Lloyd's Method"
result = LloydMethodClustering(raw_data,attribute_index,3)
print result
print len(result[0])
print len(result[1])
print len(result[2])
hamming_distance = hammingDistanceCalculator(result,raw_data,target_index)
print "hamming distance:"
print hamming_distance
print "Silhouette Coefficient:"
s = SilhouetteCoefficientCalculator(result,raw_data,attribute_index)
print s
print "classification error:"
print classificationErrorDistanceCalculator(correct, result,raw_data)
"""
import matplotlib.pyplot as plt

def plot2DGraph(clusters, raw_data, attribute_index):
	points = []
	for item in clusters:
		cluster_points =[]
		for attr in attribute_index:
			tmp = []
			for instance in item:
				tmp.append(float(raw_data[instance][attr]))
			cluster_points.append(tmp)
		points.append(cluster_points)
	# print points
	plt.scatter(points[2][8],points[2][6])
	plt.scatter(points[0][8],points[0][6])
	plt.scatter(points[1][8],points[1][6])
	plt.title('Wine')
	plt.xlabel('Attribute 9')
	plt.ylabel('Attribute 7')
	plt.show()
"""
#plot2DGraph(result,raw_data,attribute_index)





#plot2DGraph(correct,raw_data,attribute_index)











