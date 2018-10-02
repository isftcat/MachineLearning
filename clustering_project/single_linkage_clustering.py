from sklearn import preprocessing

file = open("iris.data")
raw_data = []
for row in file:
	row = row.strip("\r\n")
	raw_data.append(row.split(','))
raw_data.pop()
scaler = preprocessing.MinMaxScaler()
raw_data = scaler.fit_transform(raw_data)




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
	print distance_matrix
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



def clustering(input_clusters,distance_matrix):
	# print input_clusters
	combine_list = [input_clusters[0],input_clusters[1]]
	global_min_distance = minDistanceCalculator(input_clusters[0],input_clusters[1],distance_matrix)
	for one in range(0, len(input_clusters)):
		for two in range(0,len(input_clusters)):
			if(one != two):
				local_min_distance = minDistanceCalculator(input_clusters[one],input_clusters[two],distance_matrix)
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
		cluster = clustering(clusters,distance_matrix)
	return cluster



attribute_index = [0,1,2,3]
result = singleLinkageClustering(raw_data,attribute_index,3)
print result
print len(result[0])
print len(result[1])
print len(result[2])





