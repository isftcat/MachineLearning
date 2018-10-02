"""
def distanceCalculator(input_set,attribute_index):
	input_num = len(input_set)
	distance_matrix = [([0.0]*input_num) for j in range(input_num)]
	for i in range(0, input_num):
		for j in range(0, input_num):
			distance = 0.0
			for index in attribute_index:
				
				# print i
				# print input_set[i][index]
				# print j
				# print input_set[j][index]
				
				diff = float(input_set[i][index]) - float(input_set[j][index])
				distance = distance + pow(diff,2)
			distance = pow(distance,0.5)
			distance_matrix[i][j] = distance
	# print distance_matrix
	return distance_matrix

def SilhouetteCoefficientACalculator(cluster,input_set,attribute_index):
	distance_matrix = distanceCalculator(input_set,attribute_index)
	print distance_matrix
	a = [0]*len(input_set)
	for item in cluster:
		for i in item:
			distance_sum = 0
			for j in item:
				distance_sum = distance_sum + distance_matrix[i][j]
			a[i] = distance_sum/(len(item)-1)
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

	return b

def SilhouetteCoefficientACalculator(cluster,input_set,attribute_index):
	distance_matrix = distanceCalculator(input_set,attribute_index)
	#print distance_matrix
	a = [0]*len(input_set)
	for item in cluster:
		for i in item:
			distance_sum = 0
			for j in item:
				distance_sum = distance_sum + distance_matrix[i][j]
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



 
cluster = [[1,4,6,10],[0,2,3,5],[7,8,9]]
test_data = [[23,4],[0,1],[24,4],[23,3],[1,1],[24,3],[0,0],[0,24],[1,24],[0,23],[1,0]]
test_attribute=[0,1]
s = SilhouetteCoefficientCalculator(cluster,test_data,test_attribute)
print s

"""
"""
file = open("seeds_dataset.txt")
raw_data = []
for row in file:
	row = row.strip("\r\n")
	raw_data.append(row.split())
# print raw_data


print float(raw_data[7][0])
"""
# from xlrd import open_workbook
"""
import re

def readsheet(s,row_count = -1,col_count = -1):
	nrows = s.nrows
	ncols = s.ncols
	row_count = (row_count if row_count > 0 else nrows)
	col_count = (col_count if col_count > 0 else ncols)
	row_index = 0
	while row_index < row_count:
		yield [s.cell(row_index,col).value for col in xrange(col_count)]
		row_index += 1

wb = open_workbook('data_user_modeling.xlsx')
a = []
for s in wb.sheets():
	for row in readsheet(s,10,6):
		a.append(row)

print a

"""
"""
data = open_workbook('data_user_modeling.xlsx')
table = data.sheets()[0]
raw_data = []
for i in range(1,259):
	raw_data.append(table.row_values(i))
print raw_data
"""
file = open("breast-cancer-wisconsin.data")
raw_data = []
for row in file:
	row = row.strip("\r\n")
	raw_data.append(row.split(','))
for item in raw_data:
	del item[0]
miss =[]
for index, item in numerate(raw_data):
	if item[5] == '?':
		item[5] = '1'


print raw_data