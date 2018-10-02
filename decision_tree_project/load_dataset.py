import numpy as np
import math
from sklearn.model_selection import train_test_split




#load dataset into numpy array
file = open("breast-cancer-wisconsin.data")
raw_data = []
for row in file:
	row = row.strip("\r\n")
	raw_data.append(row.split(','))
data = np.array(raw_data)

#shuffle and split data into train_set and test_set
train_set,test_set = train_test_split(data, train_size=0.8,random_state=42)

#define constant
attributes = ['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses']
value_set = ['1','2','3','4','5','6','7','8','9','10']
labels = ['benign','malignant']
positive = 2
negative = 4
#test
label_index = 7

#calculate positiveProportion of input_set
def positiveProportionCalculator(input_set):
	count = 0
	for instance in input_set:
		if (instance[label_index] == '2'):
			count = count + 1
	return (float(count)/len(input_set))

#calculate entropy of input_set
def entropyCalculator(input_set):
	positiveProportion = positiveProportionCalculator(input_set)
	negativeProportion = 1 - positiveProportion
	if (positiveProportion == 0 or positiveProportion == 1):
		return 0
	else:
		negativeProportion = 1 - positiveProportion
		res = - positiveProportion * math.log(positiveProportion)/math.log(2) - (negativeProportion) * math.log(negativeProportion)/math.log(2)
		return res;

#calculate the information gain of input_set when ith attribute is selected
def informationGainCalculator(input_set,attribute_index,input_value_set):
	gain = 0
	input_set_entropy = entropyCalculator(input_set)
	input_set_num = len(input_set);

	expected_entropy = 0
	for value in input_value_set:
		current_value_set =[]
		for instance in input_set:
			if (instance[attribute_index] == value) :
				current_value_set.append(instance)
		value_entropy = entropyCalculator(current_value_set)
		value_set_num = len(current_value_set)
		expected_entropy = expected_entropy + value_set_num * value_entropy / input_set_num

	gain = input_set_entropy - expected_entropy
	return gain



test = [['0001','1','1','1','1','1','1','2'],['0002','1','1','2','1','1','1','2'],['0003','2','2','2','1','1','2','4'],['0004','1','1','2','1','2','2','2'],['0005','1','1','1','2','1','1','4']]
test_value_set = ['1','2']
test_attribute_set = [1,2,3,4,5,6]

#select one attribute as node based on information gain calculation
def attributeSelection(input_set,current_attribute_set,input_value_set):
	result = 0
	max_gain = 0
	for i in current_attribute_set:
		gain = informationGainCalculator(input_set,i,input_value_set)
		if (gain > max_gain):
			result = i
			max_gain = gain
	return result

##def constructDecisionTree(input_set):
	#if (input_set == []) return []
	#decision_tree = []
	#node_index = attributeSelection(input_set, current_attribute_set, input_value_set)


