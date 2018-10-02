import math
from sklearn.model_selection import train_test_split
from collections import defaultdict

#calculate positiveProportion of input_set
def proportionCalculator(input_set,label):
	count = 0
	for instance in input_set:
		if (instance[label_index] == label):
			count = count + 1
	return (float(count)/len(input_set))

#calculate entropy of input_set
def entropyCalculator(input_set):
	res = 0
	for label in labels:
		proportion = proportionCalculator(input_set,label)
		if (proportion != 0):
			res = res - proportion * math.log(proportion)/math.log(2)
	return res

def missingAttributeValueHandler(input_set,label,attribute_index):
	value_list = []
	for instance in input_set:
		if (instance[label_index] == label):
			value_list.append(instance[attribute_index])
	count_dict = defaultdict(int)
	selected_value = value_list[0]
	for value in value_list:
		count_dict[value] = count_dict[value] + 1
		if (count_dict[value] > count_dict[selected_value]):
			selected_value = value
	return selected_value


#calculate the information gain of input_set when ith attribute is selected
def informationGainCalculator(input_set,attribute_index,input_value_set):
	gain = 0
	input_set_entropy = entropyCalculator(input_set)
	input_set_num = len(input_set)

	expected_entropy = 0
	for value in input_value_set:
		current_value_set =[]
		for instance in input_set:
			
			if (instance[attribute_index] == '?'):
				instance[attribute_index] = missingAttributeValueHandler(input_set,instance[label_index],attribute_index)
			if (instance[attribute_index] == value) :
				current_value_set.append(instance)
		
		value_set_num = len(current_value_set)
		if (value_set_num == 0):
			value_entropy =0
		else:
			value_entropy = entropyCalculator(current_value_set)
		expected_entropy = expected_entropy + value_set_num * value_entropy / input_set_num

	gain = input_set_entropy - expected_entropy
	return gain



#select one attribute as node based on information gain calculation
def attributeSelection(input_set,current_attribute_set,input_value_set):
	
	result = current_attribute_set[0]
	max_gain = 0
	for i in current_attribute_set:
		gain = informationGainCalculator(input_set,i,input_value_set)
		if (gain > max_gain):
			result = i
			max_gain = gain
	return result




class TreeNode:
	def __init__(self, value, word):
		self.value = value
		self.word = word
		self.children = []

	def addChild(self,node):
		self.children.append(node)

	def printTree(self, level=0):
		print '\t' * level + repr(self.word)
		for child in self.children:
			child.printTree(level+1)


def majorityVoting(input_set):
	label_count_dict = defaultdict(int)
	res = input_set[0][label_index]
	for instance in input_set:
		label_count_dict[instance[label_index]] += 1
		if(label_count_dict[res] < label_count_dict[instance[label_index]]):
			res = instance[label_index]
	return res




def constructDecisionTree(input_set, current_attribute_set, input_value_set,attributes):
	if (input_set == []):
		root = TreeNode(0,"unknown")
		return root

	if (current_attribute_set == []):
		leaf = majorityVoting(input_set)
		for label in labels:
			if(leaf == label):
				root = TreeNode(0,leaf)
		return root

	for label in labels:
		if(proportionCalculator(input_set,label) >= 0.9):
			root = TreeNode(0,label)
			return root

	
	attribute_index = attributeSelection(input_set, current_attribute_set, input_value_set)
	
	current_attribute_set.remove(attribute_index)

	root = TreeNode(attribute_index,attributes[attribute_index-1])
	for i in input_value_set:
		output_set = []
		for instance in input_set:
			if (instance[attribute_index] == i):
				output_set.append(instance)
		root.addChild(constructDecisionTree(output_set,current_attribute_set,input_value_set,attributes));
	current_attribute_set.append(attribute_index)
	return root

def missingAttributeValueClassifier(root, instance):
	l = len(instance)
	derived_set = []
	for i in range(0,l):
		if(instance[i] == '?'):
			tmp = instance
			for value in value_set:
				tmp[i] = value
				derived_set.append(tmp)
	results_for_instance = []
	for item in derived_set:
		results_for_instance.append(classifier(root,item))
	res = results_for_instance[0]
	label_count_dict = defaultdict(int)
	for item in results_for_instance:
		label_count_dict[item] += 1
		if(label_count_dict[item] > label_count_dict[res]):
			res = item
	return item

def classifier(root,instance):
	if(root.children ==[]):
		return root.word
	
	if (instance[root.value] == '?'):
		return missingAttributeValueClassifier(root, instance);# change it latter
		#attribute_value = int('1')
	else:
		attribute_value = int(instance[root.value])

	return classifier(root.children[attribute_value],instance)

def accuracyCalculator(root,input_set):
	total_num = len(input_set)
	correct_num = 0
	for instance in input_set:
		label = classifier(root,instance)
		if (label == instance[label_index]):
			correct_num = correct_num + 1
	return float(correct_num)/total_num

def predict(root,input_set):
	result = []
	for instance in input_set:
		result.append(classifier(root,instance))
	print result

def randomForestsClassifier(forest,instance):
	potential_label = []
	for tree in forest:
		potential_label.append(classifier(tree,instance))
	res = potential_label[0]
	label_count_dict = defaultdict(int)
	for item in potential_label:
		label_count_dict[item] += 1
		if(label_count_dict[item] > label_count_dict[res]):
			res = item
	return res

def randomForestsAccuracyCalculator(root,input_set):
	total_num = len(input_set)
	correct_num = 0
	for instance in input_set:
		label = randomForestsClassifier(root,instance)
		if (label == instance[label_index]):
			correct_num = correct_num + 1
	return float(correct_num)/total_num


def postPruning(root,validation_set):
	validation_set_accuracy_before = accuracyCalculator(root,validation_set)
	stack=[]
	layer=[]
	nextlayer=[]
	layer.append(root)
	while(layer):
		for node in layer:
			for child in node.children:
				if(child.value != 0):
					nextlayer.append(child)
					stack.append(child)
		layer=[]
		layer=nextlayer
		nextlayer = []

	for i in stack[::-1]:
		value_back_up = i.value
		word_back_up = i.word
		children_back_up = i.children
		for j in labels:
			i.value = 0
			i.word = j
			i.children = []
			if(accuracyCalculator(root,validation_set) > validation_set_accuracy_before):
				validation_set_accuracy_before = accuracyCalculator(root,validation_set)
				value_back_up = i.value
				word_back_up = i.word
				children_back_up = []
		i.value = value_back_up
		i.word = word_back_up
		i.children = children_back_up

	return root




#test = [['0001','1','1','1','1','1','1','2'],['0002','1','1','2','1','1','1','2'],['0003','2','2','2','1','1','2','4'],['0004','1','1','2','1','2','2','2'],['0005','1','1','1','2','1','1','4']]
#test_value_set = ['1','2']
#test_attribute_set = [1,2,3,4,5,6]
#attributes = ['None','Sky','Air-Temp','Humidity','Wind','Water','Forecast']
#positive = 2
#negative = 4
#label_index = 7

#load dataset into numpy array
file = open("letter-recognition.data")
raw_data = []
for row in file:
	row = row.strip("\r\n")
	raw_data.append(row.split(','))
#data = np.array(raw_data)

#shuffle and split data into train_set and test_set
train_set,test_set = train_test_split(raw_data, train_size=0.8,random_state=32)
train_set,validation_set = train_test_split(train_set, train_size=0.9,random_state=42)
"""
#define constant
attributes = ['Left_Weight','Left-Distance','Right-Weight','Right-Distance']
attribute_set = [1,2,3,4]
value_set = ['1','2','3','4','5']
labels = ['L','B','R']
label_index = 0
"""
attributes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
attribute_set = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
value_set = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
label_index = 0

"""
train_set_one, train_set_left = train_test_split(train_set, train_size=0.2,random_state=42)
train_set_two, train_set_left = train_test_split(train_set_left,train_size=0.25,random_state=42)
train_set_three, train_set_left = train_test_split(train_set_left,train_size=1/3,random_state=42)
train_set_four, train_set_five = train_test_split(train_set_left,train_size=0.5,random_state=42)
"""
#tree_one = constructDecisionTree(train_set_one,attribute_set,value_set,attributes)
#tree_one.printTree(0)

#tree_two = constructDecisionTree(train_set_two,attribute_set,value_set,attributes)
#tree_two.printTree(0)
"""

tree_one = constructDecisionTree(train_set_one,attribute_set,value_set,attributes)
tree_two = constructDecisionTree(train_set_two,attribute_set,value_set,attributes)
tree_three = constructDecisionTree(train_set_three,attribute_set,value_set,attributes)
tree_four = constructDecisionTree(train_set_four,attribute_set,value_set,attributes)
tree_five = constructDecisionTree(train_set_five,attribute_set,value_set,attributes)
forest = []
forest.append(tree_one)
forest.append(tree_two)
forest.append(tree_three)
forest.append(tree_four)
forest.append(tree_five)

for i in range(0,5):
	print "tree"
	print i
	print forest[i].printTree(0)

print "accuracy for training set:"
print randomForestsAccuracyCalculator(forest,train_set)
print "accuracy for testing set:"
print randomForestsAccuracyCalculator(forest,test_set)




"""

result = constructDecisionTree(train_set,attribute_set,value_set,attributes)

print "Decision Tree:"

result.printTree(0)

print "accuracy for training set before post-pruning:"
print accuracyCalculator(result,train_set)
print "accuracy for validation set set before post-pruning:"
print accuracyCalculator(result,validation_set)
print "accuracy for testing set before post_pruning:"
print accuracyCalculator(result,test_set)

result_after = postPruning(result,validation_set)

result_after.printTree(0)
print "accuracy for training set after post-pruning:"
print accuracyCalculator(result_after,train_set)
print "accuracy for validation set set before post-pruning:"
print accuracyCalculator(result_after,validation_set)
print "accuracy for testing set after post_pruning:"
print accuracyCalculator(result_after,test_set)
