from sets import Set
from collections import defaultdict

file = open("breast-cancer-wisconsin.data")
raw_data = []
for row in file:
	row = row.strip("\r\n")
	raw_data.append(row.split(','))

"""
for attribute_index in range(1,10):
	attribute_set = set([])
	print attribute_set
	for instance in raw_data:
		attribute_set.add(instance[attribute_index])
	print attribute_index
	print attribute_set
"""

value_list = []
for instance in raw_data:
	value_list.append(instance[6])

count_dict = defaultdict(int)
for value in value_list:
	count_dict[value] = count_dict[value] + 1
print count_dict