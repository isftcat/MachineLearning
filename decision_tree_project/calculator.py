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