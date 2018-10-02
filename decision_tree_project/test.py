file = open("breast-cancer-wisconsin.data")
raw_data = []
for row in file:
	row = row.strip("\r\n")
	raw_data.append(row.split(','))

