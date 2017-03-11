import numpy as np

def hardlim(input):
	if input > 0:
		return 1
	else:
		return 0

def inner_product(vector1, vector2):
	result = np.dot(vector1, vector2)
	return result

def train_perceptron(weight, learning_rate, train_set):
	iter_num = 0
	error_num = 1
	while error_num:
		error_num = 0
		for i in range(0, len(train_set)):
			out = inner_product(weight, train_set[i][0] + [1])
			result = hardlim(out)
			e = train_set[i][1] - result
			if e == 0:
				continue
			weight = np.array(weight) + learning_rate * e * np.array(train_set[i][0] + [1])
			print "iter_num" + str(iter_num)
			print weight

			error_num += 1
			iter_num += 1
	return weight


if __name__ == '__main__':
	origin_set = [[[1, 1], 1], [[0, 2], 1], [[3, 1], 1], 
				  [[2, -1], 2], [[2, 0], 2], [[1, -2], 2],
				  [[-1, 2], 3], [[-2, 1], 3], [[-1, 1], 3]]
	train_set_1 = []
	train_set_2 = []
	train_set_3 = []
	weight_set = [[0.8, 1, -1.7], [1, 1, 1], [1, 1, 1]]


	weight_test = [0.8, 1, -1.7]
	test_set = [[[1, 1], 1], [[0, 1], 1], [[0, 0], 0], [[1, 0], 0]]

	for i in range(0, len(origin_set)):
		if origin_set[i][1] == 1:
			train_set_1.append([origin_set[i][0], 1])
		else:
			train_set_1.append([origin_set[i][0], 0])

	for i in range(0, len(origin_set)):
		if origin_set[i][1] == 2:
			train_set_2.append([origin_set[i][0], 1])
		else:
			train_set_2.append([origin_set[i][0], 0])

	for i in range(0, len(origin_set)):
		if origin_set[i][1] == 3:
			train_set_3.append([origin_set[i][0], 1])
		else:
			train_set_3.append([origin_set[i][0], 0])

	weight_set[0] = train_perceptron(weight_set[0], 1, train_set_1)
	weight_set[1] = train_perceptron(weight_set[1], 1, train_set_2)
	weight_set[2] = train_perceptron(weight_set[2], 1, train_set_3)

	print "result"
	for i in range(0, len(weight_set)):
		print weight_set[i]

	'''weight_test = train_perceptron(weight_test, 1, test_set)

	print weight_test'''

		