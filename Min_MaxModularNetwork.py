import os
import random
import math


def sigmod(x):
	return 1 / (1 + math.exp(-x))


def read_data(filename):
	data_set = []
	file = open(filename, 'r')
	for line in file:
		sample = [float(x) for x in line.split()]
		data_set.append(sample)
	return data_set

def getRandomNum(numrange, num):
	randSet = []
	tmpSet = [0] * numrange

	pn = random.randint(0, numrange - 1)
	randSet.append(pn)
	tmpSet[pn] = 1

	while len(randSet) < num:
		pn = random.randint(0, numrange - 1)
		if tmpSet[pn] == 1:
			continue
		else:
			tmpSet[pn] = 1
			randSet.append(pn)
	return randSet


def sperate_pos_neg(train_data, data_set_n1, data_set_n2, data_set_p1, data_set_p2):
	pos_data = []
	neg_data = []
	pos_random_index = []
	neg_random_index = []

	for i in range(0, len(train_data)):
		if (train_data[i][2] == 1):
			pos_data.append(train_data[i])
		else:
			neg_data.append(train_data[i])
	pos_random_index = getRandomNum(len(pos_data), 24)
	neg_random_index = getRandomNum(len(neg_data), 24)

	for i in range(0, len(pos_random_index)):
		data_set_p1.append(pos_data[pos_random_index[i]])
		pos_data[pos_random_index[i]] = -1
	for i in range(0, len(pos_data)):
		if (pos_data[i] != -1):
			data_set_p2.append(pos_data[i])

	for i in range(0, len(neg_random_index)):
		data_set_n1.append(neg_data[neg_random_index[i]])
		neg_data[neg_random_index[i]] = -1
	for i in range(0, len(neg_data)):
		if (neg_data[i] != -1):
			data_set_n2.append(neg_data[i])

def load_config(filename, u1, v1, bias1, u2, v2):
	config = open('config.txt', 'r')
	line = config.readline()
	temp_u1 = [float(x) for x in line.split()]
	u1 += temp_u1

	line = config.readline()
	temp_v1 = [float(x) for x in line.split()]
	v1 += temp_v1

	line = config.readline()
	temp_bias1 = [float(x) for x in line.split()]
	bias1 += temp_bias1

	line = config.readline()
	temp_u2 = [float(x) for x in line.split()]
	u2 += temp_u2

	line = config.readline()
	temp_v2 = [float(x) for x in line.split()]
	v2 += temp_v2

	line = config.readline()
	temp_bias2 = [float(x) for x in line.split()]

	config.close()

	return temp_bias2[0]

def forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, x, label):
	for i in range(0, 10):
		temp_in = u1[i] * x[0] * x[0] + v1[i] * x[0] + u1[i + 10] * x[1] * x[1] + v1[i + 10] * x[1] + bias1[i]
		hidden_in.append(temp_in)
		hidden_out.append(sigmod(temp_in))

	out = 0
	for i in range(0, 10):
		out += u2[i] * hidden_out[i] * hidden_out[i] + v2[i] * hidden_out[i]

	out += bias2

	y = sigmod(out)
	return y

def backpropagation(u1, v1, bias1, u2, v2, bias2, learining_rate, x, label):
	deri_error_u2 = []
	deri_error_v2 = []
	deri_error_bias2 = []
	deri_error_u1 = []
	deri_error_v1 = []
	deri_error_bias = 0

	hidden_in = []
	hidden_out = []
	bias = bias2

	y = forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, x, label)

	for i in range(0, len(u2)):
		temp_e_u = (y - label) * y * (1 - y) * hidden_out[i] * hidden_out[i]
		deri_error_u2.append(temp_e_u)
		u2[i] -= learining_rate * temp_e_u

		temp_e_v = (y - label) * y * (1 - y) * hidden_out[i]
		deri_error_v2.append(temp_e_v)
		v2[i] -= learining_rate * temp_e_v

	temp_e_b = (y - label) * y * (1 - y)
	bias -= learining_rate * temp_e_b

	for i in range(0, len(u2)):
		temp_e_u1 = (y - label) * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i]) * x[0] * x[0]
		temp_e_v1 = (y - label) * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i]) * x[0]
		temp_e_b1 = (y - label) * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i])

		temp_e_u1_10 = (y - label) * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i]) * x[1] * x[1]
		temp_e_v1_10 = (y - label) * y * (1 - y) * (2 * u2[i] * hidden_out[i] + v2[i]) * hidden_out[i] * (1 - hidden_out[i]) * x[1]



		u1[i] -= learining_rate * temp_e_u1
		v1[i] -= learining_rate * temp_e_v1

		u1[i + 10] -= learining_rate * temp_e_u1_10
		v1[i + 10] -= learining_rate * temp_e_v1_10
		bias1[i] -= learining_rate * temp_e_b1

	return bias

def mix_forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, x, label):
	y = [0] * 4
	hidden_in = []
	hidden_out = []
	for i in range(0, 4):
		y[i] = forward(u1[i], v1[i], bias1[i], hidden_in, hidden_out, u2[i], v2[i], bias2[i], x, label)
	result = max(min(y[0], y[1]), min(y[2], y[3]))
	return result

def draw_img(u1, v1, bias1, u2, v2, bias2, threshold, filename):
	out_set = []
	for i in range(0, 400):
		for j in range(0, 400):
			y = 0
			x1 = float(i - 200) / 50.0
			x2 = float(j - 200) / 50.0
			hidden_in = []
			hidden_out = []
			#result = forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, [x1, x2], 0)
			result = mix_forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, [x1, x2], 0)
			
			if result > threshold:
			    y = 1
			out_set.append([x1, x2, y])
			#print [x1, x2, y]
	output = open(filename, 'w')

	for i in range(0, len(out_set)):
		output.write(str(out_set[i][0]) + ' ' + str(out_set[i][1]) + ' ' + str(out_set[i][2]) + '\n')
	output.close()

def write_file(filename, data):
	output = open(filename, 'w')
	for i in range(0, len(data)):
		output.write(str(data[i]) + '\n')
	output.close()

if __name__ == '__main__':
	train_set = read_data('two_spiral_train.txt')
	test_set = read_data('two_spiral_test.txt') 
	pos_data = []
	neg_data = []

	data_set_n1 = []
	data_set_n2 = []
	data_set_p1 = []
	data_set_p2 = []

	sperate_pos_neg(train_set, data_set_n1, data_set_n2, data_set_p1, data_set_p2)

	u1 = [[], [], [], []]
	v1 = [[], [], [], []]
	bias1 = [[], [], [], []]
	u2 = [[], [], [], []]
	v2 = [[], [], [], []]
	bias2 = [0, 0, 0, 0]

	learining_rate = 0.1
	threshold = 0.5
	filename = 'config.txt'
	train_error_set = []
	train_error_step = []
	test_error_set = []
	test_error_step = []
	test_result = []
	error = 0

	for i in range(0, 4):
		bias2[i] = load_config(filename, u1[i], v1[i], bias1[i], u2[i], v2[i])
	
	for i in range(0, 5000):
		print "iteration" + str(i)
		for j in range(0, 24):
			bias2[0] = backpropagation(u1[0], v1[0], bias1[0], u2[0], v2[0], bias2[0], learining_rate, [data_set_n1[j][0], data_set_n1[j][1]], data_set_n1[j][2])	
			bias2[0] = backpropagation(u1[0], v1[0], bias1[0], u2[0], v2[0], bias2[0], learining_rate, [data_set_p1[j][0], data_set_p1[j][1]], data_set_p1[j][2])	

			bias2[1] = backpropagation(u1[1], v1[1], bias1[1], u2[1], v2[1], bias2[1], learining_rate, [data_set_n2[j][0], data_set_n2[j][1]], data_set_n2[j][2])	
			bias2[1] = backpropagation(u1[1], v1[1], bias1[1], u2[1], v2[1], bias2[1], learining_rate, [data_set_p1[j][0], data_set_p1[j][1]], data_set_p1[j][2])	

			bias2[2] = backpropagation(u1[2], v1[2], bias1[2], u2[2], v2[2], bias2[2], learining_rate, [data_set_n1[j][0], data_set_n1[j][1]], data_set_n1[j][2])	
			bias2[2] = backpropagation(u1[2], v1[2], bias1[2], u2[2], v2[2], bias2[2], learining_rate, [data_set_p2[j][0], data_set_p2[j][1]], data_set_p2[j][2])	

			bias2[3] = backpropagation(u1[3], v1[3], bias1[3], u2[3], v2[3], bias2[3], learining_rate, [data_set_n2[j][0], data_set_n2[j][1]], data_set_n2[j][2])	
			bias2[3] = backpropagation(u1[3], v1[3], bias1[3], u2[3], v2[3], bias2[3], learining_rate, [data_set_p2[j][0], data_set_p2[j][1]], data_set_p2[j][2])	

		error = 0
		for k in range(0, len(train_set)):
			hidden_in = []
			hidden_out = []
			y = mix_forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, [test_set[k][0], test_set[k][1]], test_set[k][2])
			label = train_set[k][2]
			temp_error = 0.5 * (y - label) * (y - label)
			train_error_step.append(temp_error)
			error += temp_error
		error = error / len(train_set)
		train_error_set.append(error)

		error = 0
		for k in range(0, len(test_set)):
			hidden_in = []
			hidden_out = []
			y = mix_forward(u1, v1, bias1, hidden_in, hidden_out, u2, v2, bias2, [test_set[k][0], test_set[k][1]], test_set[k][2])
			label = test_set[k][2]
			temp_error = 0.5 * (y - label) * (y - label)
			test_error_step.append(temp_error)
			error += temp_error
		error = error / len(test_set)
		test_error_set.append(error)

	write_file('test_error_set_mix.txt', test_error_set)
	write_file('train_error_set_mix.txt', train_error_set)

	filename = 'img_mixnet.txt'
	draw_img(u1, v1, bias1, u2, v2, bias2, threshold, filename)
	'''for i in range(0, 4):
		filename = "img" + str(i) + ".txt"
		draw_img(u1[i], v1[i], bias1[i], u2[i], v2[i], bias2[i], threshold, filename)'''